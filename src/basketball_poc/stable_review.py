"""Original-media review/export adapter for the frozen stable triage results."""
import json
from pathlib import Path
import re

from .media import export_clips, inspect_video, read_json, sha256, write_json
from .core import number
from . import source_timeline

ROOT = Path(__file__).resolve().parents[2]
VERSION = 'stable-review-v3'
LEGACY_VERSION = 'stable-review-v2'
SELECTION_VERSION = 'stable-selection-v2'


def verify_hashes(files):
    for path, digest in files.items():
        if sha256(path) != digest:
            raise ValueError(f'Evidence changed: {path}')


def frozen_method():
    path = ROOT/'configs/stable-workflow-baseline.json'
    baseline = read_json(path)
    verify_hashes({ROOT/p: h for p, h in baseline['files_sha256'].items()})
    return baseline, path


def load_triage(folder):
    folder = Path(folder).resolve(strict=True)
    result = read_json(folder/'results.json')
    if result['status'] != 'completed' or not result['supported_bytes_unchanged']:
        raise ValueError('Completed stable triage required')
    frozen = read_json(folder/'frozen.json')['inputs_sha256']
    outputs = read_json(folder/'predictions-before-evaluation.json')['files_sha256']
    verify_hashes(frozen);verify_hashes(outputs)
    analysis = Path(result['analysis']).resolve(strict=True)
    original = read_json(analysis/'review-events.json')
    groups = {kind: read_json(folder/f'{kind}-events.json') for kind in ('supported','uncertain','excluded')}
    all_events = [e for group in groups.values() for e in group['events']]
    by_id = {e['id']: e for e in all_events}
    if len(by_id) != len(all_events) or len(original['events']) != len(all_events):
        raise ValueError('Duplicate or missing classified events')
    raw = {e['id']: e for e in original['events']}
    if set(by_id) != set(raw) or any(any(by_id[i].get(k) != v for k,v in e.items()) for i,e in raw.items()):
        raise ValueError('Classified events differ from original candidates')
    classified = {e['id']:e for kind in ('supported','uncertain')
                  for e in read_json(analysis/f'{kind}-events.json')['events']}
    if by_id != classified:
        raise ValueError('Triage differs from authoritative net classification')
    original = {**original, 'events':all_events}
    if any(g['source_sha256'] != original['source_sha256'] for g in groups.values()):
        raise ValueError('Category source mismatch')
    candidate = read_json(analysis/'candidates/manifest.json')
    if candidate['source_sha256'] != original['source_sha256']:
        raise ValueError('Analysis source mismatch')
    source = candidate['source']['path']
    source_hash = candidate['source_sha256']
    mapping_paths = []
    mapping = analysis.parent/'result.json'
    if mapping.exists():
        mapped = read_json(mapping)
        verification = analysis.parent/'verification.json'
        check = read_json(verification)
        if (mapped['status'] != 'completed' or Path(mapped['pipeline']).resolve() != analysis
                or check['status'] != 'passed' or not check['frame_and_time_mapping_verified']):
            raise ValueError('Verified original-video mapping required')
        source, source_hash = mapped['original_video'], mapped['original_source_sha256']
        mapping_paths = [mapping, verification]
    info = inspect_video(source)
    if sha256(source) != source_hash:
        raise ValueError('Original source changed')
    for key in ('fps','duration_seconds'):
        if abs(info[key]-candidate['source'][key]) > 1e-6:
            raise ValueError('Original and analysis timebase differ')
    paths = [folder/'results.json', folder/'frozen.json', folder/'predictions-before-evaluation.json',
             analysis/'review-events.json', analysis/'candidates/manifest.json', *mapping_paths]
    evidence = {**frozen, **outputs, **{str(p):sha256(p) for p in paths}}
    return info, source_hash, original, groups, evidence


def prepare(folder, output, preview_mode='timeline'):
    if preview_mode not in ('timeline', 'clips'):
        raise ValueError('Unknown preview mode')
    baseline, baseline_path = frozen_method()
    info, source_hash, original, groups, evidence = load_triage(folder)
    evidence[str(baseline_path)] = sha256(baseline_path)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    original = {**original, 'analysis_source_sha256':original['source_sha256'], 'source_sha256':source_hash}
    ordered = sorted(original['events'], key=lambda e:e['time_seconds'])
    original['events'] = ordered
    write_json(output/'original-events.json', original)
    category = {e['id']:kind for kind, group in groups.items() for e in group['events']}
    visible = [e for e in ordered if category[e['id']] != 'excluded']
    preview = source_timeline.prepare_media(info, source_hash, output) if preview_mode == 'timeline' else None
    clips = export_clips(info['path'], {**original, 'events':visible}, output/'clips', before=4, after=2) if preview is None else None
    candidates = []
    for i, event in enumerate(visible):
        kind = category[event['id']]
        candidate = {'id':event['id'],'time_seconds':event['time_seconds'],'category':kind,
                     'initial_state':{'supported':'keep','uncertain':'pending'}[kind]}
        if preview is not None:
            start, end = source_timeline.interval(event['time_seconds'], info['duration_seconds'])
            candidate.update(segment_start_seconds=start, segment_end_seconds=end)
        else:
            clip = clips['clips'][i]
            candidate.update(clip='clips/'+clip['file'], clip_sha256=sha256(output/'clips'/clip['file']),
                             clip_start_seconds=clip['start_seconds'])
        candidates.append(candidate)
    manifest = {'version':VERSION if preview is not None else LEGACY_VERSION,'status':'ready_for_user_review','browser_export_enabled':True,
        'source':info,'source_sha256':source_hash,'baseline_commit':baseline['commit'],
        'input_triage':str(Path(folder).resolve()),'evidence_sha256':evidence,
        'original_events_sha256':sha256(output/'original-events.json'),'candidates':candidates,
        'automatic_metrics_modified':False,'labels_used':False}
    if preview is not None:
        manifest['preview'] = preview
    verify_hashes(evidence)
    write_json(output/'review-manifest.json', manifest)
    (output/'index.html').write_text(render_review(manifest, sha256(output/'review-manifest.json')), encoding='utf-8')
    return {'pack':str(output),'counts':{k:len(v['events']) for k,v in groups.items()}}


def verify_review(root):
    root = Path(root).resolve(strict=True)
    manifest = read_json(root/'review-manifest.json')
    if manifest['version'] not in (VERSION, LEGACY_VERSION) or manifest['status'] != 'ready_for_user_review':
        raise ValueError('Ready stable review required')
    baseline, _ = frozen_method()
    if manifest['baseline_commit'] != baseline['commit']:
        raise ValueError('Baseline mismatch')
    verify_hashes(manifest['evidence_sha256'])
    if sha256(manifest['source']['path']) != manifest['source_sha256']:
        raise ValueError('Original media changed')
    if sha256(root/'original-events.json') != manifest['original_events_sha256']:
        raise ValueError('Candidate snapshot changed')
    original = read_json(root/'original-events.json')
    # Reconstruct categories from the authoritative immutable triage snapshot.
    groups = {k:read_json(Path(manifest['input_triage'])/f'{k}-events.json')['events']
              for k in ('supported','uncertain','excluded')}
    expected = sorted([(e['id'],e['time_seconds'],kind) for kind,events in groups.items() for e in events],key=lambda r:r[1])
    actual = [(c['id'],c['time_seconds'],c['category']) for c in manifest['candidates']]
    if actual != [r for r in expected if r[2]!='excluded'] or [(e['id'],e['time_seconds']) for e in original['events']] != [r[:2] for r in expected]:
        raise ValueError('Pack category or time changed')
    for candidate in manifest['candidates']:
        if manifest['version'] == VERSION:
            start, end = source_timeline.interval(candidate['time_seconds'], manifest['source']['duration_seconds'])
            if candidate.get('segment_start_seconds') != start or candidate.get('segment_end_seconds') != end:
                raise ValueError('Segment boundaries changed')
            continue
        clip = Path(candidate['clip'])
        if clip.is_absolute() or len(clip.parts)!=2 or clip.parts[0]!='clips' or '..' in clip.parts:
            raise ValueError('Invalid clip path')
        if sha256(root/clip) != candidate['clip_sha256']:
            raise ValueError('Clip changed')
    if manifest['version'] == VERSION:
        preview = manifest['preview']
        if preview['kind'] == 'original' and (preview['sha256'] != manifest['source_sha256']
                or preview['mime_type'] not in ('video/mp4', 'video/webm')):
            raise ValueError('Original preview identity changed')
        path, _, digest = source_timeline.resource(root, manifest)
        if sha256(path) != digest:
            raise ValueError('Shared preview media changed')
    return manifest, original, sha256(root/'review-manifest.json')


def render_review(manifest, digest, include_players=False):
    payload = json.dumps({'manifest': manifest, 'manifest_sha256': digest,
                          'preview': source_timeline.playback_data(manifest)},
                         ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    template = Path(__file__).with_name('stable_review.html').read_text(encoding='utf-8')
    return template.replace('/*REVIEW_DATA*/null', payload).replace(
        '/*PLAYERS_ENABLED*/false', 'true' if include_players else 'false').replace(
        '/*PLAYER_SCRIPT*/', Path(__file__).with_name('player_review.js').read_text(encoding='utf-8')).replace(
        '/*GOAL_EDIT_SCRIPT*/', Path(__file__).with_name('goal_review_edit.js').read_text(encoding='utf-8'))


def preview_resource(root, manifest):
    return source_timeline.resource(root, manifest)


def selection_events(manifest, original, selection, digest, actor='user'):
    if actor not in ('user','integration_test'):
        raise ValueError('Unknown actor')
    if selection.get('version') != SELECTION_VERSION or selection.get('review_manifest_sha256') != digest:
        raise ValueError('Selection belongs to another pack')
    choices = selection.get('choices')
    ids = {c['id'] for c in manifest['candidates']}
    if not isinstance(choices,dict) or not set(choices)<=ids or any(type(v) is not bool for v in choices.values()):
        raise ValueError('Unknown candidate or invalid choice')
    if any(c['category']=='uncertain' and c['id'] not in choices for c in manifest['candidates']):
        raise ValueError('Pending candidates require explicit decisions')
    if original['source_sha256'] != manifest['source_sha256']:
        raise ValueError('Source mismatch')
    duration = manifest['source']['duration_seconds']
    def bounds(raw, time):
        if not isinstance(raw, dict) or set(raw) != {'start_seconds', 'end_seconds'}:
            raise ValueError('片段需要开始时间和结束时间')
        start = number(raw['start_seconds'], '开始时间')
        end = number(raw['end_seconds'], '结束时间')
        if not 0 <= start <= time < end <= duration or start == end:
            raise ValueError('片段范围必须在视频内，并包含进球时刻')
        return {'segment_start_seconds': start, 'segment_end_seconds': end}
    overrides = selection.get('segment_overrides', {})
    if not isinstance(overrides, dict) or not set(overrides) <= ids:
        raise ValueError('Unknown candidate range')
    ranges = {c['id']: bounds(overrides[c['id']], c['time_seconds'])
              for c in manifest['candidates'] if c['id'] in overrides}
    additions = selection.get('manual_events', [])
    if not isinstance(additions, list) or len(additions) > 1000:
        raise ValueError('Invalid manual events')
    manual, seen = [], set(ids)
    for raw in additions:
        if not isinstance(raw, dict) or set(raw) != {'id', 'time_seconds', 'start_seconds', 'end_seconds'}:
            raise ValueError('Invalid manual event')
        identifier = raw['id']
        if not isinstance(identifier, str) or not re.fullmatch(r'manual-[a-zA-Z0-9-]{1,64}', identifier) or identifier in seen:
            raise ValueError('Invalid or duplicate manual event id')
        seen.add(identifier)
        time = number(raw['time_seconds'], '进球时刻')
        interval = bounds({k: raw[k] for k in ('start_seconds', 'end_seconds')}, time)
        manual.append({'id': identifier, 'time_seconds': time, **interval,
                       'status': 'selected_for_export', 'selection_source': 'user_added' if actor == 'user' else actor,
                       'automatic_category': None, 'explicit_choice': True})
    events = {e['id']:e for e in original['events']}
    selected, decisions = [], []
    for c in manifest['candidates']:
        identifier = c['id']
        explicit = identifier in choices
        keep = choices.get(identifier, c['category']=='supported')
        decision = 'integration_test' if explicit and actor=='integration_test' else 'user_choice' if explicit else 'automatic_default'
        decisions.append({'id':identifier,'category':c['category'],'selected':keep,'explicit_choice':explicit,'decision_source':decision})
        if keep:
            selected.append({**events[identifier],'status':'selected_for_export','selection_source':decision,
                             'automatic_category':c['category'],'explicit_choice':explicit,
                             **ranges.get(identifier, {})})
    selected.extend(manual)
    selected.sort(key=lambda e:e['time_seconds'])
    if len({e['time_seconds'] for e in selected}) != len(selected):
        raise ValueError('同一进球时刻已有片段，请修改已有片段')
    return {**original,'status':'selected_for_export','events':selected,
        'review':{'selection_actor':actor,'review_manifest_sha256':digest,'decisions':decisions,
                  'manual_event_ids':[e['id'] for e in manual], 'segment_overrides':overrides,
                  'automatic_metrics_modified':False,'scope':'Export selection; not model accuracy or complete ground truth'}}


def apply_selection(root, selection_file, output, actor='user'):
    manifest, original, digest = verify_review(root)
    selected = selection_events(manifest,original,read_json(selection_file),digest,actor)
    output = Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    write_json(output/'selected-events.json',selected)
    write_json(output/'selection-record.json',{'selection':read_json(selection_file),
        'selection_sha256':sha256(selection_file),'selection_actor':actor,'automatic_metrics_modified':False,
        'source_sha256':manifest['source_sha256'],'baseline_commit':manifest['baseline_commit']})
    from .review_export import export_clips as export_reviewed_clips
    result = export_reviewed_clips(manifest['source']['path'],selected,output/'clips',before=4,after=2)
    return {'selected_count':len(selected['events']),
            'montage':str(output/'clips'/result['montage']) if result['montage'] else None}
