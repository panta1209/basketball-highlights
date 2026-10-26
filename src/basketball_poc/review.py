"""Local candidate review, with explicit user selection kept separate from inference."""
from datetime import datetime, timezone
import json
from pathlib import Path

from .core import timestamps
from .media import export_clips, inspect_video, read_json, sha256, write_json

VERSION = 'candidate-review-v1'
SELECTION_VERSION = 'candidate-selection-v1'


def validate_analysis_artifacts(run, manifest):
    """Review the completed method's own identity; moving runs have no static scene."""
    run = Path(run)
    if manifest.get('backend') in ('moving-roi-v1','moving-scale-v1'):
        from .moving_roi import validate_roi_cache
        count = manifest['processed_frames']
        if count != round(manifest['source']['duration_seconds'] * manifest['source']['fps']):
            raise ValueError('Incomplete moving source coverage')
        for name in ('detections', 'geometry', 'coverage'):
            suffix = '.json' if name == 'coverage' else '.jsonl'
            if sha256(run/(name+suffix)) != manifest[name+'_sha256']:
                raise ValueError('Moving observation/coverage changed')
        for name, digest in manifest['implementation_sha256'].items():
            if sha256(run/Path(name).name) != digest:
                raise ValueError('Moving method snapshot changed')
        rows = [json.loads(l) for l in (run/'detections.jsonl').read_text().splitlines()]
        geometry = [json.loads(l) for l in (run/'geometry.jsonl').read_text().splitlines()]
        if manifest['backend']=='moving-scale-v1':
            from .moving_scale import validate_scale_cache
            validate_scale_cache(manifest,rows,geometry,manifest,count,manifest['moving_profile'],
                manifest['roi_profile'],manifest['scale_profile'],manifest['implementation_sha256'])
        else:
            validate_roi_cache(manifest, rows, geometry, manifest, count, manifest['moving_profile'],
                               manifest['roi_profile'], manifest['implementation_sha256'])
        coverage = read_json(run/'coverage.json')
        if sum(coverage['frame_counts'].values()) != count:
            raise ValueError('Moving coverage counts differ')
    elif sha256(run/'scene.json') != manifest['scene_sha256'] or sha256(manifest['profile_path']) != manifest['profile_sha256']:
        raise ValueError('Automatic scene/profile changed')


def selection_events(manifest, original, selection, manifest_hash, actor='user'):
    if actor not in ('user', 'integration_test'):
        raise ValueError('Unknown selection actor')
    if selection.get('version') != SELECTION_VERSION or selection.get('review_manifest_sha256') != manifest_hash:
        raise ValueError('Selection belongs to a different review manifest')
    if selection.get('complete') is not True:
        raise ValueError('All candidates must be explicitly reviewed')
    ids = [c['id'] for c in manifest['candidates']]
    reviewed, selected = selection.get('reviewed_ids'), selection.get('selected_ids')
    for values in (reviewed, selected):
        if not isinstance(values, list) or any(not isinstance(v, str) for v in values) or len(set(values)) != len(values):
            raise ValueError('Unique string candidate IDs required')
    if len(ids) != len(set(ids)) or set(reviewed) != set(ids) or not set(selected) <= set(reviewed):
        raise ValueError('Unknown IDs or pending candidates')
    if original['source_sha256'] != manifest['source_sha256']:
        raise ValueError('Candidate source identity differs')
    events = {e['id']: e for e in original['events']}
    if len(events) != len(original['events']):
        raise ValueError('Duplicate original event IDs')
    expected = [(f'candidate-{i:04d}', e['id'], e['time_seconds'])
                for i, e in enumerate(sorted(original['events'], key=lambda e: e['time_seconds']), 1)]
    actual = [(c['id'], c['original_event_id'], c['time_seconds']) for c in manifest['candidates']]
    if actual != expected:
        raise ValueError('Review must cover every authoritative original candidate exactly once')
    chosen = []
    for candidate in manifest['candidates']:
        event = events[candidate['original_event_id']]
        if candidate['time_seconds'] != event['time_seconds']:
            raise ValueError('Review time differs from authoritative original event')
        if candidate['id'] in selected:
            chosen.append({**event, 'status': 'user_selected' if actor == 'user' else 'test_selected',
                           'decision': 'user_selected' if actor == 'user' else 'test_selected',
                           'confirmation': 'user_selection' if actor == 'user' else 'integration_test',
                           'review_candidate_id': candidate['id']})
    chosen.sort(key=lambda e: e['time_seconds'])
    return {**original, 'status': 'user_selected_candidates' if actor == 'user' else 'test_selected_candidates', 'events': chosen,
            'review': {'manifest_sha256': manifest_hash, 'selection_version': SELECTION_VERSION,
                      'selection_actor': actor,
                      'reviewed_count': len(reviewed), 'selected_count': len(selected),
                      'scope': ('User selected candidates' if actor == 'user' else 'Simulated integration selection')+'; not complete ground truth or model confirmation'}}


def render_page(manifest, manifest_hash):
    data = {'manifest': manifest, 'manifest_sha256': manifest_hash}
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    return Path(__file__).with_name('review.html').read_text(encoding='utf-8').replace('/*REVIEW_DATA*/null', payload)


def prepare_review(run, output, browser_export=False):
    run = Path(run).resolve(strict=True)
    m, events = read_json(run/'manifest.json'), read_json(run/'events.json')
    if m['status'] != 'completed':
        raise ValueError('Completed automatic run required')
    info = inspect_video(m['source']['path'])
    if info != m['source'] or sha256(info['path']) != m['source_sha256'] or events['source_sha256'] != m['source_sha256']:
        raise ValueError('Source identity changed')
    if m.get('candidate_count') != len(events['events']):
        raise ValueError('Candidate count mismatch')
    timestamps(events['events'], 'events')
    if len({e['id'] for e in events['events']}) != len(events['events']):
        raise ValueError('Duplicate original event IDs')
    validate_analysis_artifacts(run, m)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/'original-events.json').write_bytes((run/'events.json').read_bytes())
    try:
        clips = export_clips(info['path'], events, output/'clips', before=4, after=2)
        ordered = sorted(events['events'], key=lambda e: e['time_seconds'])
        candidates = []
        for index, (event, clip) in enumerate(zip(ordered, clips['clips']), 1):
            path = output/'clips'/clip['file']
            candidates.append({'id': f'candidate-{index:04d}', 'original_event_id': event['id'],
                               'time_seconds': event['time_seconds'], 'initial_state': 'pending',
                               'clip': 'clips/'+clip['file'], 'clip_sha256': sha256(path),
                               'clip_start_seconds': clip['start_seconds'], 'clip_end_seconds': clip['end_seconds']})
        manifest = {'version': VERSION, 'status': 'ready_for_user_review',
                    'created_at_utc': datetime.now(timezone.utc).isoformat(),
                    'input_run': str(run), 'input_manifest_sha256': sha256(run/'manifest.json'),
                    'original_events_sha256': sha256(output/'original-events.json'),
                    'source': info, 'source_sha256': m['source_sha256'],
                    'candidates': candidates, 'annotations_used': False,
                    'preselected_count': 0, 'automatic_metrics_modified': False,
                    'browser_export_enabled': browser_export,
                    'analysis_method': m.get('backend', m.get('event_mode', 'automatic')),
                    'coverage': read_json(run/'coverage.json') if m.get('backend') in ('moving-roi-v1','moving-scale-v1') else None,
                    'scope': 'Candidate review cannot recover undetected goals or prove complete ground truth'}
        write_json(output/'review-manifest.json', manifest)
        page = render_page(manifest, sha256(output/'review-manifest.json'))
        with (output/'index.html').open('x', encoding='utf-8') as file:
            file.write(page)
        return {'status': manifest['status'], 'candidate_count': len(candidates), 'page': str(output/'index.html')}
    except Exception as error:
        write_json(output/'failure.json', {'status': 'failed', 'reason': str(error)})
        raise


def verify_review(root):
    root = Path(root).resolve(strict=True)
    manifest = read_json(root/'review-manifest.json')
    if manifest['version'] != VERSION or manifest['status'] != 'ready_for_user_review':
        raise ValueError('Ready review pack required')
    if sha256(root/'original-events.json') != manifest['original_events_sha256']:
        raise ValueError('Original candidate snapshot changed')
    if sha256(manifest['source']['path']) != manifest['source_sha256']:
        raise ValueError('Original media changed')
    run = Path(manifest['input_run'])
    if sha256(run/'manifest.json') != manifest['input_manifest_sha256'] or sha256(run/'events.json') != manifest['original_events_sha256']:
        raise ValueError('Input automatic run changed')
    original_manifest = read_json(run/'manifest.json')
    if manifest['source'] != original_manifest['source'] or manifest['source_sha256'] != original_manifest['source_sha256']:
        raise ValueError('Review source differs from automatic run')
    validate_analysis_artifacts(run, original_manifest)
    for candidate in manifest['candidates']:
        # Clip names are generated internally and cannot select arbitrary files.
        clip = Path(candidate['clip'])
        if clip.is_absolute() or len(clip.parts) != 2 or clip.parts[0] != 'clips' or '..' in clip.parts:
            raise ValueError('Invalid review clip path')
        if sha256(root/clip) != candidate['clip_sha256']:
            raise ValueError('Review clip changed')
    return manifest, read_json(root/'original-events.json'), sha256(root/'review-manifest.json')


def apply_selection(root, selection_file, output, actor='user'):
    manifest, original, digest = verify_review(root)
    selection = read_json(selection_file)
    confirmed = selection_events(manifest, original, selection, digest, actor)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'selected-events.json', confirmed)
    write_json(output/'selection-record.json', {'selection_sha256': sha256(selection_file),
               'selection': selection, 'review_manifest_sha256': digest,
               'selection_actor': actor,
               'automatic_metrics_modified': False, 'source_sha256': manifest['source_sha256']})
    result = export_clips(manifest['source']['path'], confirmed, output/'clips', before=4, after=2)
    return {'status': 'user_selection_exported' if actor == 'user' else 'test_selection_exported', 'selected_count': len(confirmed['events']),
            'montage': str(output/'clips'/result['montage']) if result['montage'] else None}
