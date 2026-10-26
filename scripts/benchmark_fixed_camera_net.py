"""Complete automatic rim + lost-track + net flow for a fixed camera with a net.

Fresh neural inference is the default. --detections-cache explicitly reuses
unchanged inference while replaying all source frames and candidate tracking.
Annotations and supplied goal timestamps are not accepted by this entry point.
"""
import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import probe_net_assistance as net
from basketball_poc.media import inspect_video
from basketball_poc.net_recovery import recover_lost_tracks, supported_events

ROOT = Path(__file__).resolve().parents[1]
NET_PROFILE = ROOT/'configs/fixed-camera-net-assistance-v2.json'
RECOVERY_IMPLEMENTATION = ROOT/'src/basketball_poc/net_recovery.py'
MODEL = ROOT/'third_party/josephattalla/e320817d0f87eceb4093c871de6d29d1adca0006/bball_model.pt'


def identities(mapping):
    for path, digest in mapping.items():
        net.require(net.sha256(path) == digest, 'Input changed: '+path)


def recovery(base, profile):
    manifest = net.read_json(base/'manifest.json')
    scene = net.read_json(base/'scene.json')
    events = net.read_json(base/'events.json')
    result = recover_lost_tracks(net.lines(base/'associations.jsonl'), net.lines(base/'attempts.jsonl'),
        scene['hoops'], events['events'], fps=manifest['fps'],
        duration_seconds=manifest['duration_seconds'], candidate_profile=manifest['profile'],
        net_window_seconds=profile['window_seconds'])
    return manifest, scene, events, result


def cases_for(events, manifest, scene, profile):
    cases = []
    fps = manifest['fps']
    for event in events:
        hid = event['hoop_id']
        net.require(type(hid) is int and 0 <= hid < len(scene['hoops']), 'Invalid automatic hoop mapping')
        h = scene['hoops'][hid]['xyxy']
        x1, y1, x2, y2 = h; w, height = x2-x1, y2-y1
        anchor = event.get('net_anchor_seconds', event['time_seconds'])
        first = max(0, math.floor((anchor-profile['window_seconds'])*fps))
        last = min(manifest['processed_frames']-1, math.ceil((anchor+profile['window_seconds'])*fps))
        cases.append({'id': event['id'], 'time_seconds': anchor,
            'event_time_seconds': event['time_seconds'], 'hoop_id': hid, 'hoop_xyxy': h,
            'context_roi': [max(0, math.floor(x1-w)), max(0, math.floor(y1-height)),
                            min(manifest['source']['width'], math.ceil(x2+w)),
                            min(manifest['source']['height'], math.ceil(y2+height))],
            'frame_indices': list(range(first, last+1))})
    return cases


def experiment(video, out, cache):
    out.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    info = inspect_video(video)
    paths = [video, NET_PROFILE, ROOT/'configs/automatic-defaults.json', MODEL,
             Path(__file__), RECOVERY_IMPLEMENTATION, Path(net.__file__),
             ROOT/'scripts/benchmark_automatic.py', ROOT/'src/basketball_poc/temporal.py',
             ROOT/'src/basketball_poc/scene.py', ROOT/'src/basketball_poc/upstream_adapter.py']
    if cache:
        cached = net.read_json(cache/'manifest.json')
        net.require(cached['status'] == 'completed', 'Detection cache incomplete')
        net.require(cached['source_sha256'] == net.sha256(video), 'Cache source differs')
        net.require(cached['model_sha256'] == net.sha256(MODEL), 'Cache model differs')
        net.require(cached['profile_sha256'] == net.sha256(ROOT/'configs/automatic-defaults.json'), 'Cache profile differs')
        net.require(cached['detections_sha256'] == net.sha256(cache/'detections.jsonl'), 'Cached detections differ')
        net.require(cached['scene_sha256'] == net.sha256(cache/'scene.json'), 'Cached scene differs')
        paths += [cache/name for name in ('manifest.json', 'scene.json', 'detections.jsonl')]
    inputs = {str(path): net.sha256(path) for path in paths}
    profile = net.read_json(NET_PROFILE)
    net.write_json(out/'pipeline-inputs.json', {'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'inputs_sha256': inputs, 'source': info, 'net_profile': profile,
        'detection_cache': str(cache) if cache else None,
        'labels_or_supplied_goal_times_used': False, 'manual_geometry_used': False,
        'scope': 'fixed camera and visible net; stable hoop is automatically discovered'})
    for name, path in [('pipeline-implementation.py', Path(__file__)),
                       ('recovery-implementation.py', RECOVERY_IMPLEMENTATION),
                       ('net-implementation.py', Path(net.__file__))]:
        (out/name).write_bytes(path.read_bytes())
    command = [sys.executable, '-u', str(ROOT/'scripts/benchmark_automatic.py'), str(video),
               '--out', str(out/'candidates'), '--event-mode', 'rim-v2', '--scene-mode', 'nearest-rank-v1']
    if cache:
        command += ['--detections-cache', str(cache)]
    print('Replaying full detection cache' if cache else 'Running fresh full detection', flush=True)
    with (out/'candidate-process.log').open('x', encoding='utf-8') as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    base = out/'candidates'
    candidate_manifest = net.read_json(base/'manifest.json')
    if candidate_manifest['status'] == 'unsupported':
        identities(inputs)
        net.write_json(out/'results.json', {'status': 'unsupported',
            'reason': candidate_manifest['reason'], 'source': info,
            'seconds': time.perf_counter()-began, 'net_supported_count': 0,
            'annotations_used': False, 'neural_inference_reused': cache is not None})
        print('Automatic analysis unavailable:', candidate_manifest['reason'], flush=True)
        return
    manifest, scene, original, recovered = recovery(base, profile)
    net.require(manifest['status'] == 'completed', 'Automatic detection incomplete or scene unavailable')
    raw = sorted([{**e, 'evidence_origin': 'rim-v2'} for e in original['events']] + recovered['events'],
                 key=lambda e: (e['time_seconds'], e['id']))
    envelope = {k: v for k, v in original.items() if k != 'events'}
    cases = cases_for(raw, manifest, scene, profile)
    net.write_json(out/'recovery.json', recovered)
    net.write_json(out/'review-events.json', {**envelope, 'events': raw,
        'semantic_limit': 'Unverified crossing or lost-track review candidates'})
    frozen_inputs = {str(base/name): net.sha256(base/name) for name in
                    ('manifest.json', 'scene.json', 'events.json', 'associations.jsonl', 'attempts.jsonl', 'detections.jsonl')}
    frozen_inputs.update(inputs)
    net.write_json(out/'frozen.json', {'inputs_sha256': frozen_inputs, 'profile': profile,
        'cases': cases, 'source': manifest['source'], 'raw_events': raw,
        'annotations_used_for_inference': False, 'manual_timestamp_used': False})
    detections = net.lines(base/'detections.jsonl')
    crops, count = net.decode_cases(video, cases)
    net.require(count == manifest['processed_frames'] == len(detections), 'Incomplete full source decode')
    records = []
    for case in cases:
        images = crops[case['id']]
        net.np.savez_compressed(out/(case['id']+'-native-crops.npz'), frames=images)
        actual = net.probe_case(case, images, detections, manifest['fps'], profile)
        actual['decoded_crop_pixel_sha256'] = [net.pixels_sha(image) for image in images]
        records.append(actual)
        print(case['id'], round(case['event_time_seconds'], 3), actual['net_support'],
              actual['maximum_consecutive_support_frames'], flush=True)
    selected = supported_events(raw, records)
    uncertain = [{**e, 'net_support': False, 'net_state': r['state'],
                  'physical_goal_confirmation': 'pending'}
                 for e, r in zip(raw, records, strict=True) if not r['net_support']]
    net.write_json(out/'supported-events.json', {**envelope, 'events': selected,
        'semantic_limit': 'Trajectory candidates with net support; physical goal confirmation remains pending'})
    net.write_json(out/'uncertain-events.json', {**envelope, 'events': uncertain})
    pages = net.sheets(out, records, crops, manifest['fps'])
    identities(frozen_inputs)
    net.write_json(out/'results.json', {'status': 'completed', 'profile': profile,
        'records': records, 'source_pages': pages, 'source': manifest['source'],
        'decoded_source_frames': count, 'original_candidate_count': len(original['events']),
        'recovery_candidate_count': len(recovered['events']), 'raw_candidate_count': len(raw),
        'net_supported_count': len(selected), 'uncertain_count': len(uncertain),
        'seconds': time.perf_counter()-began, 'annotations_used': False,
        'neural_inference_reused': cache is not None,
        'new_neural_inference_frames': 0 if cache else manifest['processed_frames'],
        'user_choices_changed': False, 'net_parameters_changed': False})
    print('Complete', count, 'source frames,', len(recovered['events']), 'recovery candidates,',
          len(selected), 'supported events', flush=True)


def verify(out):
    net.require(not (out/'verification.json').exists(), 'Already verified')
    frozen, results = [net.read_json(out/name) for name in ('frozen.json', 'results.json')]
    identities(frozen['inputs_sha256'])
    base = out/'candidates'
    manifest, scene, original, recovered = recovery(base, frozen['profile'])
    net.require(recovered == net.read_json(out/'recovery.json'), 'Recovery replay differs')
    raw = sorted([{**e, 'evidence_origin': 'rim-v2'} for e in original['events']] + recovered['events'],
                 key=lambda e: (e['time_seconds'], e['id']))
    net.require(raw == frozen['raw_events'] == net.read_json(out/'review-events.json')['events'], 'Review events differ')
    net.require(cases_for(raw, manifest, scene, frozen['profile']) == frozen['cases'], 'Window selection differs')
    crops, count = net.decode_cases(Path(frozen['source']['path']), frozen['cases'])
    detections = net.lines(base/'detections.jsonl')
    net.require(count == results['decoded_source_frames'] == len(detections), 'Decode coverage differs')
    for record in results['records']:
        with net.np.load(out/(record['id']+'-native-crops.npz'), allow_pickle=False) as saved:
            net.require(net.np.array_equal(saved['frames'], crops[record['id']]), 'Native source crop differs')
        net.require([net.pixels_sha(f) for f in crops[record['id']]] == record['decoded_crop_pixel_sha256'], 'Pixel hashes differ')
        case = next(c for c in frozen['cases'] if c['id'] == record['id'])
        actual = net.probe_case(case, crops[record['id']], detections, manifest['fps'], frozen['profile'])
        net.require(all(record[k] == v for k, v in actual.items()), 'Net replay differs')
        before = net.np.asarray([r['displacement_p75_hoop_widths'] for r in record['rows']
                                if r['valid'] and r['relative_seconds'] <= frozen['profile']['pre_end_seconds']])
        if len(before) >= 6:
            median = float(net.np.median(before))
            threshold = median + frozen['profile']['noise_mad_multiplier']*float(net.np.median(abs(before-median))) + frozen['profile']['minimum_displacement_hoop_widths']
            net.require(threshold == record['threshold'], 'Independent threshold differs')
    net.require(supported_events(raw, results['records']) == net.read_json(out/'supported-events.json')['events'], 'Supported selection differs')
    expected_uncertain = [{**e, 'net_support': False, 'net_state': r['state'],
                          'physical_goal_confirmation': 'pending'}
                         for e, r in zip(raw, results['records'], strict=True) if not r['net_support']]
    net.require(expected_uncertain == net.read_json(out/'uncertain-events.json')['events'], 'Uncertain selection differs')
    identities(frozen['inputs_sha256'])
    net.write_json(out/'verification.json', {'status': 'passed', 'full_source_decoded_frames': count,
        'native_pixels_equal': True, 'recovery_replay_equal': True, 'net_replay_equal': True,
        'threshold_arithmetic_equal': True, 'all_outputs_match_events': True,
        'frozen_inputs_unchanged': True, 'manual_timestamp_used': False})
    print('Verified', count, 'source frames, recovery, net decisions and event outputs', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', nargs='?', type=Path)
    parser.add_argument('--detections-cache', type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--out', type=Path); group.add_argument('--verify', type=Path)
    args = parser.parse_args()
    net.cv2.setNumThreads(1); net.cv2.setRNGSeed(0)
    if args.verify:
        net.require(args.video is None and args.detections_cache is None, 'Verification uses its own frozen inputs')
        verify(args.verify.resolve(strict=True))
    else:
        net.require(args.video is not None, 'A source video is required')
        experiment(args.video.resolve(strict=True), args.out.resolve(),
                   args.detections_cache.resolve(strict=True) if args.detections_cache else None)
