"""Scale-only experiment: fresh detector followed by unchanged candidate/net code."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
import probe_net_assistance as net
import benchmark_fixed_camera_net as legacy
from basketball_poc.shot_context import shot_context
from basketball_poc.net_recovery import supported_events

ROOT = Path(__file__).resolve().parents[1]


def derive(base):
    manifest, scene, original, recovery = legacy.recovery(base, net.read_json(legacy.NET_PROFILE))
    prior = sorted([{**e, 'evidence_origin': 'rim-v2'} for e in original['events']] + recovery['events'],
                   key=lambda e: (e['time_seconds'], e['id']))
    context = shot_context(net.lines(base/'detections.jsonl'), scene['hoops'], prior,
        fps=manifest['fps'], duration_seconds=manifest['duration_seconds'], candidate_profile=manifest['profile'],
        context_profile=net.read_json(ROOT/'configs/fixed-camera-shot-context-v2.json'),
        net_window_seconds=net.read_json(legacy.NET_PROFILE)['window_seconds'])
    events = sorted(prior + context['events'], key=lambda e: (e['time_seconds'], e['id']))
    envelope = {k: v for k, v in original.items() if k != 'events'}
    return manifest, scene, recovery, context, {**envelope, 'events': events}


def compute(out, verify=False):
    began = time.perf_counter()
    m, scene, recovery, context, raw = derive(out/'candidates')
    profile = net.read_json(legacy.NET_PROFILE)
    cases = legacy.cases_for(raw['events'], m, scene, profile)
    crops, count = net.decode_cases(Path(m['source']['path']), cases)
    rows = net.lines(out/'candidates/detections.jsonl')
    net.require(count == m['processed_frames'] == len(rows), 'Frame coverage mismatch')
    records = []
    for case in cases:
        frames = crops[case['id']]
        record = net.probe_case(case, frames, rows, m['fps'], profile)
        record['decoded_crop_pixel_sha256'] = [net.pixels_sha(im) for im in frames]
        records.append(record)
        cropfile = out/(case['id']+'-native-crops.npz')
        if verify:
            with net.np.load(cropfile, allow_pickle=False) as saved:
                net.require(net.np.array_equal(saved['frames'], frames), 'Native crop pixels changed')
        else:
            net.np.savez_compressed(cropfile, frames=frames)
    envelope = {k: v for k, v in raw.items() if k != 'events'}
    selected = supported_events(raw['events'], records)
    uncertain = [{**e, 'net_support': False, 'net_state': r['state'], 'physical_goal_confirmation': 'pending'}
        for e, r in zip(raw['events'], records, strict=True) if not r['net_support']]
    outputs = {'recovery.json': recovery, 'context.json': context, 'review-events.json': raw,
        'supported-events.json': {**envelope, 'events': selected},
        'uncertain-events.json': {**envelope, 'events': uncertain},
        'net-records.json': {'profile': profile, 'records': records}}
    for name, data in outputs.items():
        if verify:
            net.require(data == net.read_json(out/name), 'Replay differs: '+name)
        else:
            net.write_json(out/name, data)
    if not verify:
        net.sheets(out, records, crops, m['fps'])
        net.write_json(out/'results.json', {'status': 'completed', 'source_frames': count,
            'fresh_ball_inference_frames': count, 'scout_inference_samples': len(scene['samples']),
            'raw_candidate_count': len(raw['events']), 'supported_count': len(selected),
            'uncertain_count': len(uncertain), 'net_processing_seconds': time.perf_counter()-began,
            'annotations_used': False, 'manual_times_used': False, 'model_trained': False,
            'net_and_context_parameters_changed': False})
    return count


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('video', nargs='?', type=Path)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--out', type=Path)
    group.add_argument('--verify', type=Path)
    args = p.parse_args()
    net.cv2.setNumThreads(1); net.cv2.setRNGSeed(0)
    if args.verify:
        out = args.verify.resolve(strict=True)
        frozen = net.read_json(out/'frozen.json')
        legacy.identities(frozen['inputs_sha256'])
        # Replay original event generator from the verified experimental cache.
        command = [sys.executable, '-u', str(ROOT/'scripts/benchmark_automatic_scale.py'),
                   frozen['video'], '--out', str(out/'candidate-replay'), '--event-mode', 'rim-v2',
                   '--detections-cache', str(out/'candidates')]
        with (out/'replay.log').open('x', encoding='utf-8') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        for name in ['detections.jsonl', 'associations.jsonl', 'attempts.jsonl', 'events.json', 'scene.json']:
            net.require((out/'candidates'/name).read_bytes() == (out/'candidate-replay'/name).read_bytes(),
                        'Candidate replay differs: '+name)
        count = compute(out, verify=True)
        legacy.identities(frozen['inputs_sha256'])
        net.write_json(out/'verification.json', {'status': 'passed', 'source_frames': count,
            'candidate_replay_equal': True, 'context_recovery_net_equal': True, 'native_pixels_equal': True,
            'protected_inputs_unchanged': True})
        print('Verified', out.name, count, flush=True)
        return
    net.require(args.video is not None, 'Video required')
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    video = args.video.resolve(strict=True)
    paths = [video, legacy.MODEL, Path(__file__), ROOT/'scripts/benchmark_automatic_scale.py',
             ROOT/'scripts/benchmark_fixed_camera_net.py', Path(net.__file__),
             ROOT/'configs/automatic-ball-scale320.json', legacy.NET_PROFILE,
             ROOT/'configs/fixed-camera-shot-context-v2.json']
    paths += list((ROOT/'src/basketball_poc').glob('*.py'))
    inputs = {str(path): net.sha256(path) for path in paths}
    net.write_json(out/'frozen.json', {'inputs_sha256': inputs, 'video': str(video),
                                    'labels_used': False, 'only_ball_input_scale_changed': True})
    command = [sys.executable, '-u', str(ROOT/'scripts/benchmark_automatic_scale.py'), str(video),
               '--out', str(out/'candidates'), '--event-mode', 'rim-v2']
    with (out/'detection.log').open('x', encoding='utf-8') as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    compute(out)
    legacy.identities(inputs)
    print(out.name, net.read_json(out/'results.json'), flush=True)


if __name__ == '__main__':
    main()
