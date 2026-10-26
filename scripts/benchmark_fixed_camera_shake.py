"""Experimental small-shake fallback; accepted fixed-camera path stays unchanged."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import cv2
import benchmark_fixed_camera_scale as scale
from basketball_poc.camera_shake import stabilize
from basketball_poc.media import read_json, write_json, sha256

ROOT = Path(__file__).resolve().parents[1]


def pipeline(video, out, cache=None):
    out.mkdir(parents=True, exist_ok=False)
    args = [sys.executable, '-u', str(ROOT/'scripts/benchmark_automatic_scale.py'), str(video),
            '--out', str(out/'candidates'), '--event-mode', 'rim-v2']
    if cache: args += ['--detections-cache', str(cache)]
    with (out/'detection.log').open('x', encoding='utf-8') as log:
        subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    manifest = read_json(out/'candidates/manifest.json')
    if manifest['status'] == 'unsupported':
        result = {'status': 'unsupported', 'reason': manifest['reason'],
                  'supported_count': None, 'uncertain_count': None, 'fresh_ball_inference_frames': 0}
        write_json(out/'results.json', result)
        return result
    scale.compute(out)
    result = read_json(out/'results.json')
    # Correct the reused helper's fresh-only metadata for explicit cache regression.
    result['fresh_ball_inference_frames'] = 0 if cache else manifest['processed_frames']
    result['inference_mode'] = 'verified_detection_cache' if cache else 'fresh'
    (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    write_json(out/'execution-mode.json', {'detections_cache': str(cache) if cache else None,
        'fresh_ball_inference_frames': 0 if cache else manifest['processed_frames'],
        'cached_ball_detection_frames': manifest['processed_frames'] if cache else 0})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--detections-cache', type=Path, help='Explicit verified baseline regression only')
    args = parser.parse_args()
    cv2.setNumThreads(1); cv2.setRNGSeed(0)
    video = args.video.resolve(strict=True); out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    paths = [video, Path(__file__), ROOT/'scripts/benchmark_fixed_camera_scale.py',
             ROOT/'scripts/benchmark_automatic_scale.py', ROOT/'scripts/benchmark_fixed_camera_net.py',
             ROOT/'scripts/probe_net_assistance.py', scale.legacy.MODEL]
    paths += list((ROOT/'src/basketball_poc').glob('*.py'))
    paths += [ROOT/'configs'/name for name in ('automatic-ball-scale320.json',
        'fixed-camera-net-assistance-v2.json', 'fixed-camera-shot-context-v2.json','minor-camera-shake.json')]
    inputs = {str(p):sha256(p) for p in paths}
    write_json(out/'frozen.json', {'inputs_sha256': inputs, 'video': str(video),
        'manual_times_or_labels_used': False,'baseline_tag':'baseline-scale320-2026-10-14'})
    original = pipeline(video, out/'original', args.detections_cache)
    active = out/'original'; mode = 'original_passthrough'; result = original
    if original['status'] == 'unsupported' and args.detections_cache is None:
        motion = stabilize(video, out/'stabilization', read_json(active/'candidates/scene.json'),
                           read_json(ROOT/'configs/minor-camera-shake.json'))
        mode = 'stabilization_rejected'
        if motion['status'] == 'completed':
            active = out/'corrected'
            result = pipeline(Path(motion['output']), active)
            mode = 'stabilized_fallback'
        else:
            result = {**original, 'reason': motion['reason']}
    scale.legacy.identities(inputs)
    write_json(out/'result.json', {**result, 'mode': mode, 'pipeline': str(active),
        'original_video': str(video),'original_source_sha256':sha256(video),
        'original_seconds_offset': 0,'user_acceptance':'pending','physical_goal_count':None,
        'detections_cache':str(args.detections_cache) if args.detections_cache else None,
        'fresh_ball_inference_frames':0 if args.detections_cache else result['fresh_ball_inference_frames']})
    write_json(out/'predictions-before-review.json', {'files_sha256':{
        str(p):sha256(p) for p in out.rglob('*') if p.is_file()},'labels_used':False})
    print(read_json(out/'result.json'),flush=True)


if __name__ == '__main__': main()
