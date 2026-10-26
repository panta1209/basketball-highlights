"""Accepted analysis with the separately verified lossless encoding candidate."""
import argparse
from pathlib import Path

import cv2

import benchmark_fixed_camera_shake as reference
from basketball_poc.fast_stabilization import stabilize
from basketball_poc.media import read_json, write_json, sha256

ROOT = Path(__file__).resolve().parents[1]
PERFORMANCE = ROOT/'configs/video-processing-performance-v1.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1); cv2.setRNGSeed(0)
    video = args.video.resolve(strict=True); out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    paths = [video, Path(__file__), PERFORMANCE, reference.scale.legacy.MODEL,
        ROOT/'scripts/benchmark_fixed_camera_shake.py', ROOT/'scripts/benchmark_fixed_camera_scale.py',
        ROOT/'scripts/benchmark_automatic_scale.py', ROOT/'scripts/benchmark_fixed_camera_net.py',
        ROOT/'scripts/probe_net_assistance.py']
    paths += list((ROOT/'src/basketball_poc').glob('*.py'))
    paths += [ROOT/'configs'/name for name in ('automatic-ball-scale320.json',
        'fixed-camera-net-assistance-v2.json', 'fixed-camera-shot-context-v2.json', 'minor-camera-shake.json')]
    inputs = {str(p): sha256(p) for p in paths}
    write_json(out/'frozen.json', {'inputs_sha256': inputs, 'video': str(video),
        'manual_times_or_labels_used': False, 'processing_mode': 'efficient-v1',
        'recognition_parameters_changed': False})
    original = reference.pipeline(video, out/'original')
    active = out/'original'; mode = 'original_passthrough'; result = original
    if original['status'] == 'unsupported':
        motion = stabilize(video, out/'stabilization', read_json(active/'candidates/scene.json'),
            read_json(ROOT/'configs/minor-camera-shake.json'), read_json(PERFORMANCE))
        mode = 'stabilization_rejected'
        if motion['status'] == 'completed':
            active = out/'corrected'
            result = reference.pipeline(Path(motion['output']), active)
            mode = 'stabilized_fallback'
        else:
            result = {**original, 'reason': motion['reason']}
    reference.scale.legacy.identities(inputs)
    write_json(out/'result.json', {**result, 'mode': mode, 'pipeline': str(active),
        'original_video': str(video), 'original_source_sha256': sha256(video),
        'original_seconds_offset': 0, 'user_acceptance': 'pending', 'physical_goal_count': None,
        'processing_mode': 'efficient-v1', 'fresh_ball_inference_frames': result['fresh_ball_inference_frames'],
        'detections_cache': None})
    write_json(out/'predictions-before-review.json', {'files_sha256': {
        str(p): sha256(p) for p in out.rglob('*') if p.is_file()}, 'labels_used': False})
    print(read_json(out/'result.json'), flush=True)


if __name__ == '__main__':
    main()
