"""Independent full frame reconstruction and accepted candidate/context/net replay."""
import argparse
from pathlib import Path
import subprocess
import sys

import cv2

import benchmark_fixed_camera_scale as scale
from basketball_poc.media import read_json, write_json, sha256
from basketball_poc.stable_review import verify_hashes
from basketball_poc.stream_frames import CorrectedFrames, routed_frames

ROOT = Path(__file__).resolve().parents[1]


def verify(analysis, output):
    analysis = Path(analysis).resolve(strict=True); output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1); cv2.setRNGSeed(0)
    frozen = read_json(analysis/'frozen.json')['inputs_sha256']
    predictions = read_json(analysis/'predictions-before-review.json')['files_sha256']
    verify_hashes(frozen); verify_hashes(predictions)
    result = read_json(analysis/'result.json'); pipe = Path(result['pipeline'])
    motion = read_json(analysis/'stabilization/stabilization.json')
    if result['status'] != 'completed' or motion['storage'] != 'dynamic_frames_v1':
        raise ValueError('Completed dynamic analysis required')
    provider = CorrectedFrames(result['original_video'], None, motion['profile'], saved=motion)
    count = 0
    capture = provider.capture(str(provider.source))
    try:
        while capture.read()[0]:
            count += 1
    finally:
        capture.release()
    if count != motion['frames']:
        raise ValueError('Full reconstruction frame count changed')
    entry = ROOT/'scripts/benchmark_automatic_scale.py'
    with routed_frames(provider, entry, allow_replay=True):
        with (output/'candidate-replay.log').open('x', encoding='utf-8') as log:
            subprocess.run([sys.executable, '-u', str(entry), str(provider.source), '--out', str(output/'candidates'),
                '--event-mode', 'rim-v2', '--detections-cache', str(pipe/'candidates')],
                stdout=log, stderr=subprocess.STDOUT, check=True)
        for name in ('detections.jsonl','associations.jsonl','attempts.jsonl','events.json','scene.json'):
            if (pipe/'candidates'/name).read_bytes() != (output/'candidates'/name).read_bytes():
                raise ValueError('Candidate replay differs: '+name)
        scale.compute(pipe, verify=True)
    verify_hashes(frozen); verify_hashes(predictions)
    record = {'status':'passed', 'recognition_status':'completed', 'verification_mode':'full-stream-replay-v1',
        'frame_and_time_mapping_verified':True, 'candidate_context_net_replay_equal':True,
        'full_replay_performed':True, 'all_frame_pixel_hashes_verified':count,
        'predictions_unchanged':True, 'method_identities_unchanged':True,
        'primary_verification_unchanged':sha256(analysis/'verification.json')}
    write_json(output/'verification.json', record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(verify(args.run, args.out), flush=True)
