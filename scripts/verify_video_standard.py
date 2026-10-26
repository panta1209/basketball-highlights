"""Standard integrity/mapping verification. Does not claim full inference replay."""
import argparse
from pathlib import Path

import cv2
import numpy as np

from basketball_poc.camera_shake import validate_motion
from basketball_poc.media import read_json, write_json, sha256, inspect_video
from basketball_poc.net_recovery import supported_events
from basketball_poc.stable_review import verify_hashes
from basketball_poc.stream_frames import frame_source_identity
from basketball_poc.inference_runtime import verify_execution
from basketball_poc.frame_execution import verify_frame_execution


def verify(folder):
    folder = Path(folder).resolve(strict=True)
    cv2.setNumThreads(1)
    frozen = read_json(folder/'frozen.json')['inputs_sha256']
    predictions = read_json(folder/'predictions-before-review.json')['files_sha256']
    verify_hashes(frozen); verify_hashes(predictions)
    result = read_json(folder/'result.json')
    if result['status'] != 'completed':
        raise ValueError('Completed analysis required')
    pipeline = Path(result['pipeline'])
    manifest = read_json(pipeline/'candidates/manifest.json')
    info = inspect_video(result['original_video'])
    count = round(info['fps']*info['duration_seconds'])
    if 'inference_execution' in result:
        scene = read_json(pipeline/'candidates/scene.json')
        verify_execution(result['inference_execution'], count, len(scene['hoops']))
        if (manifest.get('inference_execution') != result['inference_execution']
                or manifest['device'] != result['inference_execution']['ball_device']):
            raise ValueError('Inference device manifest mismatch')
    if result.get('batch_execution'):
        batches = result['batch_execution']
        execution = batches[-1]
        images = count*len(scene['hoops'])
        if (execution['roi_images'] != images or execution['delivered_calls'] != images
                or execution['source_frames_skipped'] != 0
                or sum(int(k)*v for k,v in execution['batch_sizes'].items()) != images
                or execution['max_batch_frames'] > execution['policy']['batch_size']
                or any(b['roi_images'] for b in batches[:-1])):
            raise ValueError('Incomplete ordered batch inference coverage')
    if (sha256(info['path']) != result['original_source_sha256']
            or manifest['status'] != 'completed' or manifest['processed_frames'] != count
            or manifest['fps'] != info['fps'] or manifest['duration_seconds'] != info['duration_seconds']):
        raise ValueError('Source time/frame inventory mismatch')
    import json
    with (pipeline/'candidates/detections.jsonl').open(encoding='utf-8') as file:
        rows = [json.loads(line) for line in file]
    if len(rows) != count or any(r['frame'] != i or abs(r['time_seconds']-i/info['fps']) > 1e-6 for i, r in enumerate(rows)):
        raise ValueError('Detection coverage incomplete')
    raw = read_json(pipeline/'review-events.json')['events']
    supported = read_json(pipeline/'supported-events.json')['events']
    uncertain = read_json(pipeline/'uncertain-events.json')['events']
    records = read_json(pipeline/'net-records.json')['records']
    if 'frame_execution' in result:
        import benchmark_fixed_camera_net as legacy
        scene = read_json(pipeline/'candidates/scene.json')
        profile = read_json(pipeline/'net-records.json')['profile']
        cases = legacy.cases_for(raw, manifest, scene, profile)
        verify_frame_execution(result['frame_execution'], cases, count)
    ids = [e['id'] for e in raw]
    if (len(set(ids)) != len(ids) or [r['id'] for r in records] != ids
            or supported_events(raw, records) != supported
            or {e['id'] for e in supported+uncertain} != set(ids)
            or len(supported)+len(uncertain) != len(ids)
            or any(r['net_support'] for e, r in zip(raw, records) if e['id'] in {x['id'] for x in uncertain})):
        raise ValueError('Event/net partition mismatch')
    pixel_checks = []
    if result['mode'] == 'stabilized_fallback':
        motion = read_json(folder/'stabilization/stabilization.json')
        transforms = motion['transforms']
        dynamic = motion.get('storage') == 'dynamic_frames_v1'
        if (len(transforms) != count or motion['frames'] != count
                or (not dynamic and sha256(motion['output']) != motion['output_sha256'])):
            raise ValueError('Stabilization coverage/identity mismatch')
        if dynamic and (motion['source_sha256'] != result['original_source_sha256']
                or motion['frame_source_sha256'] != frame_source_identity(motion['source_sha256'], motion['profile'])
                or manifest['source_sha256'] != motion['frame_source_sha256']
                or manifest.get('frame_source', {}).get('frame_source_sha256') != motion['frame_source_sha256']
                or manifest['frame_source'].get('original_source_sha256') != result['original_source_sha256']
                or any(p.suffix.lower() in ('.avi', '.mkv', '.mp4') for p in (folder/'stabilization').iterdir())):
            raise ValueError('Dynamic frame source identity mismatch')
        for i, row in enumerate(transforms):
            if row['frame'] != i or row['time_seconds'] != i/info['fps']:
                raise ValueError('Transform timestamp mismatch')
            if dynamic and (not isinstance(row.get('corrected_frame_sha256'), str) or len(row['corrected_frame_sha256']) != 64):
                raise ValueError('Missing corrected frame identity')
            validate_motion(np.array(row['reference_to_current']), info['width'], info['height'], motion['profile'])
            if (row['inlier_points'] < motion['profile']['minimum_inliers']
                    or row['inlier_fraction'] < motion['profile']['minimum_inlier_fraction']):
                raise ValueError('Transform support insufficient')
        original = cv2.VideoCapture(info['path'])
        corrected = None if dynamic else cv2.VideoCapture(motion['output'])
        try:
            if original.get(cv2.CAP_PROP_FRAME_COUNT) != count or (corrected is not None and corrected.get(cv2.CAP_PROP_FRAME_COUNT) != count):
                raise ValueError('Video frame count mismatch')
            for i in sorted({0, count//4, count//2, count*3//4, count-1}):
                original.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, a = original.read()
                if corrected is not None:
                    corrected.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ok2, b = corrected.read()
                else:
                    ok2 = True
                if not ok or not ok2:
                    raise ValueError('Mapping sample unavailable')
                matrix = np.array(transforms[i]['reference_to_current'])
                expected = cv2.warpAffine(a, cv2.invertAffineTransform(matrix), (info['width'], info['height']),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                import hashlib
                equal = (hashlib.sha256(expected.tobytes()).hexdigest() == transforms[i].get('corrected_frame_sha256')) if dynamic else np.array_equal(expected, b)
                if not equal:
                    raise ValueError(f'Mapping pixel mismatch at frame {i}')
                pixel_checks.append(i)
        finally:
            original.release()
            if corrected is not None:
                corrected.release()
    elif result['mode'] != 'original_passthrough' or manifest['source_sha256'] != result['original_source_sha256']:
        raise ValueError('Unexpected source mapping')
    verify_hashes(predictions); verify_hashes(frozen)
    record = {'status': 'passed', 'recognition_status': 'completed', 'verification_mode': 'standard-v1',
        'source_transform_pixel_checks': pixel_checks, 'frame_and_time_mapping_verified': True,
        'candidate_inventory_verified': True, 'event_net_partition_verified': True,
        'candidate_context_net_replay_equal': False, 'full_replay_performed': False,
        'predictions_unchanged': True, 'method_identities_unchanged': True,
        'scope': 'full frozen identities/frame-event inventories; bounded pixel mapping checks; no candidate/net replay'}
    write_json(folder/'verification.json', record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('run', type=Path)
    print(verify(parser.parse_args().run), flush=True)
