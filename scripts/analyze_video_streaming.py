"""Fresh accepted inference with on-read stabilization and no corrected media file."""
import argparse
from pathlib import Path

import cv2

import benchmark_fixed_camera_shake as reference
from basketball_poc.camera_shake import MotionUnavailable
from basketball_poc.media import read_json, write_json, sha256
from basketball_poc.stream_frames import CorrectedFrames, routed_frames
from basketball_poc.inference_runtime import ball_inference, verify_execution
from basketball_poc.frame_execution import frame_execution
from basketball_poc.resource_policy import probe_resources, select_resources
from basketball_poc.batch_execution import batch_execution
from contextlib import nullcontext

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--inference-device', choices=['auto', 'cpu', 'cuda:0'], default='cpu')
    parser.add_argument('--opencv-threads', type=int, choices=[0,1,2,4,8], default=1)
    parser.add_argument('--execution-mode', choices=['sequential', 'batched'], default='sequential')
    parser.add_argument('--batch-size', type=int, choices=[0,1,2,4,8,16,32], default=0)
    parser.add_argument('--net-mode', choices=['full', 'windows'], default='full')
    args = parser.parse_args()
    cv2.setNumThreads(1); cv2.setRNGSeed(0)
    source, out = args.video.resolve(strict=True), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    from basketball_poc.media import inspect_video
    info = inspect_video(source)
    policy = select_resources(probe_resources(), info['width']*info['height']*3,
                              args.inference_device, args.batch_size)
    threads = args.opencv_threads or policy['opencv_threads']
    batches = []

    def pipeline(destination):
        context = batch_execution(policy, source) if args.execution_mode == 'batched' else nullcontext(None)
        with context as batch:
            value = reference.pipeline(source, destination)
        if batch is not None:
            batches.append(batch)
        return value
    paths = [source, Path(__file__), reference.scale.legacy.MODEL,
             *[ROOT/'scripts'/name for name in ('benchmark_fixed_camera_shake.py',
                 'benchmark_fixed_camera_scale.py', 'benchmark_automatic_scale.py',
                 'benchmark_fixed_camera_net.py', 'probe_net_assistance.py')],
             ROOT/'scripts/verify_stream_analysis.py',
             *list((ROOT/'src/basketball_poc').glob('*.py')),
             *[ROOT/'configs'/name for name in ('automatic-ball-scale320.json',
                 'fixed-camera-net-assistance-v2.json', 'fixed-camera-shot-context-v2.json', 'minor-camera-shake.json')]]
    inputs = {str(p): sha256(p) for p in paths}
    write_json(out/'frozen.json', {'inputs_sha256': inputs, 'video': str(source),
        'processing_mode': 'streaming-v1', 'recognition_parameters_changed': False,
        'manual_times_or_labels_used': False})
    with ball_inference(policy['selected_device'], ROOT/'scripts/benchmark_automatic_scale.py',
                        inline=args.execution_mode == 'batched') as execution, \
            frame_execution(reference.scale.net, threads, args.net_mode) as frames_execution:
        result = pipeline(out/'original')
        active = out/'original'; mode = 'original_passthrough'
        if result['status'] == 'unsupported':
            provider = None
            mode = 'stabilization_rejected'
            try:
                provider = CorrectedFrames(source, read_json(active/'candidates/scene.json'),
                                           read_json(ROOT/'configs/minor-camera-shake.json'))
                active = out/'corrected'
                with routed_frames(provider, ROOT/'scripts/benchmark_automatic_scale.py'):
                    result = pipeline(active)
                if result['status'] == 'completed':
                    mode = 'stabilized_fallback'
                    motion = provider.record()
                    if args.execution_mode == 'batched':
                        motion['frame_storage'] = 'bounded two-batch lookahead; transforms/hashes retained, no whole-video image cache'
                        motion['frame_buffer_policy'] = policy
                        motion['timing_semantics'] = 'sum of worker durations; concurrent work overlaps and must not be added to wall time'
                    manifest_path = active/'candidates/manifest.json'
                    manifest = read_json(manifest_path)
                    manifest['frame_source'] = {'kind': 'dynamic-corrected-frames-v1',
                        'original_source_sha256': provider.digest, 'frame_source_sha256': provider.identity,
                        'mapping': str(out/'stabilization/stabilization.json')}
                    import json
                    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
                else:
                    motion = provider.record('unsupported', result['reason'])
            except MotionUnavailable as error:
                result = {'status': 'unsupported', 'reason': str(error), 'fresh_ball_inference_frames': 0}
                motion = provider.record('unsupported', str(error)) if provider else {
                    'status': 'unsupported', 'reason': str(error), 'storage': 'dynamic_frames_v1', 'frames': 0}
            write_json(out/'stabilization/stabilization.json', motion)
    if result['status'] == 'completed':
        scene = read_json(active/'candidates/scene.json')
        verify_execution(execution, result['fresh_ball_inference_frames'], len(scene['hoops']))
        import json
        manifest_path = active/'candidates/manifest.json'
        manifest = read_json(manifest_path)
        manifest.update(device=execution['ball_device'], scene_device='cpu', inference_execution=execution)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    reference.scale.legacy.identities(inputs)
    write_json(out/'result.json', {**result, 'mode': mode, 'pipeline': str(active),
        'original_video': str(source), 'original_source_sha256': sha256(source),
        'original_seconds_offset': 0, 'processing_mode': 'streaming-v1',
        'user_acceptance': 'pending', 'physical_goal_count': None, 'detections_cache': None,
        'inference_execution': execution, 'frame_execution': frames_execution,
        'hardware_policy': policy, 'batch_execution': batches})
    write_json(out/'predictions-before-review.json', {'files_sha256': {
        str(p): sha256(p) for p in out.rglob('*') if p.is_file()}, 'labels_used': False})
    print(read_json(out/'result.json'), flush=True)


if __name__ == '__main__':
    main()
