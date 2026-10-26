"""Explicit ball inference device routing inside an isolated analysis process."""
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from pathlib import Path
import runpy
import subprocess
import sys
import time


@contextmanager
def ball_inference(device, detector_entry, inline=False):
    if device not in ('cpu', 'cuda:0'):
        raise ValueError('Inference device must be cpu or cuda:0')
    record = {'requested_device': device, 'scene_device': 'cpu',
              'ball_device': device, 'precision': 'float32',
              'ball_calls': 0, 'ball_predict_seconds': 0.,
              'source_frames_skipped': 0, 'silent_fallback': False}
    record.update(scene_calls=0, scene_predict_seconds=0., model_phase_seconds={})
    if device == 'cpu' and not inline:
        record['scope'] = 'legacy CPU path; prediction adapter disabled'
        yield record
        return
    import torch
    from ultralytics import YOLO
    if device == 'cuda:0' and not torch.cuda.is_available():
        raise ValueError('GPU inference requested but CUDA is unavailable; use the CUDA runtime or explicit cpu')
    record.update(hardware=torch.cuda.get_device_name(0) if device == 'cuda:0' else 'CPU',
                  torch_version=torch.__version__, cuda_version=torch.version.cuda, tf32=False)
    entry = Path(detector_entry).resolve(strict=True)
    original_predict, original_run = YOLO.predict, subprocess.run
    original_argv = sys.argv
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32

    def predict(model, *args, **kwargs):
        if kwargs.get('classes') != [0] or kwargs.get('imgsz') != 320:
            began = time.perf_counter()
            results = original_predict(model, *args, **kwargs)
            record['scene_predict_seconds'] += time.perf_counter()-began
            record['scene_calls'] += 1
            return results
        if kwargs.get('device') != 'cpu':
            raise ValueError('Unexpected accepted detector device request')
        kwargs.update(device=device, quantize=32)
        if device == 'cuda:0':
            torch.cuda.synchronize(0)
        began = time.perf_counter()
        results = original_predict(model, *args, **kwargs)
        if device == 'cuda:0':
            torch.cuda.synchronize(0)
        if not results or any(str(r.boxes.data.device) != device for r in results):
            raise ValueError('GPU prediction did not return CUDA evidence')
        if any(r.boxes.data.dtype != torch.float32 for r in results):
            raise ValueError('GPU prediction precision changed')
        record['ball_calls'] += len(results)
        record['ball_predict_seconds'] += time.perf_counter()-began
        for result in results:
            for phase, milliseconds in (getattr(result, 'speed', None) or {}).items():
                if milliseconds is not None:
                    record['model_phase_seconds'][phase] = record['model_phase_seconds'].get(phase,0)+milliseconds/1000
        return results

    def run(arguments, **kwargs):
        if (not isinstance(arguments, (list, tuple)) or len(arguments) < 3
                or Path(arguments[2]).resolve() != entry):
            return original_run(arguments, **kwargs)
        if '--detections-cache' in arguments or '--scene-cache' in arguments:
            raise ValueError('GPU experiment requires fresh inference')
        try:
            sys.argv = [str(entry), *map(str, arguments[3:])]
            with redirect_stdout(kwargs['stdout']), redirect_stderr(kwargs['stdout']):
                runpy.run_path(str(entry), run_name='__main__')
            return subprocess.CompletedProcess(arguments, 0)
        finally:
            sys.argv = original_argv

    YOLO.predict, subprocess.run = predict, run
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield record
    finally:
        YOLO.predict, subprocess.run, sys.argv = original_predict, original_run, original_argv
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


def verify_execution(record, frames, hoops):
    """Validate actual GPU calls; legacy CPU paths do not install an adapter."""
    if (record.get('requested_device') not in ('cpu', 'cuda:0')
            or record.get('ball_device') != record['requested_device']
            or record.get('scene_device') != 'cpu'
            or record.get('precision') != 'float32'
            or record.get('source_frames_skipped') != 0
            or record.get('silent_fallback') is not False):
        raise ValueError('Invalid inference execution evidence')
    if record['requested_device'] == 'cuda:0':
        if (record.get('ball_calls') != frames*hoops or not record.get('hardware')
                or not record.get('cuda_version') or record.get('tf32') is not False):
            raise ValueError('Incomplete GPU inference coverage')
