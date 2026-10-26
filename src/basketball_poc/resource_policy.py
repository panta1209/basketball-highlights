"""Conservative resource selection, independent of any particular Windows PC."""
import os

GIB = 1024**3


def probe_resources():
    import torch
    result = {'logical_cpus': os.cpu_count() or 1, 'available_memory_bytes': None,
              'cuda_available': False, 'cuda_free_bytes': 0, 'cuda_name': None,
              'probe_errors': [], 'torch_version': torch.__version__}
    try:
        import psutil
        result['available_memory_bytes'] = psutil.virtual_memory().available
        result['logical_cpus'] = len(psutil.Process().cpu_affinity())
    except Exception as error:
        result['logical_cpus'] = 1
        result['probe_errors'].append(type(error).__name__)
    try:
        if torch.cuda.is_available():
            # Availability alone does not prove this driver/GPU can execute installed kernels.
            torch.zeros(1, device='cuda:0').add_(1)
            torch.cuda.synchronize(0)
            result.update(cuda_available=True, cuda_free_bytes=torch.cuda.mem_get_info(0)[0],
                          cuda_name=torch.cuda.get_device_name(0))
    except (RuntimeError, OSError, AssertionError) as error:
        result['cuda_available'] = False
        result['probe_errors'].append(type(error).__name__)
    return result


def select_resources(resources, frame_bytes, device='auto', batch_size=0):
    if device not in ('auto', 'cpu', 'cuda:0') or batch_size not in (0, 1, 2, 4, 8, 16, 32):
        raise ValueError('Invalid hardware selection')
    cores = max(1, resources.get('logical_cpus') or 1)
    free = max(0, resources.get('available_memory_bytes') or 0)
    cuda_free = max(0, resources.get('cuda_free_bytes') or 0)
    usable_cuda = resources.get('cuda_available') and cuda_free >= 2*GIB
    if device == 'cuda:0' and not usable_cuda:
        raise ValueError('Explicit CUDA requires a working runtime and at least 2 GiB free VRAM')
    selected = 'cuda:0' if device == 'cuda:0' or (device == 'auto' and usable_cuda) else 'cpu'
    # Two batches, concurrent source/warped images and inference temporaries are bounded.
    budget = min(256*1024**2, free//16) if free else 8*1024**2
    capacity = max(1, min(32, budget//max(1, frame_bytes*6)))
    proposed = (16 if cuda_free >= 6*GIB else 8 if cuda_free >= 3*GIB else 1) if selected == 'cuda:0' else 1
    requested = batch_size or proposed
    batch = max(n for n in (1, 2, 4, 8, 16, 32) if n <= min(requested, capacity))
    workers = min(4, max(1, cores//4), max(1, batch)) if free >= 2*GIB else 1
    threads = 4 if cores >= 12 else 2 if cores >= 4 else 1
    if free < GIB:
        threads = workers = 1
    return {'version': 'adaptive-resources-v1', 'requested_device': device,
            'selected_device': selected, 'batch_size': batch, 'correction_workers': workers,
            'opencv_threads': threads, 'torch_cpu_threads': min(8,max(1,cores//2)),
            'frame_buffer_budget_bytes': budget,
            'frame_bytes': frame_bytes, 'max_buffered_frames': 2*batch+workers+1,
            'selection_reason': 'usable CUDA runtime and free VRAM' if selected == 'cuda:0'
                else 'CPU requested or CUDA unavailable/insufficient; same FP32 model',
            'resources': resources}
