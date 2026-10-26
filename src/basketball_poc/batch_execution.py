"""Bounded frame lookahead; existing tracker consumes one original frame at a time."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import time

import cv2
import numpy as np


class BufferedCapture:
    def __init__(self, raw, policy, owner):
        self.raw, self.policy, self.owner = raw, policy, owner
        self.current = None
        self.batch = []
        self.position = None
        self.pool = None
        self.next_batch = None
        self.closed = False
        self.cache = {}
        self.offset = 0
        self.correct = None
        self.corrections = ThreadPoolExecutor(max_workers=policy['correction_workers'])
        self.reader = raw
        # The accepted corrected capture can separate ordered decoding from independent correction.
        if hasattr(raw, 'provider') and hasattr(raw, 'raw'):
            self.reader, self.correct = raw.raw, raw.provider.correct

    def packet(self):
        index = round(self.reader.get(cv2.CAP_PROP_POS_FRAMES))
        ok, image = self.reader.read()
        if not ok:
            return None
        return (index, self.reader.get(cv2.CAP_PROP_POS_MSEC), image)

    def produce(self, limit):
        packets = []
        workers = self.policy['correction_workers']
        pending = deque()
        for _ in range(limit):
            packet = self.packet()
            if packet is None:
                break
            index, stamp, image = packet
            if self.correct:
                future = self.corrections.submit(self.correct, image, index)
                pending.append((index, stamp, future))
            else:
                packets.append(packet)
            if len(pending) >= workers:
                i, t, future = pending.popleft()
                packets.append((i, t, future.result()))
        while pending:
            i, t, future = pending.popleft()
            packets.append((i, t, future.result()))
        return packets

    def enable(self):
        if self.pool is not None:
            return
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.batch = [self.current, *self.produce(self.policy['batch_size']-1)]
        self.offset = 0
        self.next_batch = self.pool.submit(self.produce, self.policy['batch_size'])

    def read(self):
        if self.closed:
            return False, None
        if self.pool is None:
            # Scene discovery is unchanged and does not prefetch.
            index = round(self.raw.get(cv2.CAP_PROP_POS_FRAMES))
            ok, image = self.raw.read()
            if not ok:
                return False, None
            packet = (index, self.raw.get(cv2.CAP_PROP_POS_MSEC), image)
        else:
            self.offset += 1
            if self.offset >= len(self.batch):
                self.batch = self.next_batch.result()
                self.cache.clear()
                self.offset = 0
                if not self.batch:
                    return False, None
                self.next_batch = self.pool.submit(self.produce, self.policy['batch_size'])
            packet = self.batch[self.offset]
        self.current = packet
        self.position = (packet[0]+1, packet[1])
        self.owner['active'] = self
        return True, packet[2]

    def get(self, key):
        if self.position is not None:
            if key == cv2.CAP_PROP_POS_FRAMES:
                return self.position[0]
            if key == cv2.CAP_PROP_POS_MSEC:
                return self.position[1]
        return self.raw.get(key)

    def set(self, key, value):
        if self.pool is not None:
            raise ValueError('Cannot seek during ordered batched inference')
        result = self.raw.set(key, value)
        self.position = None
        return result

    def isOpened(self):
        return self.raw.isOpened()

    def release(self):
        if not self.closed:
            if self.pool:
                self.pool.shutdown(wait=True, cancel_futures=True)
            self.corrections.shutdown(wait=True, cancel_futures=True)
            self.raw.release()
            self.batch.clear()
            self.cache.clear()
            self.current = None
            self.closed = True


def crop_coordinates(image, frame):
    if image.ndim != 3 or image.strides != frame.strides or not np.shares_memory(image, frame):
        raise ValueError('Batch ROI must be a direct original-frame view')
    delta = image.ctypes.data-frame.ctypes.data
    y, remainder = divmod(delta, frame.strides[0])
    x, remainder = divmod(remainder, frame.strides[1])
    h, w = image.shape[:2]
    if remainder or x < 0 or y < 0 or x+w > frame.shape[1] or y+h > frame.shape[0]:
        raise ValueError('Invalid ROI memory mapping')
    return x, y, x+w, y+h


@contextmanager
def batch_execution(policy, source):
    from pathlib import Path
    from ultralytics import YOLO
    from ultralytics.utils import torch_utils
    import torch
    from . import upstream_adapter
    original_capture, original_predict = cv2.VideoCapture, YOLO.predict
    original_adapt = upstream_adapter.adapt_source
    original_thread_budget, original_torch_threads = torch_utils.NUM_THREADS, torch.get_num_threads()
    torch_utils.NUM_THREADS = policy.get('torch_cpu_threads', original_thread_budget)
    captures = []
    owner = {'active': None}
    record = {'version': 'bounded-batch-v1', 'policy': policy, 'model_batches': 0,
              'roi_images': 0, 'delivered_calls': 0, 'predict_seconds': 0.,
              'oom_retries': [], 'drawing_disabled': True, 'batch_sizes': {},
              'effective_batch_limit': policy['batch_size'],
              'max_batch_frames': 0, 'source_frames_skipped': 0}

    def capture(path, *args, **kwargs):
        raw = original_capture(path, *args, **kwargs)
        if not isinstance(path, (str, Path)) or Path(path).resolve() != Path(source).resolve():
            return raw
        wrapped = BufferedCapture(raw, policy, owner)
        captures.append(wrapped)
        return wrapped

    def execute(model, images, kwargs):
        limit = record['effective_batch_limit']
        if len(images) > limit:
            return [result for i in range(0,len(images),limit)
                    for result in execute(model, images[i:i+limit], kwargs)]
        try:
            began = time.perf_counter()
            results = original_predict(model, images, **kwargs)
            record['predict_seconds'] += time.perf_counter()-began
            if len(results) != len(images):
                raise ValueError('Batch result inventory mismatch')
            record['model_batches'] += 1
            record['roi_images'] += len(images)
            key = str(len(images))
            record['batch_sizes'][key] = record['batch_sizes'].get(key, 0)+1
            return [r.cpu() for r in results]
        except Exception as error:
            import torch
            if not isinstance(error, torch.cuda.OutOfMemoryError) or len(images) == 1:
                raise
            record['oom_retries'].append({'size': len(images), 'error': type(error).__name__})
            torch.cuda.empty_cache()
            split = len(images)//2
            record['effective_batch_limit'] = min(record['effective_batch_limit'], split)
            return execute(model, images[:split], kwargs)+execute(model, images[split:], kwargs)

    def predict(model, image=None, *args, **kwargs):
        if args or kwargs.get('classes') != [0] or kwargs.get('imgsz') != 320:
            return original_predict(model, image, *args, **kwargs)
        active = owner['active']
        if active is None or active.current is None:
            raise ValueError('Missing ordered source frame for batch')
        roi = crop_coordinates(image, active.current[2])
        active.enable()
        key = (id(model), roi, tuple(sorted((k, repr(v)) for k, v in kwargs.items())))
        if key not in active.cache:
            x1, y1, x2, y2 = roi
            images = [p[2][y1:y2, x1:x2] for p in active.batch]
            active.cache[key] = execute(model, images, kwargs)
            record['max_batch_frames'] = max(record['max_batch_frames'], len(images))
        record['delivered_calls'] += 1
        return [active.cache[key][active.offset]]

    def adapt(source_text):
        text = original_adapt(source_text)
        # Only three unused unconditional drawing statements. Tracking code stays byte-identical.
        lines = text.splitlines(keepends=True)
        found = 0
        for i, line in enumerate(lines):
            if (line.strip().startswith('frame = cv2.rectangle(frame, (x1, y1), (x2, y2)')
                    or line.strip().startswith("cv2.putText(frame, f'{self.makes}/{self.attempts}")):
                lines[i] = line[:len(line)-len(line.lstrip())]+'pass  # Headless analysis\n'
                found += 1
        if found != 3:
            raise ValueError('Pinned headless adapter mismatch')
        return ''.join(lines)

    cv2.VideoCapture, YOLO.predict, upstream_adapter.adapt_source = capture, predict, adapt
    try:
        yield record
    finally:
        try:
            for cap in captures:
                cap.release()
        finally:
            cv2.VideoCapture, YOLO.predict, upstream_adapter.adapt_source = original_capture, original_predict, original_adapt
            torch_utils.NUM_THREADS = original_thread_budget
            torch.set_num_threads(original_torch_threads)
