"""Analysis-local corrected frames with small transform/hash evidence, no video writer."""
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys
import time

import cv2
import numpy as np

from . import media
from .camera_shake import BackgroundRegistration, MotionUnavailable


def frame_source_identity(source_digest, profile):
    payload = {'kind': 'dynamic-corrected-frames-v1', 'original_source_sha256': source_digest,
               'profile': profile, 'reader_sha256': media.sha256(__file__),
               'registration_sha256': media.sha256(Path(__file__).with_name('camera_shake.py'))}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class CorrectedFrames:
    def __init__(self, source, scene, profile, saved=None):
        self.source = Path(source).resolve(strict=True)
        self.info = media.inspect_video(source)
        self.digest = media.sha256(source)
        self.profile = profile
        self.identity = frame_source_identity(self.digest, profile)
        self.capture_type = cv2.VideoCapture
        self.rows = {} if saved is None else {r['frame']: r for r in saved['transforms']}
        self.replay = saved is not None
        self.registration = None
        self.reads = 0; self.captures = []
        self.timings = {'registration_seconds': 0., 'warp_seconds': 0.}
        if saved is None:
            capture = self.capture_type(str(self.source))
            try:
                ok, frame = capture.read()
                if not ok:
                    raise MotionUnavailable('Cannot decode reference frame')
                boxes = [b['xyxy'] for sample in scene['samples'] for b in sample['boxes']]
                self.registration = BackgroundRegistration(frame, boxes, profile)
            finally:
                capture.release()

    def matches(self, source):
        return isinstance(source, (str, Path)) and Path(source).resolve() == self.source

    def correct(self, frame, index):
        row = self.rows.get(index)
        if row is None:
            if self.replay:
                raise ValueError('Missing frozen transform')
            start = time.perf_counter()
            matrix, evidence = self.registration.estimate(frame)
            self.timings['registration_seconds'] += time.perf_counter()-start
            row = {'frame': index, 'time_seconds': index/self.info['fps'],
                   'reference_to_current': matrix.tolist(), **evidence}
            self.rows[index] = row
        matrix = np.asarray(row['reference_to_current'])
        start = time.perf_counter()
        corrected = cv2.warpAffine(frame, cv2.invertAffineTransform(matrix),
            (self.info['width'], self.info['height']), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        self.timings['warp_seconds'] += time.perf_counter()-start
        digest = hashlib.sha256(corrected.tobytes()).hexdigest()
        if row.get('corrected_frame_sha256', digest) != digest:
            raise ValueError(f'Corrected frame pixels changed at {index}')
        row['corrected_frame_sha256'] = digest
        self.reads += 1
        return corrected

    def capture(self, source, *args, **kwargs):
        if not self.matches(source):
            return self.capture_type(source, *args, **kwargs)
        capture = CorrectedCapture(self, self.capture_type(str(self.source), *args, **kwargs))
        self.captures.append(capture)
        return capture

    def record(self, status='completed', reason=None):
        count = round(self.info['fps']*self.info['duration_seconds'])
        if status == 'completed' and sorted(self.rows) != list(range(count)):
            raise ValueError('Dynamic correction must cover all source frames')
        return {'status': status, 'reason': reason, 'storage': 'dynamic_frames_v1',
                'source': self.info, 'source_sha256': self.digest, 'frame_source_sha256': self.identity,
                'profile': self.profile, 'frames': len(self.rows),
                'transforms': [self.rows[i] for i in sorted(self.rows)], 'timings': self.timings,
                'corrected_frame_reads': self.reads, 'encoded_video_bytes': 0,
                'frame_mapping': 'same_index_same_fps', 'original_seconds_offset': 0,
                'frame_storage': 'current frame only; memoized transforms and pixel hashes',
                'analysis_audio': 'not decoded; original source retained for preview/export'}


class CorrectedCapture:
    def __init__(self, provider, capture):
        self.provider, self.raw = provider, capture

    def read(self):
        index = round(self.raw.get(cv2.CAP_PROP_POS_FRAMES))
        ok, frame = self.raw.read()
        return (True, self.provider.correct(frame, index)) if ok else (False, None)

    def get(self, key):
        return self.raw.get(key)

    def set(self, key, value):
        return self.raw.set(key, value)

    def isOpened(self):
        return self.raw.isOpened()

    def release(self):
        self.raw.release()


@contextmanager
def routed_frames(provider, detector_entry, allow_replay=False):
    """Use only inside an isolated analysis process, never the threaded web server."""
    old_capture, old_run, old_sha = cv2.VideoCapture, subprocess.run, media.sha256
    old_argv = sys.argv
    entry = Path(detector_entry).resolve(strict=True)

    def run(arguments, **kwargs):
        if not isinstance(arguments, (list, tuple)) or len(arguments) < 3 or Path(arguments[2]).resolve() != entry:
            return old_run(arguments, **kwargs)
        if not allow_replay and ('--detections-cache' in arguments or '--scene-cache' in arguments):
            raise ValueError('Dynamic analysis requires fresh inference')
        try:
            sys.argv = [str(entry), *map(str, arguments[3:])]
            media.sha256 = lambda path: provider.identity if provider.matches(path) else old_sha(path)
            with redirect_stdout(kwargs['stdout']), redirect_stderr(kwargs['stdout']):
                runpy.run_path(str(entry), run_name='__main__')
            return subprocess.CompletedProcess(arguments, 0)
        finally:
            media.sha256 = old_sha
            sys.argv = old_argv

    cv2.VideoCapture = provider.capture
    subprocess.run = run
    try:
        yield
    finally:
        for capture in provider.captures:
            capture.release()
        cv2.VideoCapture, subprocess.run, media.sha256, sys.argv = old_capture, old_run, old_sha, old_argv
