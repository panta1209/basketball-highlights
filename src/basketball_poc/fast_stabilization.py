"""Same camera registration and warp; independently versioned lossless encoding."""
from pathlib import Path
from fractions import Fraction
import subprocess
import time

import cv2
import numpy as np

from .camera_shake import BackgroundRegistration, MotionUnavailable
from .media import inspect_video, sha256, write_json


class LosslessWriter:
    def __init__(self, target, width, height, fps, profile):
        if (profile['codec'] != 'libx264rgb' or profile['crf'] != 0
                or profile['preset'] != 'ultrafast' or profile['keyframe_interval'] != 1
                or type(profile['encoder_threads']) is not int or profile['encoder_threads'] < 1):
            raise ValueError('Lossless RGB encoding profile required')
        self.shape = (height, width, 3)
        self.log = Path(target).with_suffix('.encoder.log').open('xb')
        self.process = subprocess.Popen(['ffmpeg', '-v', 'error', '-nostdin', '-n',
            '-f', 'rawvideo', '-pixel_format', 'bgr24', '-video_size', f'{width}x{height}',
            '-framerate', str(Fraction(fps).limit_denominator(1000000)), '-i', 'pipe:0', '-an', '-c:v', 'libx264rgb',
            '-preset', profile['preset'], '-crf', '0', '-g', '1',
            '-threads', str(profile['encoder_threads']), str(target)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log)

    def write(self, frame):
        if frame.shape != self.shape or frame.dtype != np.uint8:
            raise ValueError('Unexpected frame shape/type')
        self.process.stdin.write(frame.tobytes())

    def close(self):
        try:
            self.process.stdin.close()
            if self.process.wait() != 0:
                raise ValueError('Lossless video encoder failed; see encoder log')
        finally:
            self.log.close()


def stabilize(video, out, scene, motion_profile, encoding_profile):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    info = inspect_video(video)
    capture = cv2.VideoCapture(str(video))
    ok, first = capture.read()
    if not ok:
        capture.release()
        raise MotionUnavailable('Cannot decode reference frame')
    writer = None; rows = []; reason = None
    target = out/'stabilized.avi'
    timings = {'registration_seconds': 0., 'warp_seconds': 0., 'encoding_write_seconds': 0.}
    began = time.perf_counter()
    try:
        boxes = [box['xyxy'] for sample in scene['samples'] for box in sample['boxes']]
        registration = BackgroundRegistration(first, boxes, motion_profile)
        writer = LosslessWriter(target, info['width'], info['height'], info['fps'], encoding_profile)
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            start = time.perf_counter()
            matrix, evidence = registration.estimate(frame)
            timings['registration_seconds'] += time.perf_counter()-start
            start = time.perf_counter()
            corrected = cv2.warpAffine(frame, cv2.invertAffineTransform(matrix),
                (info['width'], info['height']), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            timings['warp_seconds'] += time.perf_counter()-start
            start = time.perf_counter(); writer.write(corrected)
            timings['encoding_write_seconds'] += time.perf_counter()-start
            rows.append({'frame': index, 'time_seconds': index/info['fps'],
                         'reference_to_current': matrix.tolist(), **evidence})
            index += 1
            if index % 300 == 0:
                print('stabilized', index, 'frames', flush=True)
        if index != round(info['fps']*info['duration_seconds']):
            raise MotionUnavailable('Source frame count incomplete')
    except MotionUnavailable as error:
        reason = str(error)
    finally:
        capture.release()
        if writer is not None:
            writer.close()
    result = {'status': 'unsupported' if reason else 'completed', 'reason': reason,
        'source': info, 'source_sha256': sha256(video), 'profile': motion_profile,
        'frames': len(rows), 'transforms': rows, 'output': str(target),
        'original_seconds_offset': 0, 'frame_mapping': 'same_index_same_fps',
        'encoding': 'lossless RGB H264; unchanged geometric interpolation',
        'encoding_profile': encoding_profile, 'timings': timings,
        'elapsed_seconds': time.perf_counter()-began,
        'analysis_audio': 'omitted; review uses original source audio'}
    if not reason:
        result.update(output_info=inspect_video(target), output_sha256=sha256(target))
        if result['output_info']['fps'] != info['fps']:
            raise ValueError('Corrected video frame rate changed')
    write_json(out/'stabilization.json', result)
    return result
