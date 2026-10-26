"""Bounded background registration; all transforms map reference to current."""
import math
from pathlib import Path

import cv2
import numpy as np

from .media import inspect_video, sha256, write_json


class MotionUnavailable(ValueError):
    pass


def validate_motion(matrix, width, height, profile):
    if matrix is None or not np.isfinite(matrix).all():
        raise MotionUnavailable('No finite background transform')
    scale = math.hypot(matrix[0, 0], matrix[1, 0])
    angle = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
    center = np.array([width / 2, height / 2])
    delta = matrix[:, :2] @ center + matrix[:, 2] - center
    if (abs(scale - 1) > profile['maximum_scale_change']
            or abs(angle) > profile['maximum_rotation_degrees']
            or abs(delta[0]) / width > profile['maximum_translation_fraction']
            or abs(delta[1]) / height > profile['maximum_translation_fraction']):
        raise MotionUnavailable('Motion exceeds the small-shake limits')
    return {'scale': scale, 'rotation_degrees': angle, 'center_dx': float(delta[0]),
            'center_dy': float(delta[1])}


class BackgroundRegistration:
    def __init__(self, reference, hoop_boxes, profile):
        self.profile = profile
        self.height, self.width = reference.shape[:2]
        self.ratio = min(1., profile['tracking_width'] / self.width)
        self.size = (round(self.width * self.ratio), round(self.height * self.ratio))
        self.reference = self.gray(reference)
        h, w = self.reference.shape
        mask = np.zeros((h, w), np.uint8)
        centers = [(box[0]+box[2])/2 for box in hoop_boxes]
        center = float(np.median(centers))*self.ratio if centers else w/2
        half = w*profile['background_horizontal_fraction']/2
        mask[:int(h * profile['background_top_fraction']),
             max(0, int(center-half)):min(w, math.ceil(center+half))] = 255
        for box in hoop_boxes:
            x1, y1, x2, y2 = np.array(box) * self.ratio
            bw, bh = x2 - x1, y2 - y1
            mask[max(0, int(y1 - 3*bh)):min(h, math.ceil(y2 + bh)),
                 max(0, int(x1 - 2*bw)):min(w, math.ceil(x2 + 2*bw))] = 0
        self.points = cv2.goodFeaturesToTrack(self.reference, profile['max_corners'], .01, 6, mask=mask)
        if self.points is None or len(self.points) < profile['minimum_inliers']:
            raise MotionUnavailable('Insufficient background texture')

    def gray(self, frame):
        return cv2.cvtColor(cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)

    def estimate(self, frame):
        current = self.gray(frame)
        options = {'winSize': (21, 21), 'maxLevel': 3,
                   'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01)}
        forward, s1, _ = cv2.calcOpticalFlowPyrLK(self.reference, current, self.points, None, **options)
        if forward is None:
            raise MotionUnavailable('No forward background tracks')
        backward, s2, _ = cv2.calcOpticalFlowPyrLK(current, self.reference, forward, None, **options)
        if backward is None:
            raise MotionUnavailable('No backward background tracks')
        valid = ((s1.ravel() == 1) & (s2.ravel() == 1)
                 & (np.linalg.norm(backward-self.points, axis=2).ravel()
                    <= self.profile['maximum_forward_backward_pixels']))
        a, b = self.points.reshape(-1, 2)[valid], forward.reshape(-1, 2)[valid]
        if len(a) < self.profile['minimum_inliers']:
            raise MotionUnavailable('Too few consistent background tracks')
        matrix, inliers = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC,
            ransacReprojThreshold=self.profile['ransac_pixels'], maxIters=2000, confidence=.99)
        if matrix is None or inliers is None:
            raise MotionUnavailable('Background fit unavailable')
        good = inliers.ravel().astype(bool)
        if good.sum() < self.profile['minimum_inliers'] or good.mean() < self.profile['minimum_inlier_fraction']:
            raise MotionUnavailable('Background consensus too weak')
        span = np.ptp(a[good], axis=0)
        if (span[0] < self.size[0]*self.profile['minimum_span_x_fraction']
                or span[1] < self.size[1]*self.profile['minimum_span_y_fraction']):
            raise MotionUnavailable('Background support is too localized')
        residual = np.linalg.norm(a @ matrix[:, :2].T + matrix[:, 2] - b, axis=1)
        matrix[:, 2] /= self.ratio
        motion = validate_motion(matrix, self.width, self.height, self.profile)
        return matrix, {**motion, 'inlier_points': int(good.sum()), 'inlier_fraction': float(good.mean()),
                        'residual_p95_tracking_pixels': float(np.percentile(residual[good], 95))}


def stabilize(video, out, scene, profile):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    info = inspect_video(video)
    cap = cv2.VideoCapture(str(video)); ok, first = cap.read()
    if not ok:
        cap.release()
        raise MotionUnavailable('Cannot decode reference frame')
    writer = None; rows = []; reason = None
    target = out/'stabilized.avi'
    try:
        boxes = [box['xyxy'] for s in scene['samples'] for box in s['boxes']]
        registration = BackgroundRegistration(first, boxes, profile)
        writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*'FFV1'), info['fps'],
                                 (info['width'], info['height']))
        if not writer.isOpened():
            raise RuntimeError('Lossless FFV1 writer unavailable')
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok: break
            matrix, evidence = registration.estimate(frame)
            corrected = cv2.warpAffine(frame, cv2.invertAffineTransform(matrix),
                (info['width'], info['height']), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            writer.write(corrected)
            rows.append({'frame': index, 'time_seconds': index/info['fps'],
                         'reference_to_current': matrix.tolist(), **evidence})
            index += 1
            if index % 600 == 0: print('stabilized', index, 'frames', flush=True)
        if index != round(info['fps'] * info['duration_seconds']):
            raise MotionUnavailable('Source frame count incomplete')
    except MotionUnavailable as error:
        reason = str(error)
    finally:
        cap.release()
        if writer is not None: writer.release()
    result = {'status': 'unsupported' if reason else 'completed', 'reason': reason,
              'source': info, 'source_sha256': sha256(video), 'profile': profile,
              'frames': len(rows), 'transforms': rows, 'output': str(target),
              'original_seconds_offset': 0, 'frame_mapping': 'same_index_same_fps',
              'encoding': 'FFV1 lossless; interpolation occurs only in geometric correction',
              'analysis_audio': 'omitted; review uses original source audio'}
    if not reason:
        result['output_info'] = inspect_video(target)
        result['output_sha256'] = sha256(target)
        if result['output_info']['fps'] != info['fps']:
            raise ValueError('Corrected video frame rate changed')
    write_json(out/'stabilization.json', result)
    return result
