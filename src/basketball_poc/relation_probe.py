"""Pinned RF-DETR numeric-output probe; label identity is deliberately unresolved."""
import math

import cv2
import numpy as np

PROFILE = {
    'version': 'relation-numeric-probe-v1',
    'input_size': 640,
    'mean': [.485, .456, .406],
    'std': [.229, .224, .225],
    'confidence_threshold': .25,
    'top_k': 300,
    'window_half_seconds': .5,
    'class_count': 11,
    'label_identity': 'unverified',
    'arms': ['full_frame', 'automatic_roi'],
    'net_required': False,
}


def preprocess(image):
    if image.ndim != 3 or image.shape[2] != 3 or not image.size:
        raise ValueError('Nonempty BGR image required')
    resized = cv2.resize(image, (PROFILE['input_size'], PROFILE['input_size']))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)/255
    normalized = (rgb-np.array(PROFILE['mean'], np.float32))/np.array(PROFILE['std'], np.float32)
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])


def decode(outputs, region):
    """Native sigmoid, query x class top-k (no NMS), then source-pixel inverse.

    Return numeric class ids, including the extra output column, without
    assigning names from a taxonomy that is only an evaluation display order.
    """
    if len(outputs) != 2:
        raise ValueError('Two ONNX outputs required')
    arrays = [np.asarray(x, np.float32) for x in outputs]
    if any(a.ndim != 3 or a.shape[0] != 1 for a in arrays):
        raise ValueError('Expected batch-one rank-three outputs')
    boxes = [a[0] for a in arrays if a.shape[-1] == 4]
    logits = [a[0] for a in arrays if a.shape[-1] == PROFILE['class_count']]
    if len(boxes) != 1 or len(logits) != 1 or len(boxes[0]) != len(logits[0]):
        raise ValueError('Unexpected box/logit shapes')
    boxes, logits = boxes[0], logits[0]
    if not np.isfinite(boxes).all() or not np.isfinite(logits).all():
        raise ValueError('Nonfinite model output')
    x1, y1, x2, y2 = region
    if x2 <= x1 or y2 <= y1:
        raise ValueError('Positive input region required')
    probs = 1/(1+np.exp(-np.clip(logits, -80, 80)))
    flat = probs.reshape(-1)
    if not len(flat):
        return []
    k = min(PROFILE['top_k'], len(flat))
    part = np.argpartition(flat, -k)[-k:]
    order = part[np.argsort(flat[part])[::-1]]
    detections = []
    for index in order:
        score = float(flat[index])
        if score <= PROFILE['confidence_threshold']:
            continue
        query_id, class_id = divmod(int(index), PROFILE['class_count'])
        cx, cy, w, h = boxes[query_id]
        detections.append({
            'class_id': class_id, 'query_id': query_id, 'score': score,
            'xyxy': [float(x1+(cx-w/2)*(x2-x1)), float(y1+(cy-h/2)*(y2-y1)),
                     float(x1+(cx+w/2)*(x2-x1)), float(y1+(cy+h/2)*(y2-y1))],
            'label_identity': 'unverified',
        })
    return detections


def frame_window(time_seconds, fps, frame_count):
    if not math.isfinite(time_seconds) or time_seconds < 0 or fps <= 0 or frame_count < 1:
        raise ValueError('Finite time, positive fps and frame count required')
    half = PROFILE['window_half_seconds']
    first = max(0, math.ceil((time_seconds-half)*fps))
    last = min(frame_count-1, math.floor((time_seconds+half)*fps))
    if last < first:
        raise ValueError('Candidate outside video')
    return range(first, last+1)


def center_in_box(detection, box):
    a, b, c, d = detection['xyxy']
    x, y = (a+c)/2, (b+d)/2
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]
