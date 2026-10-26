"""Explicit FP32 YOLO preprocessing and decoding for portable model validation."""
import cv2
import numpy as np


def prepare(image, size=320):
    height, width = image.shape[:2]
    scale = min(size/width, size/height)
    resized = (round(width*scale), round(height*scale))
    left, top = round((size-resized[0])/2-.1), round((size-resized[1])/2-.1)
    right, bottom = size-resized[0]-left, size-resized[1]-top
    frame = cv2.resize(image, resized, interpolation=cv2.INTER_LINEAR)
    frame = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT,
                              value=(114, 114, 114))
    tensor = np.ascontiguousarray(frame[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)/255.
    return tensor, {'scale': scale, 'left': left, 'top': top, 'width': width, 'height': height}


def overlap(box, boxes):
    left = np.maximum(box[:2], boxes[:, :2])
    right = np.minimum(box[2:4], boxes[:, 2:4])
    area = np.prod(np.maximum(right-left, 0), axis=1)
    size = np.prod(np.maximum(box[2:4]-box[:2], 0))
    others = np.prod(np.maximum(boxes[:, 2:4]-boxes[:, :2], 0), axis=1)
    return area/np.maximum(size+others-area, 1e-9)


def decode(output, geometry, classes=2, conf=.15, iou=.7, max_det=300):
    values = np.asarray(output)
    if values.ndim != 3 or values.shape[0] != 1 or values.shape[1] != 4+classes:
        raise ValueError('Unexpected YOLO output layout')
    predictions = values[0].T
    labels = predictions[:, 4:].argmax(axis=1)
    scores = predictions[np.arange(len(predictions)), labels+4]
    valid = np.isfinite(predictions).all(axis=1) & (scores >= conf)
    predictions, labels, scores = predictions[valid], labels[valid], scores[valid]
    boxes = np.column_stack((predictions[:, :2]-predictions[:, 2:4]/2,
                             predictions[:, :2]+predictions[:, 2:4]/2))
    order = np.argsort(-scores, kind='stable')
    selected = []
    while len(order) and len(selected) < max_det:
        current = order[0]
        selected.append(current)
        remaining = order[1:]
        order = remaining[(labels[remaining] != labels[current]) |
                          (overlap(boxes[current], boxes[remaining]) <= iou)]
    result = []
    for index in selected:
        box = boxes[index].astype(float)
        box[[0, 2]] = np.clip((box[[0, 2]]-geometry['left'])/geometry['scale'], 0, geometry['width'])
        box[[1, 3]] = np.clip((box[[1, 3]]-geometry['top'])/geometry['scale'], 0, geometry['height'])
        result.append({'box': box.tolist(), 'confidence': float(scores[index]), 'class_id': int(labels[index])})
    return result
