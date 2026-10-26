"""Auxiliary named relation evidence, never a required goal or net condition."""
import math

from .relation_probe import center_in_box

MODEL_SHA = 'f7a1445dda0c6915273753dd685aea478924db970af2a9f2649a1cd5a6da875c'
PROFILE = {
    'version': 'named-relation-probe-v1', 'imgsz': 640, 'conf': .25, 'iou': .5,
    'max_det': 300, 'device': 'cpu', 'rect': True, 'half': False,
    'window_half_seconds': .5, 'arms': ['full_frame', 'automatic_roi'],
    'relationship_name': 'ball-in-basket', 'net_required': False,
    'association': 'center_inside_unique_automatic_hoop_box',
    'missing_relationship': 'uncertain',
}


def relationship_id(names):
    matches = [int(k) for k, v in names.items() if v == PROFILE['relationship_name']]
    if len(matches) != 1:
        raise ValueError('Exactly one embedded ball-in-basket class required')
    return matches[0]


def restore_rows(rows, names, region):
    """YOLO result xyxy already undoes letterbox, relative to the input crop."""
    x1, y1, x2, y2 = region
    if x2 <= x1 or y2 <= y1:
        raise ValueError('Positive region required')
    result = []
    for row in rows:
        if len(row) != 6 or not all(math.isfinite(float(x)) for x in row):
            raise ValueError('Finite xyxy/conf/class row required')
        a, b, c, d, score, cls = map(float, row)
        if c <= a or d <= b or not 0 <= score <= 1 or cls != int(cls) or int(cls) not in names:
            raise ValueError('Invalid model row')
        result.append({'class_id': int(cls), 'class_name': names[int(cls)], 'score': score,
                       'xyxy': [a+x1, b+y1, c+x1, d+y1]})
    return result


def associated_relations(detections, scenes, target_index):
    """Ambiguous overlapping hoops give no positive association."""
    if not 0 <= target_index < len(scenes):
        raise ValueError('Target scene index out of range')
    result = []
    for detection in detections:
        if detection['class_name'] != PROFILE['relationship_name']:
            continue
        owners = [i for i, scene in enumerate(scenes) if center_in_box(detection, scene['xyxy'])]
        if owners == [target_index]:
            result.append(detection)
    return result


def evidence_state(count):
    if not isinstance(count, int) or count < 0:
        raise ValueError('Nonnegative observation count required')
    return 'positive_auxiliary_observation' if count else 'uncertain'
