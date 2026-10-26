"""Automatic fixed-camera scene geometry; no video-specific pixel thresholds."""
import math
from statistics import median


def iou(a, b):
    overlap = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    area_a = (a[2]-a[0])*(a[3]-a[1])
    area_b = (b[2]-b[0])*(b[3]-b[1])
    union = area_a + area_b - overlap
    return overlap / union if union > 0 else 0.0


def consensus_box(observations):
    return [median(o['xyxy'][index] for o in observations) for index in range(4)]


def stable_hoops(samples, profile, jitter_estimator='nearest-rank-v1'):
    """Require repeated frame support and small dimensionless center jitter."""
    if jitter_estimator not in ('nearest-rank-v1', 'linear-p95-v1'):
        raise ValueError('Unknown scene jitter estimator')
    groups = []
    for sample in samples:
        assigned = set()
        for box in sorted(sample['boxes'], key=lambda b: b['confidence'], reverse=True):
            scores = [(iou(box['xyxy'], consensus_box(group)), index) for index, group in enumerate(groups)]
            similarity, index = max(scores, default=(0, -1))
            observation = {**box, 'frame': sample['frame']}
            if similarity >= profile['cluster_iou']:
                if index in assigned:
                    continue  # Duplicate boxes in one source frame add no extra hoop or support.
                groups[index].append(observation)
                assigned.add(index)
            else:
                groups.append([observation])
                assigned.add(len(groups)-1)
    accepted = []
    for group in groups:
        support = len({o['frame'] for o in group})
        if support < profile['stable_min_observations'] or support < len(samples)*profile['stable_support_fraction']:
            continue
        box = consensus_box(group)
        width, height = box[2]-box[0], box[3]-box[1]
        if width <= 0 or height <= 0:
            continue
        cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
        deviations = sorted(math.hypot(((o['xyxy'][0]+o['xyxy'][2])/2-cx)/width,
                                      ((o['xyxy'][1]+o['xyxy'][3])/2-cy)/height) for o in group)
        if jitter_estimator == 'linear-p95-v1':
            position = .95*(len(deviations)-1)
            lower, upper = math.floor(position), math.ceil(position)
            jitter = deviations[lower] + (position-lower)*(deviations[upper]-deviations[lower])
        else:
            jitter = deviations[math.ceil(.95*len(deviations))-1]
        if jitter > profile['max_center_jitter_fraction']:
            continue
        accepted.append({'xyxy': box, 'confidence': median(o['confidence'] for o in group),
                         'support_samples': support, 'support_fraction': support/len(samples),
                         'center_jitter_fraction_p95': jitter})
    accepted.sort(key=lambda h: (-(h['xyxy'][2]-h['xyxy'][0])*(h['xyxy'][3]-h['xyxy'][1]), h['xyxy'][0]))
    return [{'id': f'hoop-{index+1:03d}', **hoop} for index, hoop in enumerate(accepted)]


def roi_for_hoop(box, width, height, profile):
    x1,y1,x2,y2 = box
    w,h = x2-x1,y2-y1
    return [max(0,math.floor(x1-profile['roi_side_margin_widths']*w)),
            max(0,math.floor(y1-profile['roi_above_margin_heights']*h)),
            min(width,math.ceil(x2+profile['roi_side_margin_widths']*w)),
            min(height,math.ceil(y2+profile['roi_below_margin_heights']*h))]


def merge_boxes(boxes, threshold):
    """Class-aware NMS prevents duplicate balls from overlapping auto regions."""
    selected = []
    for box in sorted(boxes, key=lambda b: b['confidence'], reverse=True):
        if not any(box['class_id'] == other['class_id'] and iou(box['xyxy'],other['xyxy']) >= threshold
                   for other in selected):
            selected.append(box)
    return selected


def seconds_to_frames(seconds, fps):
    if not math.isfinite(seconds) or not math.isfinite(fps) or seconds <= 0 or fps <= 0:
        raise ValueError('Positive finite seconds and fps required')
    # Floor ensures an allowed gap never exceeds the declared seconds.
    return math.floor(seconds*fps+1e-9)
