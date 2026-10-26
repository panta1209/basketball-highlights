"""Sampling and coordinate validation for observational rim diagnostics only."""
import math


def sample_frames(count, fps, events, diagnostics, profile):
    if count <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid source timing')
    frames = {}

    def add(seconds, reason):
        frame = round(seconds * fps)
        if 0 <= frame < count:
            frames.setdefault(frame, []).append(reason)

    step = profile['uniform_step_seconds']
    if not math.isfinite(step) or step <= 0:
        raise ValueError('Invalid uniform interval')
    for i in range(math.ceil(count / fps / step)):
        add(i * step, {'kind': 'uniform', 'time_seconds': i * step})
    for event in events:
        for offset in profile['window_offsets_seconds']:
            add(event['time_seconds'] + offset, {'kind': 'candidate', 'event_id': event['id'],
                'center_seconds': event['time_seconds'], 'offset_seconds': offset})
    for seconds in diagnostics:
        for offset in profile['window_offsets_seconds']:
            add(seconds + offset, {'kind': 'known_failure_diagnostic',
                'center_seconds': seconds, 'offset_seconds': offset})
    return [{'frame': f, 'time_seconds': f / fps, 'reasons': frames[f]} for f in sorted(frames)]


def restore_rows(rows, region, names):
    x1, y1, x2, y2 = region
    if x1 < 0 or y1 < 0 or x1 >= x2 or y1 >= y2:
        raise ValueError('Invalid source region')
    detections = []
    for row in rows:
        if len(row) != 6 or not all(math.isfinite(v) for v in row):
            raise ValueError('Invalid model row')
        a, b, c, d, score, label = row
        if (not 0 <= a < c <= x2-x1 or not 0 <= b < d <= y2-y1
            or not 0 <= score <= 1 or label != int(label) or int(label) not in names):
            raise ValueError('Invalid local model geometry/class')
        detections.append({'class_id': int(label), 'class_name': names[int(label)],
                           'confidence': score, 'xyxy': [a+x1, b+y1, c+x1, d+y1]})
    return detections
