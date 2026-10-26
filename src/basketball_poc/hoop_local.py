"""Auditable hoop-local inputs, without result classification or identity repair."""
import math

import cv2
import numpy as np


def window_frames(seconds, fps, count, before=2.0, after=2.0):
    if fps <= 0 or count <= 0 or not 0 <= seconds <= count / fps:
        raise ValueError('Invalid source time')
    lo = max(0, math.ceil((seconds-before)*fps))
    hi = min(count-1, math.floor((seconds+after)*fps))
    if lo > hi:
        raise ValueError('Empty window')
    return list(range(lo, hi+1))


def target_hoop(geometry, target_id):
    matches = [h for h in geometry['hoops'] if h['id'] == target_id]
    if len(matches) > 1:
        raise ValueError('Duplicate target hoop identity')
    return matches[0] if matches else None


def crop_transform(hoop, width, height, side=384, field_widths=7.0):
    box = hoop['xyxy']
    if len(box) != 4 or not all(math.isfinite(v) for v in box):
        raise ValueError('Invalid hoop')
    x1,y1,x2,y2 = box
    if x2 <= x1 or y2 <= y1 or min(width,height,side,field_widths) <= 0:
        raise ValueError('Invalid crop geometry')
    cx, cy = (x1+x2)/2, y1
    length = (x2-x1)*field_widths
    rect = [cx-length/2,cy-length/2,cx+length/2,cy+length/2]
    scale = side/length
    matrix = [[scale,0,(side-1)/2-scale*cx], [0,scale,(side-1)/2-scale*cy]]
    visible_area = max(0,min(width,rect[2])-max(0,rect[0])) * max(0,min(height,rect[3])-max(0,rect[1]))
    return {'field_xyxy': rect, 'affine_source_to_local': matrix,
            'padding_area_fraction': 1-visible_area/(length*length),
            'source_hoop_width_pixels': x2-x1, 'local_hoop_width_pixels': side/field_widths,
            'geometry_source': hoop['geometry_source'], 'hoop_xyxy': list(box),
            'last_observed_seconds': hoop['last_observed_seconds'], 'target_hoop_id': hoop['id']}


def crop_image(image, transform, side=384):
    if transform is None:
        return np.zeros((side,side,3), dtype=np.uint8)
    return cv2.warpAffine(image, np.asarray(transform['affine_source_to_local'],dtype=np.float64),
                          (side,side), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def local_box(box, transform):
    matrix = np.array(transform['affine_source_to_local'])
    corners = np.array([[box[0],box[1],1],[box[2],box[3],1]]) @ matrix.T
    return corners.reshape(-1).tolist()


def contained(box, transform, side=384):
    mapped = local_box(box,transform)
    return 0 <= mapped[0] < mapped[2] <= side-1 and 0 <= mapped[1] < mapped[3] <= side-1


def longest_run(values, desired):
    longest = current = 0
    for value in values:
        current = current+1 if bool(value) == desired else 0
        longest = max(longest,current)
    return longest


def describe(values):
    if not values:
        return None
    array = np.asarray(values)
    return {'min':float(array.min()), 'median':float(np.median(array)),
            'p95':float(np.quantile(array,.95)), 'max':float(array.max())}


def quality(rows, event_seconds, fps):
    valid = [r['valid'] for r in rows]
    core = [r for r in rows if abs(r['time_seconds']-event_seconds) <= .5+1e-12]
    available = [r for r in rows if r['valid']]
    balls = [r for r in available if r['event_ball'] is not None]
    return {'frames':len(rows), 'valid_frames':sum(valid), 'valid_fraction':sum(valid)/len(rows),
            'core_frames':len(core), 'core_valid_frames':sum(r['valid'] for r in core),
            'core_valid_fraction':sum(r['valid'] for r in core)/len(core),
            'longest_invalid_seconds':longest_run(valid,False)/fps,
            'longest_valid_seconds':longest_run(valid,True)/fps,
            'detected_frames':sum(r['transform']['geometry_source']=='detected' for r in available),
            'propagated_frames':sum(r['transform']['geometry_source']!='detected' for r in available),
            'hoop_width_source_pixels':describe([r['transform']['source_hoop_width_pixels'] for r in available]),
            'padding_area_fraction':describe([r['transform']['padding_area_fraction'] for r in available]),
            'event_ball_observed_frames':len(balls), 'event_ball_fully_in_crop_frames':sum(r['event_ball_in_crop'] for r in balls),
            'event_ball_source_width_pixels':describe([r['event_ball']['ball_xyxy'][2]-r['event_ball']['ball_xyxy'][0] for r in balls]),
            'event_ball_local_width_pixels':describe([r['event_ball_local_xyxy'][2]-r['event_ball_local_xyxy'][0] for r in balls]),
            'event_ball_fullframe_224_width_pixels':describe([(r['event_ball']['ball_xyxy'][2]-r['event_ball']['ball_xyxy'][0])*224/r['source_width'] for r in balls]),
            'core_event_ball_observed_frames':sum(r['event_ball'] is not None for r in core),
            'source_coverage_only_not_outcome': True}
