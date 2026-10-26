"""Untrained geometric hypotheses, never semantic rim detection or goal decisions."""
import math

import cv2
import numpy as np

from .scene import iou


PROFILE = {
    'version': 'rim-observability-v1',
    'crop_half_side_hoop_widths': .75,
    'major_axis_hoop_widths': [.5, 1.5],
    'center_distance_hoop_widths': .5,
    'min_contour_points': 12,
    'pixel_uncertainty': 1.,
    'min_minor_axis_pixel_uncertainties': 4.,
    'max_rms_pixel_uncertainties': 1.,
    'max_p95_pixel_uncertainties': 2.,
    'max_angular_gap_degrees': 90.,
    'canny_low_high_ratio': .5,
}


def crop_for_hoop(box, width, height):
    x1, y1, x2, y2 = map(float, box)
    w = x2-x1
    if w <= 0 or y2 <= y1 or width <= 0 or height <= 0:
        raise ValueError('Positive hoop and frame dimensions required')
    cx = (x1+x2)/2
    half = PROFILE['crop_half_side_hoop_widths']*w
    crop = [max(0, math.floor(cx-half)), max(0, math.floor(y1-half)),
            min(width, math.ceil(cx+half)), min(height, math.ceil(y1+half))]
    if crop[2] <= crop[0] or crop[3] <= crop[1]:
        raise ValueError('Hoop neighborhood outside source')
    return crop


def ellipse_hypothesis(points, hoop_box):
    """Return fit diagnostics and explicit rejection reasons in source coordinates.

    Residual is radial distance to the ellipse, not just algebraic conic error.
    A fit passing these checks remains an ellipse with unknown semantic identity.
    """
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(points) < PROFILE['min_contour_points'] or not np.isfinite(points).all():
        return {'accepted_as_hypothesis': False, 'reasons': ['insufficient_finite_points']}
    if len(np.unique(points, axis=0)) < PROFILE['min_contour_points']:
        return {'accepted_as_hypothesis': False, 'reasons': ['insufficient_unique_points']}
    if np.linalg.matrix_rank(points-points.mean(axis=0)) < 2:
        return {'accepted_as_hypothesis': False, 'reasons': ['collinear_points']}
    try:
        (cx, cy), (diameter_x, diameter_y), angle = cv2.fitEllipse(points)
    except cv2.error:
        return {'accepted_as_hypothesis': False, 'reasons': ['ellipse_fit_failed']}
    if not all(math.isfinite(v) for v in (cx, cy, diameter_x, diameter_y, angle)) or min(diameter_x, diameter_y) <= 0:
        return {'accepted_as_hypothesis': False, 'reasons': ['degenerate_ellipse']}
    theta = math.radians(angle)
    rotation = np.array([[math.cos(theta), math.sin(theta)],
                         [-math.sin(theta), math.cos(theta)]])
    local = (points-np.array([cx, cy])) @ rotation.T
    normalized = local/np.array([diameter_x/2, diameter_y/2])
    radius = np.linalg.norm(local, axis=1)
    q = np.linalg.norm(normalized, axis=1)
    residual = np.abs(radius*(1-1/np.maximum(q, 1e-12)))
    angles = np.sort(np.mod(np.arctan2(normalized[:, 1], normalized[:, 0]), 2*math.pi))
    max_gap = math.degrees(float(np.max(np.diff(np.r_[angles, angles[0]+2*math.pi]))))
    rms = float(np.sqrt(np.mean(residual**2)))
    p95 = float(np.quantile(residual, .95))
    w = hoop_box[2]-hoop_box[0]
    if w <= 0:
        raise ValueError('Positive hoop width required')
    major, minor = max(diameter_x, diameter_y), min(diameter_x, diameter_y)
    center_distance = math.hypot(cx-(hoop_box[0]+hoop_box[2])/2, cy-hoop_box[1])/w
    reasons = []
    lower, upper = PROFILE['major_axis_hoop_widths']
    if not lower <= major/w <= upper:
        reasons.append('implausible_size_relative_to_hoop')
    if center_distance > PROFILE['center_distance_hoop_widths']:
        reasons.append('outside_hoop_top_neighborhood')
    if minor < PROFILE['min_minor_axis_pixel_uncertainties']*PROFILE['pixel_uncertainty']:
        reasons.append('front_back_separation_unresolved')
    if rms > PROFILE['max_rms_pixel_uncertainties']*PROFILE['pixel_uncertainty'] or p95 > PROFILE['max_p95_pixel_uncertainties']*PROFILE['pixel_uncertainty']:
        reasons.append('poor_geometric_fit')
    if max_gap > PROFILE['max_angular_gap_degrees']:
        reasons.append('incomplete_angular_support')
    return {'accepted_as_hypothesis': not reasons, 'reasons': reasons,
            'semantic_identity': 'unknown', 'center': [cx, cy],
            'axis_diameters': [diameter_x, diameter_y], 'angle_degrees': angle,
            'major_axis_hoop_widths': major/w, 'minor_axis_pixels': minor,
            'center_distance_hoop_widths': center_distance,
            'radial_rms_pixels': rms, 'radial_p95_pixels': p95,
            'max_angular_gap_degrees': max_gap, 'point_count': len(points)}


def inspect_static_edges(image, hoop_box, crop):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    magnitude = np.clip(np.hypot(gx, gy), 0, 255).astype(np.uint8)
    high, _ = cv2.threshold(magnitude, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if not magnitude.any():
        edges = np.zeros_like(gray)
    else:
        edges = cv2.Canny(gray, high*PROFILE['canny_low_high_ratio'], high, L2gradient=True)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    fits = []
    for contour in contours:
        if len(contour) < PROFILE['min_contour_points']:
            continue
        points = contour.reshape(-1, 2).astype(np.float32)+np.array(crop[:2], dtype=np.float32)
        fits.append(ellipse_hypothesis(points, hoop_box))
    accepted = [fit for fit in fits if fit['accepted_as_hypothesis']]
    return {'status': 'hypotheses_only' if accepted else 'unavailable',
            'rim_identity_verified': False, 'occlusion_order': 'unavailable',
            'net_required': False, 'canny_high_threshold': high,
            'contour_fit_count': len(fits), 'accepted_hypothesis_count': len(accepted),
            'fits': fits}, edges


def containment_counterfactual(attempt, fps):
    """A normalized diagnostic; do not turn fit uncertainty into a miss."""
    above, below = attempt['above'], attempt['below']
    gap = below['frame']-above['frame']
    if gap <= 0 or fps <= 0:
        raise ValueError('Chronological bracket and positive fps required')
    clearance = attempt['horizontal_clearance_ball_radii']
    return {'time_seconds': attempt['time_seconds'],
            'hoop_id': attempt['hoop_id'], 'ball_id': attempt['ball_id'],
            'bracket_gap_frames': gap, 'bracket_gap_seconds': gap/fps,
            'horizontal_clearance_ball_radii': clearance,
            'whole_ball_containment_at_proxy': clearance >= 1,
            'semantic_outcome': 'unchanged_candidate',
            'plane': 'original_hoop_box_top_proxy', 'net_required': False}


def map_hoop_box(hoop, scenes, overlap_threshold):
    """Map quantized upstream geometry; do not assume its IDs match scene order."""
    box = [hoop['x']-hoop['w']/2, hoop['y']-hoop['h']/2,
           hoop['x']+hoop['w']/2, hoop['y']+hoop['h']/2]
    matches = [i for i, scene in enumerate(scenes) if iou(box, scene['xyxy']) >= overlap_threshold]
    if len(matches) != 1:
        raise ValueError('Upstream hoop does not map uniquely to automatic scene')
    return matches[0]
