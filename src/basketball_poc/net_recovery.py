"""Review lost, descending near-rim tracks; net motion alone never creates an event."""
from collections import defaultdict

from .core import number

VERSION = 'lost-ball-net-v1'


def tracked_hoop(box):
    """Reproduce the pinned tracker's integer box/center and rim proxy."""
    x1, y1, x2, y2 = [int(number(v, 'hoop coordinate')) for v in box]
    if x2 <= x1 or y2 <= y1:
        raise ValueError('Automatic hoop box must have positive area')
    w, h = x2-x1, y2-y1
    return {'x': int(x1+w/2), 'y': int(y1+h/2), 'w': w, 'h': h}


def recover_lost_tracks(associations, decisions, hoops, existing_events, *,
                        fps, duration_seconds, candidate_profile, net_window_seconds):
    """Return review events and rejection reasons using detector evidence only.

    Association frame numbers are zero-based. A missing below-rim observation
    remains missing: continuation estimates time/position, not a confirmed goal.
    """
    fps = number(fps, 'fps', 1e-12)
    duration = number(duration_seconds, 'duration', 1e-12)
    ttl = number(candidate_profile['track_ttl_seconds'], 'ttl', 1e-12)
    history = number(candidate_profile['history_seconds'], 'history', 1e-12)
    window = number(net_window_seconds, 'net window', 1e-12)
    tracks = defaultdict(list)
    for point in associations:
        if type(point['ball_id']) is not int or type(point['frame']) is not int:
            raise ValueError('Track IDs and source frames must be integers')
        frame = number(point['frame'], 'frame')
        seconds = number(point['time_seconds'], 'observation time')
        if abs(seconds-frame/fps) > 1e-6 or seconds >= duration:
            raise ValueError('Association timing differs from the source')
        for key in ('x', 'y', 'w', 'h', 'conf'):
            number(point[key], key, 1e-12 if key in ('w', 'h') else 0)
        track = tracks[point['ball_id']]
        if track and point['frame'] <= track[-1]['frame']:
            raise ValueError('Track observations must be strictly chronological')
        track.append(point)

    resolved = {(d['ball_id'], d['hoop_id']) for d in decisions
                if d.get('decision') in ('candidate', 'outside_projection')}
    rejected, proposals = [], []
    for ball_id, track in tracks.items():
        last = track[-1]
        recent = [p for p in track if last['time_seconds']-p['time_seconds'] <= history+1e-12]
        for hoop_id, scene_hoop in enumerate(hoops):
            reason = None
            hoop = tracked_hoop(scene_hoop['xyxy'])
            plane = hoop['y']-hoop['h']/2
            left, right = hoop['x']-hoop['w']/2, hoop['x']+hoop['w']/2
            previous = recent[-2] if len(recent) >= 2 else None
            if (ball_id, hoop_id) in resolved:
                reason = 'already_resolved_crossing'
            elif len(recent) < 3:
                reason = 'insufficient_shot_history'
            elif any(b['time_seconds']-a['time_seconds'] > ttl+1e-12
                     for a, b in zip(recent, recent[1:])):
                reason = 'discontinuous_shot_history'
            elif last['y'] <= previous['y'] or previous['y'] > plane or last['y'] > plane:
                reason = 'not_descending_above_rim'
            elif any(a['y'] <= plane < b['y'] for a, b in zip(recent, recent[1:])):
                reason = 'observed_prior_descent_below_rim'
            elif last['y']+last['h']/2 < plane:
                reason = 'ball_not_reaching_rim'
            elif last['w'] > hoop['w'] or last['h'] > hoop['h']:
                reason = 'ball_larger_than_hoop'
            elif not left < last['x'] < right:
                reason = 'last_ball_outside_hoop'
            elif last['time_seconds']+max(ttl, window) >= duration:
                reason = 'incomplete_followup_window'
            if reason:
                rejected.append({'ball_id': ball_id, 'hoop_id': hoop_id, 'reason': reason})
                continue

            ratio = (plane-last['y'])/(last['y']-previous['y'])
            projected_x = last['x']+ratio*(last['x']-previous['x'])
            delay = ratio*(last['time_seconds']-previous['time_seconds'])
            if delay > ttl or not left < projected_x < right:
                rejected.append({'ball_id': ball_id, 'hoop_id': hoop_id,
                                 'reason': 'continuation_outside_hoop_or_time_limit'})
                continue
            proposals.append({'id': f'lost-ball-{ball_id:04d}-hoop-{hoop_id:03d}',
                'status': 'candidate', 'evidence_origin': VERSION,
                'reason': 'descending_near_rim_track_lost', 'ball_id': ball_id,
                'hoop_id': hoop_id, 'hoop': hoop,
                'time_seconds': last['time_seconds']+delay,
                'timestamp_method': 'short_descending_continuation_estimate',
                'net_anchor_seconds': last['time_seconds'],
                'track_expiry_seconds': last['time_seconds']+ttl,
                'rim_plane_y': plane, 'rim_geometry': 'hoop_box_top_proxy',
                'predicted_x': projected_x, 'previous_observation': previous,
                'last_observation': last, 'history_observations': len(recent),
                'below_rim_observed': False, 'net_required': True})

    events = []
    for proposal in sorted(proposals, key=lambda p: (p['time_seconds'], p['ball_id'], p['hoop_id'])):
        duplicates = [e for e in [*existing_events, *events]
                      if e['hoop_id'] == proposal['hoop_id'] and
                      abs(e['time_seconds']-proposal['time_seconds']) <= window]
        if duplicates:
            rejected.append({'ball_id': proposal['ball_id'], 'hoop_id': proposal['hoop_id'],
                             'reason': 'same_hoop_event_in_net_window',
                             'duplicate_of': duplicates[0]['id']})
        else:
            events.append(proposal)
    return {'version': VERSION, 'events': events, 'rejections': rejected,
            'tracks_checked': len(tracks), 'manual_timestamps_used': False,
            'labels_used': False, 'net_motion_alone_creates_events': False}


def supported_events(events, records):
    """Every event needs its own net record; unsupported ones remain reviewable."""
    if [e['id'] for e in events] != [r['id'] for r in records]:
        raise ValueError('Net records must match each event exactly and in order')
    return [{**event, 'net_support': True, 'physical_goal_confirmation': 'pending'}
            for event, record in zip(events, records, strict=True) if record['net_support']]
