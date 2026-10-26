"""Post-analysis review routing, never a physical miss classifier."""
from collections import defaultdict
from math import isfinite


def validate_profile(profile):
    for key in ('exclude_premature_projection', 'exclude_observed_lateral_exit'):
        if type(profile[key]) is not bool:
            raise ValueError(f'{key} must be boolean')
    for key in ('minimum_observation_confidence', 'maximum_observation_gap_seconds', 'followup_seconds'):
        value = profile[key]
        if not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
            raise ValueError(f'Invalid {key}')
    if profile['minimum_observation_confidence'] > 1:
        raise ValueError('Confidence must not exceed one')


def exclusion(event, tracks, profile):
    last = event.get('last_observation')
    plane = event.get('rim_plane_y')
    if (profile['exclude_premature_projection'] and event.get('reason') == 'lost'
            and event.get('evidence_origin') == 'shot-context-v2'
            and not event.get('below_rim_observed') and last and plane is not None):
        bottom = last['y'] + last['h']/2
        if bottom < plane:
            return 'projection_before_rim_reach', {
                'last_observation': last, 'last_ball_bottom': bottom, 'rim_plane_y': plane,
                'interpretation': 'insufficient observed proximity; not a confirmed miss'}
    if (not profile['exclude_observed_lateral_exit']
            or event.get('reason') != 'crossing' or 'context_track_id' not in event):
        return None
    hoop = event['hoop']
    left, right = hoop['x']-hoop['w']/2, hoop['x']+hoop['w']/2
    bottom = hoop['y']+hoop['h']/2
    start = event['previous_observation']['time_seconds']
    end = event['time_seconds']+profile['followup_seconds']
    points = [p for p in tracks[event['context_track_id']] if start <= p['time_seconds'] <= end]
    for a, b in zip(points, points[1:]):
        gap = b['time_seconds']-a['time_seconds']
        if gap <= 0:
            raise ValueError('Non-chronological context track')
        if gap > profile['maximum_observation_gap_seconds']:
            return None
        if b['time_seconds'] > event['time_seconds'] and b['y'] <= plane:
            return None
        if not a['y'] <= bottom < b['y']:
            continue
        if min(a['conf'], b['conf']) < profile['minimum_observation_confidence']:
            return None
        side = None
        if all(p['x']+p['w']/2 < left for p in (a, b)):
            side = 'left'
        elif all(p['x']-p['w']/2 > right for p in (a, b)):
            side = 'right'
        if side:
            return 'observed_lateral_net_exit', {
                'observations': [a, b], 'side': side, 'net_bottom_y': bottom,
                'hoop_left': left, 'hoop_right': right,
                'interpretation': '2D outside-exit evidence; not a confirmed miss'}
        return None
    return None


def triage(supported, uncertain, net_records, associations, profile):
    validate_profile(profile)
    events = supported + uncertain
    ids = [e['id'] for e in events]
    records = {r['id']: r for r in net_records}
    if len(ids) != len(set(ids)) or len(records) != len(net_records) or set(ids) != set(records):
        raise ValueError('Candidate/net partition mismatch')
    if any(not records[e['id']]['net_support'] for e in supported):
        raise ValueError('Supported event lacks net support')
    if any(records[e['id']]['net_support'] for e in uncertain):
        raise ValueError('Uncertain event has net support')
    tracks = defaultdict(list)
    for point in associations:
        for key in ('time_seconds', 'x', 'y', 'w', 'h', 'conf'):
            if not isfinite(point[key]):
                raise ValueError('Non-finite observation')
        if point['w'] <= 0 or point['h'] <= 0:
            raise ValueError('Invalid ball dimensions')
        tracks[point['track_id']].append(point)
    remaining, excluded = [], []
    decisions = [{'id': e['id'], 'category': 'supported', 'reason': 'supported_passthrough'}
                 for e in supported]
    for event in uncertain:
        result = exclusion(event, tracks, profile)
        if result:
            reason, evidence = result
            excluded.append(event)
            decisions.append({'id': event['id'], 'category': 'diagnostic_exclusion',
                              'reason': reason, 'evidence': evidence})
        else:
            remaining.append(event)
            decisions.append({'id': event['id'], 'category': 'uncertain',
                              'reason': 'no_sufficient_exclusion_evidence'})
    return {'version': profile['version'], 'supported': supported, 'uncertain': remaining,
            'excluded': excluded, 'decisions': decisions, 'labels_used': False,
            'physical_miss_classification': False}
