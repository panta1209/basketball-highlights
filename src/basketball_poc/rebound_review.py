"""Conservative review routing from a fragmented, observed rebound arc.

This does not change detection or prove physical ball identity / a physical miss.
Only existing uncertain candidates can be excluded; labels are not an input.
"""
from collections import defaultdict
from math import hypot, isfinite


def validate_profile(profile):
    keys = ('minimum_confidence', 'maximum_link_gap_seconds', 'maximum_link_hoop_widths',
            'maximum_size_ratio', 'maximum_observation_gap_seconds', 'followup_seconds',
            'minimum_arc_seconds', 'minimum_strong_observations', 'minimum_rise_ball_heights',
            'minimum_observed_rise_ball_heights', 'minimum_fall_ball_heights',
            'outside_clearance_hoop_widths', 'jitter_ball_dimensions')
    for key in keys:
        value = profile[key]
        if type(value) not in (int, float) or not isfinite(value) or value <= 0:
            raise ValueError(f'Invalid {key}')
    if (profile['minimum_confidence'] > 1 or profile['maximum_size_ratio'] < 1
            or type(profile['minimum_strong_observations']) is not int
            or profile['minimum_strong_observations'] < 3
            or profile['minimum_arc_seconds'] >= profile['followup_seconds']):
        raise ValueError('Invalid rebound profile bounds')


def track_index(associations):
    tracks = defaultdict(list)
    for point in associations:
        for key in ('time_seconds', 'x', 'y', 'w', 'h', 'conf'):
            if not isfinite(point[key]):
                raise ValueError('Non-finite observation')
        if point['w'] <= 0 or point['h'] <= 0 or not 0 <= point['conf'] <= 1:
            raise ValueError('Invalid observation')
        track = tracks[point['track_id']]
        if track and point['time_seconds'] <= track[-1]['time_seconds']:
            raise ValueError('Non-chronological context track')
        track.append(point)
    return tracks


def size_compatible(a, b, limit):
    return all(max(a[k], b[k])/min(a[k], b[k]) <= limit for k in ('w', 'h'))


def rebound_evidence(event, tracks, profile):
    if (event.get('reason') != 'lost' or event.get('evidence_origin') != 'shot-context-v2'
            or event.get('below_rim_observed') or 'context_track_id' not in event):
        return None
    last, previous = event.get('last_observation'), event.get('previous_observation')
    plane, hoop = event.get('rim_plane_y'), event.get('hoop')
    if not last or not previous or plane is None or not hoop:
        return None
    confidence = profile['minimum_confidence']
    left, right = hoop['x']-hoop['w']/2, hoop['x']+hoop['w']/2
    if (min(previous['conf'], last['conf']) < confidence
            or not 0 < last['time_seconds']-previous['time_seconds'] <= profile['maximum_link_gap_seconds']
            or last['y']-previous['y'] < last['h']/2
            or not last['y']-last['h']/2 <= plane <= last['y']+last['h']/2
            or not left <= last['x'] <= right):
        return None
    start = last['time_seconds']
    end = start+profile['followup_seconds']
    old_id = event['context_track_id']
    # The event must describe an actual observed endpoint, not a stale projection.
    old = tracks.get(old_id, [])
    if not any(all(p[k] == last[k] for k in ('time_seconds', 'x', 'y', 'w', 'h')) for p in old):
        return None
    if any(start < p['time_seconds'] <= end for p in old):
        return None
    radius = profile['maximum_link_hoop_widths']*hoop['w']
    nearby = {}
    for identifier, points in tracks.items():
        if identifier == old_id:
            continue
        close = [p for p in points if 0 < p['time_seconds']-start <= profile['maximum_link_gap_seconds']
                 and hypot(p['x']-last['x'], p['y']-last['y']) <= radius]
        if close:
            nearby[identifier] = points
    # Count alternatives before confidence/size/upward checks. Weak alternatives
    # must not be silently discarded to manufacture a unique association.
    if len(nearby) != 1:
        return None
    identifier, track = next(iter(nearby.items()))
    first = track[0]
    if (not 0 < first['time_seconds']-start <= profile['maximum_link_gap_seconds']
            or hypot(first['x']-last['x'], first['y']-last['y']) > radius
            or first['conf'] < confidence
            or not size_compatible(first, last, profile['maximum_size_ratio'])):
        return None
    observed = [p for p in track if p['time_seconds'] <= end]
    strong = [p for p in observed if p['conf'] >= confidence]
    if (len(strong) < profile['minimum_strong_observations']
            or strong[-1]['time_seconds']-first['time_seconds'] < profile['minimum_arc_seconds']
            or observed[-1] != strong[-1]
            or any(b['time_seconds']-a['time_seconds'] > profile['maximum_observation_gap_seconds']
                   for a, b in zip(strong, strong[1:]))
            or any(not size_compatible(p, last, profile['maximum_size_ratio']) for p in observed)):
        return None
    apex_index = min(range(len(strong)), key=lambda i: strong[i]['y'])
    if apex_index == 0 or apex_index == len(strong)-1:
        return None
    apex, final = strong[apex_index], strong[-1]
    if (first['y']+first['h']/2 >= plane
            or last['y']-first['y'] < profile['minimum_rise_ball_heights']*last['h']
            or first['y']-apex['y'] < profile['minimum_observed_rise_ball_heights']*first['h']
            or final['y']-apex['y'] < profile['minimum_fall_ball_heights']*final['h']):
        return None
    side = -1 if first['x'] < hoop['x'] else 1
    clearance = (left-(final['x']+final['w']/2) if side == -1
                 else final['x']-final['w']/2-right)
    if clearance < profile['outside_clearance_hoop_widths']*hoop['w']:
        return None
    # All detections, including weak ones, can veto a return or a second turn.
    for a, b in zip(observed, observed[1:]):
        tolerance = profile['jitter_ball_dimensions']*min(a['w'], b['w'])
        if side*(b['x']-a['x']) < -tolerance:
            return None
        if a['time_seconds'] >= apex['time_seconds']:
            if b['y']-a['y'] < -profile['jitter_ball_dimensions']*min(a['h'], b['h']):
                return None
        elif b['y']-a['y'] > profile['jitter_ball_dimensions']*min(a['h'], b['h']):
            return None
    # A separate fragment around the rim later in this same window is ambiguous.
    for other_id, points in tracks.items():
        if other_id in (old_id, identifier):
            continue
        if any(first['time_seconds'] <= p['time_seconds'] <= end
               and left-p['w']/2 <= p['x'] <= right+p['w']/2
               and abs(p['y']-plane) <= hoop['h']+p['h'] for p in points):
            return None
    return {'previous_observation': previous, 'last_observation': last,
            'successor_track_id': identifier, 'link_gap_seconds': first['time_seconds']-start,
            'observations': observed, 'apex': apex, 'departure': final,
            'outside_clearance_hoop_widths': clearance/hoop['w'],
            'followup_end_seconds': end, 'observed_end_seconds': final['time_seconds'],
            'interpretation': 'unique plausible 2D rebound departure; not proven identity or physical miss'}


def filter_review(base, associations, profile):
    validate_profile(profile)
    tracks = track_index(associations)
    ids = [e['id'] for key in ('supported', 'uncertain', 'excluded') for e in base[key]]
    decisions = {d['id']: d for d in base['decisions']}
    if len(ids) != len(set(ids)) or len(decisions) != len(base['decisions']) or set(ids) != set(decisions):
        raise ValueError('Invalid triage partition')
    remaining, added, updated = [], [], []
    for event in base['uncertain']:
        evidence = rebound_evidence(event, tracks, profile)
        if evidence:
            added.append(event)
            updated.append({'id': event['id'], 'category': 'diagnostic_exclusion',
                            'reason': 'observed_fragmented_rebound_departure', 'evidence': evidence})
        else:
            remaining.append(event)
    replacements = {d['id']: d for d in updated}
    return {'supported': base['supported'], 'uncertain': remaining,
            'excluded': base['excluded']+added,
            'decisions': [replacements.get(d['id'], d) for d in base['decisions']],
            'new_exclusions': updated, 'version': profile['version'], 'labels_used': False}
