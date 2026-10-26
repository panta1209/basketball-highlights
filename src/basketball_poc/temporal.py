"""Net-independent temporal evidence, layered over the pinned upstream tracker.

The top of a hoop detector box is only a rim-plane proxy, not a 3D rim.
Neither missing net motion nor speed change is used as negative evidence.
"""
from math import hypot


def crossing_evidence(observations, hoop, fps, max_gap_seconds):
    """Use adjacent, chronological observations; never bridge a whole trajectory.

    Frame numbers follow the upstream one-based convention. Confirmation needs
    the complete detected ball below the plane, independent of hoop/net height.
    """
    if fps <= 0 or max_gap_seconds <= 0:
        raise ValueError('Positive fps and continuity time required')
    plane = hoop.y - hoop.h / 2
    evidence = {'decision': 'uncertain', 'reason': 'no_descending_bracket',
                'rim_plane_y': plane, 'rim_geometry': 'hoop_box_top_proxy',
                'net_evidence': 'unused', 'net_required': False}
    if any(b.frame <= a.frame for a, b in zip(observations, observations[1:])):
        return {**evidence, 'reason': 'non_chronological_track'}
    for index in range(len(observations)-2, -1, -1):
        above, below = observations[index:index+2]
        if not above.y <= plane < below.y:
            continue
        tail = observations[index:]
        if any((b.frame-a.frame)/fps > max_gap_seconds + 1e-12
               for a, b in zip(tail, tail[1:])):
            return {**evidence, 'reason': 'excessive_observation_gap'}
        if any(p.y <= plane for p in observations[index+2:]):
            return {**evidence, 'reason': 'returned_above_plane'}
        confirmation = observations[-1]
        if confirmation.y - confirmation.h/2 <= plane:
            return {**evidence, 'reason': 'ball_not_fully_below_plane'}
        ratio = (plane-above.y)/(below.y-above.y)
        x = above.x + ratio*(below.x-above.x)
        made = hoop.x-hoop.w/2 < x < hoop.x+hoop.w/2
        return {**evidence, 'decision': 'candidate' if made else 'outside_projection',
                'reason': 'descending_projected_crossing' if made else 'crossing_outside_hoop_width',
                'time_seconds': (above.frame-1+ratio*(below.frame-above.frame))/fps,
                'predicted_x': x, 'above': above, 'below': below,
                'below_confirmation': confirmation,
                'horizontal_clearance_ball_radii':
                    min(x-(hoop.x-hoop.w/2), hoop.x+hoop.w/2-x) /
                    ((above.w+ratio*(below.w-above.w))/2)}
    return evidence


class TemporalEventsMixin:
    """Keep native gates; prevent multiple observations per track/source frame."""

    def add_ball(self, ball):
        if ball.conf < .4 and not (self.hoop_area(ball) and ball.conf > .3):
            return None
        options = []
        for key, track in self.balls.items():
            previous = track.get_last_detection()
            if previous.frame >= ball.frame:
                continue
            distance = hypot(previous.x-ball.x, previous.y-ball.y)
            limit = 4 if key in [p[0] for p in self.up_ball] else 2
            if distance < hypot(previous.w, previous.h)*limit:
                options.append((distance, key))
        if options:
            _, key = min(options)
            self.balls[key].add_detection(ball)
        else:
            key = self.ballUid
            self.balls[key] = self._ball_type(ball)
            self.ballUid += 1
        self._audit_association(key, ball)
        return key

    def detect_up(self):
        if self.event_mode == 'association-v2':
            return super().detect_up()
        existing = {tuple(pair) for pair in self.up_ball}
        super().detect_up()
        # The native region extends down to the middle of a net-containing box.
        # Only new arms must be above our plane; preserve previously armed balls
        # while they descend so detect_down can confirm the whole ball below it.
        for pair in list(self.up_ball):
            if tuple(pair) not in existing:
                ball_id, hoop_id = pair
                hoop = self.hoops[hoop_id]
                if self.balls[ball_id].get_last_detection().y >= hoop.y-hoop.h/2:
                    self.up_ball.remove(pair)

    def detect_down(self):
        if self.event_mode == 'association-v2':
            return super().detect_down()
        for pair in list(self.up_ball):
            ball_id, hoop_id = pair
            if ball_id not in self.balls or hoop_id not in self.hoops:
                self.up_ball.remove(pair)
                continue
            hoop = self.hoops[hoop_id]
            last = self.balls[ball_id].get_last_detection()
            if last.y-last.h/2 > hoop.y-hoop.h/2:
                self.down_ball.append(pair)
                self.up_ball.remove(pair)

    def update_score(self):
        if self.event_mode == 'association-v2':
            return super().update_score()
        for pair in list(self.down_ball):
            ball_id, hoop_id = pair
            self.down_ball.remove(pair)
            if ball_id not in self.balls or hoop_id not in self.hoops:
                continue
            hoop = self.hoops[hoop_id]
            points = list(self.balls[ball_id].detections)
            evidence = crossing_evidence(points, hoop, self.source_fps, self.track_ttl_seconds)
            self._audit_temporal(ball_id, hoop_id, hoop, evidence)
            if evidence['decision'] != 'uncertain':
                self.attempts += 1
                self.makes += int(evidence['decision'] == 'candidate')
