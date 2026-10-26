"""Explicit, checked transformations of one pinned upstream source version."""


def adapt_source(source):
    replacements = {
        '        self.model = YOLO(model, verbose=self.verbose)':
            '        self.verbose = verbose\n        self.model = _route(self)',
        '            m = (y2 - y1) / (x2 - x1)\n            b = y1 - (m * x1)\n\n            # CALCULATE X-COORDINATE OF BALL WHEN IT WAS AT HOOP HEIGHT\n            x_pred = (y_hoop - b) / m':
            '            x_pred = x2 + (y_hoop-y2)*(x1-x2)/(y1-y2)',
        '            self.attempts += 1\n            self.down_ball.remove([ballKey, hoopKey])':
            '            self._audit_decision(ballKey, hoopKey, hoop, b, prevDetection, x_pred, x1_rim < x_pred < x2_rim)\n            self.attempts += 1\n            self.down_ball.remove([ballKey, hoopKey])',
        '            if self.frame_count - ball.get_last_detection().frame > 20:':
            '            if self.frame_count - ball.get_last_detection().frame > _ttl_frames:',
        '            if len(ball.detections) > 30:\n                ball.detections.popleft()':
            '            while len(ball.detections) > 1 and self.frame_count-ball.detections[0].frame > _history_frames:\n                ball.detections.popleft()',
        '            if self.frame_count - hoop.frame > 20:':
            '            if self.frame_count - hoop.frame > _ttl_frames:',
    }
    for original, replacement in replacements.items():
        if source.count(original) != 1:
            raise ValueError('Pinned upstream adapter source mismatch')
        source = source.replace(original, replacement)
    return source
