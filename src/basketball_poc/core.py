from dataclasses import asdict, dataclass
import math


def number(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return float(value)


def rectangle(value, name, width, height):
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{name} must be [x1, y1, x2, y2]")
    for item in value:
        if type(item) is not int:
            raise ValueError(f"{name} coordinates must be integers")
    x1, y1, x2, y2 = value
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"{name} is outside the {width}x{height} frame or empty")
    return value


DEFAULTS = {
    "confidence": 0.20, "imgsz": 960, "max_gap_seconds": 0.15,
    "max_speed_rim_widths_per_second": 30.0,
    "confirm_seconds": 0.40, "cooldown_seconds": 0.75,
}


def calibration(raw, width, height):
    unknown = set(raw) - set(DEFAULTS) - {"roi", "rim", "ball_class"}
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    config = {**DEFAULTS, **raw}
    roi = rectangle(config.get("roi"), "roi", width, height)
    rim = rectangle(config.get("rim"), "rim", width, height)
    if not (roi[0] <= rim[0] < rim[2] <= roi[2] and roi[1] <= rim[1] < rim[3] <= roi[3]):
        raise ValueError("rim must be entirely inside roi")
    for key in DEFAULTS:
        number(config[key], key)
    if not 0 < config["confidence"] <= 1:
        raise ValueError("confidence must be > 0 and <= 1")
    if type(config["imgsz"]) is not int or config["imgsz"] < 32 or config["imgsz"] % 32:
        raise ValueError("imgsz must be an integer multiple of 32")
    for key in ("max_gap_seconds", "max_speed_rim_widths_per_second", "confirm_seconds"):
        if config[key] <= 0:
            raise ValueError(f"{key} must be > 0")
    if type(config.get("ball_class")) is not int or config["ball_class"] < 0:
        raise ValueError("ball_class must be an explicit non-negative model class id")
    return config


@dataclass
class Point:
    time: float
    x: float
    y: float
    confidence: float


class GoalTracker:
    """Conservative single-ball descending crossing baseline in source pixels."""

    def __init__(self, config):
        self.config = config
        self.previous = None
        self.pending = None
        self.last_event = -math.inf
        self.last_time = -math.inf

    def reset(self):
        self.previous = None
        self.pending = None

    def step(self, time, balls):
        number(time, "detection timestamp")
        if time <= self.last_time:
            raise ValueError("Detection timestamps must increase strictly")
        self.last_time = time
        # Multiple balls cannot safely be connected by a single-ball tracker.
        if len(balls) != 1:
            if len(balls) > 1 or (self.previous and time - self.previous.time > self.config["max_gap_seconds"]):
                self.reset()
            return None
        point = balls[0]
        x1, y1, x2, y2 = self.config["rim"]
        width = x2 - x1
        previous = self.previous
        self.previous = point
        if previous is None:
            return None
        gap = time - previous.time
        distance = math.hypot(point.x - previous.x, point.y - previous.y)
        if gap > self.config["max_gap_seconds"] or distance > width * self.config["max_speed_rim_widths_per_second"] * gap:
            self.pending = None
            return None
        if time - self.last_event < self.config["cooldown_seconds"]:
            self.pending = None
            return None
        if self.pending:
            if time - self.pending["time_seconds"] > self.config["confirm_seconds"] or point.y < previous.y or not x1 <= point.x <= x2:
                self.pending = None
            elif point.y > y2:
                event = {**self.pending, "confirmed_at_seconds": time,
                         "status": "candidate", "confirmation": asdict(point)}
                self.last_event = time
                self.reset()
                return event
        # A later, distinct below-rim observation is required for confirmation.
        if previous.y < y1 <= point.y and point.y > previous.y:
            ratio = (y1 - previous.y) / (point.y - previous.y)
            crossing_x = previous.x + ratio * (point.x - previous.x)
            if x1 <= crossing_x <= x2:
                self.pending = {"time_seconds": previous.time + ratio * gap,
                                "crossing_x": crossing_x,
                                "evidence": [asdict(previous), asdict(point)]}
        return None


def timestamps(events, name):
    if not isinstance(events, list):
        raise ValueError(f"{name} must be a list")
    values = sorted(number(e["time_seconds"], name) for e in events)
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate timestamps")
    return values


def evaluate(predictions, annotations, tolerance):
    tolerance = number(tolerance, "tolerance")
    if annotations.get("complete") is not True:
        raise ValueError("annotations must declare complete: true after full manual review")
    start = number(annotations["interval"]["start_seconds"], "interval start")
    end = number(annotations["interval"]["end_seconds"], "interval end")
    if end <= start:
        raise ValueError("annotation end must exceed start")
    interval = predictions["interval"]
    prediction_start = number(interval["start_seconds"], "prediction start")
    prediction_end = number(interval["end_seconds"], "prediction end")
    if prediction_start > start or prediction_end < end:
        raise ValueError("predictions do not cover the entire annotated interval")
    if not predictions.get("source_sha256") or predictions["source_sha256"] != annotations.get("source_sha256"):
        raise ValueError("annotations and predictions must refer to the same source_sha256")
    truth = timestamps(annotations["goals"], "goals")
    if any(t < start or t >= end for t in truth):
        raise ValueError("goal lies outside annotated [start, end) interval")
    predicted = [t for t in timestamps(predictions["events"], "events") if start <= t < end]
    # Ordered 1D matching: maximize cardinality, then minimize absolute error.
    n, m = len(predicted), len(truth)
    scores = [[(0, 0.0)] * (m + 1) for _ in range(n + 1)]
    actions = {}
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            candidates = [(scores[i-1][j], "prediction"), (scores[i][j-1], "truth")]
            error = abs(predicted[i-1] - truth[j-1])
            if error <= tolerance:
                count, negative_error = scores[i-1][j-1]
                candidates.append(((count + 1, negative_error - error), "match"))
            scores[i][j], actions[i, j] = max(candidates, key=lambda c: c[0])
    i, j = n, m
    matches, used_p, used_t = [], set(), set()
    while i and j:
        action = actions[i, j]
        if action == "match":
            i -= 1
            j -= 1
            used_p.add(i)
            used_t.add(j)
            matches.append({"prediction_seconds": predicted[i], "truth_seconds": truth[j],
                            "error_seconds": predicted[i] - truth[j]})
        elif action == "prediction":
            i -= 1
        else:
            j -= 1
    tp = len(matches)
    precision = tp / n if n else None
    recall = tp / m if m else None
    return {"interval": {"start_seconds": start, "end_seconds": end}, "tolerance_seconds": tolerance,
            "true_positives": tp, "false_positives": n - tp, "false_negatives": m - tp,
            "precision": precision, "recall": recall,
            "f1": 2 * tp / (n + m) if n and m else None,
            "mean_absolute_error_seconds": sum(abs(x["error_seconds"]) for x in matches) / tp if tp else None,
            "matches": list(reversed(matches)),
            "unmatched_predictions": [t for k, t in enumerate(predicted) if k not in used_p],
            "unmatched_goals": [t for k, t in enumerate(truth) if k not in used_t]}
