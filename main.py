#!/usr/bin/env python3
"""Roblox Up-and-Coming top-10 forecaster (roblox-up-and-coming-forecast-01).

Reads the chart history at $MAKEFUN_INPUT_DIR (default /input) and writes one
forecast to $MAKEFUN_OUTPUT_DIR/forecast.json (default /output): for each of
the top 10 places, a probability for each game named being at that place, plus
the top 10 in order. That one forecast is scored against every hourly read of
the 72-hour window after entries close.

Model: per-game drift-diffusion. Each game's rank is treated as a Gaussian
random walk: its mean drifts with its recent rank velocity (shrunk toward zero,
ranks mean-revert), and its spread grows with horizon, calibrated from the
history itself (std of rank changes at several lags). The place probabilities
are the time-average of that Gaussian over the window, so early hours are
sharp on the current chart and later hours are diffuse. Probabilities per
place are normalised to 0.85 named mass, leaving 0.15 for games the model
does not name. Deterministic: no randomness, no network, stdlib only.
"""

import json
import math
import os
from datetime import datetime, timedelta

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

PLACES = 10
NAMED_MASS = 0.85
MAX_PER_PLACE = 30
MIN_W = 0.0015
HORIZONS = [0, 12, 24, 36, 48, 60, 72]
VEL_WINDOW_H = 48
VEL_SHRINK = 0.5
VEL_CAP = 0.7
# Drift shape, settled by a practice-window grid (history to 2026-11-07,
# scored on the 10 archive reads to 2026-11-10): v * t with shrink 0.5 scores
# -35.56, beating a small fixed drift (best -35.66), shrink 0.25 (-35.60) and
# shrink 0.75 (-35.58). The differences are small; the drift is not the
# binding term, the calibrated spread is.
CHART_SIZE = 97


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def load_history():
    with open(os.path.join(INPUT_DIR, "task.json")) as f:
        task = json.load(f)
    snaps = []
    with open(os.path.join(INPUT_DIR, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rows = {}
            for r in d.get("up_and_coming") or []:
                rows[r["universe_id"]] = (r["rank"], r.get("playing") or 0)
            snaps.append((parse_time(d["at"]), rows))
    snaps.sort(key=lambda s: s[0])
    return task, snaps


def rank_at(snaps, when, uid):
    """Interpolated rank of a game at a time; 97 if it is off the chart."""
    before = None
    after = None
    for t, rows in snaps:
        if uid in rows:
            if t <= when:
                before = (t, rows[uid][0])
            elif after is None:
                after = (t, rows[uid][0])
                break
    if before is None and after is None:
        return float(CHART_SIZE)
    if before is None:
        return float(after[1])
    if after is None:
        return float(before[1])
    (t0, r0), (t1, r1) = before, after
    if t1 == t0:
        return float(r0)
    f = (when - t0).total_seconds() / (t1 - t0).total_seconds()
    return r0 + (r1 - r0) * f


def calibrate_sigma(snaps):
    """Fit spread(t) = sqrt(c0 + c1*t) from rank changes at several lags."""
    pts = []
    for lag in (6, 12, 24, 48):
        diffs = []
        for i, (t0, rows0) in enumerate(snaps):
            for t1, rows1 in snaps[i + 1:]:
                dt = (t1 - t0).total_seconds() / 3600.0
                if dt < lag - 2 or dt > lag + 2:
                    continue
                for uid, (r0, _) in rows0.items():
                    if uid in rows1:
                        diffs.append(rows1[uid][0] - r0)
        if len(diffs) >= 20:
            mean = sum(diffs) / len(diffs)
            var = sum((d - mean) ** 2 for d in diffs) / len(diffs)
            pts.append((lag, var))
    if len(pts) >= 2:
        n = len(pts)
        sx = sum(p[0] for p in pts)
        sy = sum(p[1] for p in pts)
        sxx = sum(p[0] ** 2 for p in pts)
        sxy = sum(p[0] * p[1] for p in pts)
        den = n * sxx - sx * sx
        if den:
            c1 = (n * sxy - sx * sy) / den
            c0 = (sy - c1 * sx) / n
            c0 = min(max(c0, 1.0), 100.0)
            c1 = min(max(c1, 0.5), 25.0)
            return c0, c1
    return 9.0, 7.0


def main():
    task, snaps = load_history()
    places = int(task.get("places", PLACES))
    t_last, last_rows = snaps[-1]
    c0, c1 = calibrate_sigma(snaps)

    def sigma(t):
        return min(max(math.sqrt(c0 + c1 * t), 2.0), 30.0)

    # Candidate pool: games in the last 5 snapshots, or top 30 in the last 48h.
    pool = set(last_rows)
    cutoff = t_last - timedelta(hours=VEL_WINDOW_H)
    for t, rows in snaps:
        if t < cutoff:
            continue
        for uid, (r, _) in rows.items():
            if r <= 30:
                pool.add(uid)
    for t, rows in snaps[-5:]:
        pool |= set(rows)

    t_vel = t_last - timedelta(hours=VEL_WINDOW_H)
    games = []
    for uid in pool:
        r0 = last_rows[uid][0] if uid in last_rows else CHART_SIZE + 3
        r_then = rank_at(snaps, t_vel, uid)
        v = (r0 - r_then) / VEL_WINDOW_H
        v = max(min(v, VEL_CAP), -VEL_CAP) * VEL_SHRINK
        games.append((uid, float(r0), v))

    # Time-averaged place weights per game.
    weights = {uid: [0.0] * (places + 1) for uid, _, _ in games}
    top10_score = {}
    for uid, r0, v in games:
        exp_rank = 0.0
        for t in HORIZONS:
            mu = r0 + v * t
            mu = min(max(mu, 1.0), float(CHART_SIZE))
            sd = sigma(t)
            exp_rank += mu
            top10_score[uid] = top10_score.get(uid, 0.0) + norm_cdf(
                (places + 0.5 - mu) / sd
            )
            for k in range(1, places + 1):
                p = norm_cdf((k + 0.5 - mu) / sd) - norm_cdf((k - 0.5 - mu) / sd)
                weights[uid][k] += p
        exp_rank /= len(HORIZONS)
        top10_score[uid] /= len(HORIZONS)
        weights[uid][0] = exp_rank

    probabilities = []
    for k in range(1, places + 1):
        cand = sorted(
            ((weights[uid][k] / len(HORIZONS), uid) for uid, _, _ in games),
            reverse=True,
        )
        cand = [(w, uid) for w, uid in cand if w > MIN_W][:MAX_PER_PLACE]
        total = sum(w for w, _ in cand)
        if total <= 0:
            continue
        named = []
        for w, uid in cand:
            named.append((uid, NAMED_MASS * w / total))
        s = sum(p for _, p in named)
        if s > NAMED_MASS:
            uid0, p0 = named[0]
            named[0] = (uid0, p0 - (s - NAMED_MASS))
        for uid, p in named:
            probabilities.append(
                {"universe_id": int(uid), "place": k, "p": round(p, 6)}
            )

    ranked = sorted(
        games,
        key=lambda g: (-top10_score[g[0]], weights[g[0]][0]),
    )
    top10 = [int(uid) for uid, _, _ in ranked[:places]]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = {"probabilities": probabilities, "top10": top10}
    with open(os.path.join(OUTPUT_DIR, task.get("output", "forecast.json")), "w") as f:
        f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
