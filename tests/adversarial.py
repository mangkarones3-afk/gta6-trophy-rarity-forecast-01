#!/usr/bin/env python3
"""Adversarial synthetic family 2: logistic S-curve earn rates.

A different plausible world from tests/synthetic.py.  Here each trophy's
earn RATE itself follows a logistic S-curve in time,

    rate(t) = U / (1 + exp(-(t - m) / w))          (percent),

whose inflection point m sits at or beyond the end of the 55.2 h history
for most tiers - bronze story chapters turn at 25-50 h, the other bronzes
at 45-90 h, silvers at 70-120 h, golds at 120-190 h, and the platinum
(U only 0.05-0.3%) at 300-500 h - so the history mostly shows the slow
foot of each rise while the 108.5 h horizon covers the steep part.  The
tracked player count grows fast on a saturating exponential and then
keeps creeping through a slow logarithmic factor, reaching roughly
55-95x its launch value by the target.  The page refreshes every 5-9 h;
the house reads it every 30 minutes (111 reads in the history), each read
repeating the page as of the last refresh.  The truth is the page state
at the last refresh at or before the target.

Scoring, baselines and the runner are shared with tests/synthetic.py:
the exact competition pinball loss on the logit scale, the persistence
baseline and the house curve baseline re-implemented from the RULES.txt
prose.  Exit code 0 iff our program beats BOTH baselines on the average
over seeds.  A final dense-read spot check (reads every 10 minutes,
~330 reads in the history) times one end-to-end run of our program.
"""

import json
import math
import os
import random
import sys
import time
from datetime import timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from synthetic import (  # noqa: E402
    KIT_INPUT, LEVELS, START, baseline_curve, baseline_persistence, logit,
    pinball_mean, run_ours,
)

HISTORY_HOURS = 55.2
HORIZON_HOURS = 108.5
SEEDS = (7, 42, 99, 123, 555)
STORY_IDS = (3, 4, 5, 6, 7)


def _s_curve(t, u, m, w):
    z = -(t - m) / w
    if z > 60.0:
        return 0.0
    if z < -60.0:
        return u
    return u / (1.0 + math.exp(z))


def build_world(seed, read_step_h=0.5):
    """Returns (case_dict, truth {id: rate}, tier_of {id: tier})."""
    rng = random.Random(seed)
    with open(KIT_INPUT, "r", encoding="utf-8") as fh:
        p00 = json.load(fh)
    trophies = p00["trophies"]
    launch_rates = p00["history"][0]["rates"]
    n0 = float(p00["history"][0]["players_tracked"])

    # Per-trophy S-curve parameters (U in percent, m and w in hours).
    curves = []
    for tr, r0 in zip(trophies, launch_rates):
        tier = tr["tier"]
        if tier == "platinum":
            u = rng.uniform(0.05, 0.30)
            m = rng.uniform(300.0, 500.0)
            w = rng.uniform(60.0, 100.0)
        elif tier == "gold":
            u = rng.uniform(2.0, 9.0)
            m = rng.uniform(120.0, 190.0)
            w = rng.uniform(24.0, 40.0)
        elif tier == "silver":
            u = rng.uniform(5.0, 18.0)
            m = rng.uniform(70.0, 120.0)
            w = rng.uniform(18.0, 30.0)
        elif tr["id"] in STORY_IDS:
            u = rng.uniform(30.0, 60.0)
            m = rng.uniform(25.0, 50.0)
            w = rng.uniform(10.0, 18.0)
        elif r0 is not None and r0 >= 99.0:
            u = rng.uniform(80.0, 95.0)
            m = rng.uniform(25.0, 45.0)
            w = rng.uniform(10.0, 16.0)
        else:  # remaining bronzes
            u = rng.uniform(8.0, 30.0)
            m = rng.uniform(45.0, 90.0)
            w = rng.uniform(14.0, 24.0)
        curves.append((u, m, w))

    # Players: fast saturating growth times a slow logarithmic creep.
    k_sat = rng.uniform(32.0, 47.0)
    tau_n = rng.uniform(16.0, 28.0)
    beta = rng.uniform(0.22, 0.28)

    def players(t):
        sat = 1.0 + (k_sat - 1.0) * (1.0 - math.exp(-t / tau_n))
        creep = 1.0 + beta * math.log1p(t / 5.0)
        return n0 * sat * creep

    def page_at(t):
        rates = [round(_s_curve(t, u, m, w), 2) for (u, m, w) in curves]
        return int(round(players(t))), rates

    # Refresh times: every 5-9 hours from t=0 through past the target.
    refresh_times = [0.0]
    t = 0.0
    end_all = HISTORY_HOURS + HORIZON_HOURS + 10.0
    while t < end_all:
        t += rng.uniform(5.0, 9.0)
        refresh_times.append(t)

    def page_shown_at(t):
        rt = max(x for x in refresh_times if x <= t)
        return page_at(rt)

    history = []
    read_t = 0.0
    while read_t <= HISTORY_HOURS + 1e-9:
        n, rates = page_shown_at(read_t)
        stamp = START + timedelta(hours=read_t)
        history.append({
            "slot": stamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "at": stamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "players_tracked": n,
            "trophies_earned": 0,
            "rates": rates,
        })
        read_t += read_step_h

    at_time = START + timedelta(hours=HISTORY_HOURS)
    target_time = at_time + timedelta(hours=HORIZON_HOURS)
    case = {
        "schema": "makefun/gta6-trophies-case/v1",
        "id": "adversarial",
        "at": at_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "horizon_minutes": int(HORIZON_HOURS * 60),
        "target": target_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "levels": list(LEVELS),
        "game": p00["game"],
        "trophies": trophies,
        "history": history,
        "notes": "adversarial synthetic family 2 (logistic S-curve rates)",
    }
    _n, truth_rates = page_shown_at(HISTORY_HOURS + HORIZON_HOURS)
    truth = {tr["id"]: r for tr, r in zip(trophies, truth_rates)}
    tier_of = {tr["id"]: tr["tier"] for tr in trophies}
    return case, truth, tier_of


def _tier_loss(forecast, truth, tier_of, tier, name, totals, counts):
    for tid, y in truth.items():
        if tier_of[tid] != tier:
            continue
        s = 0.0
        for tau, q in zip(LEVELS, forecast[tid]):
            d = logit(y) - logit(q)
            s += 2.0 * max(tau * d, (tau - 1.0) * d)
        key = (tier, name)
        totals[key] = totals.get(key, 0.0) + s / len(LEVELS)
        counts[key] = counts.get(key, 0) + 1


def main():
    names = ("ours", "persistence", "curve")
    totals = {k: 0.0 for k in names}
    tier_totals = {}
    tier_counts = {}
    print(f"{'seed':>5} {'ours':>9} {'persist':>9} {'curve':>9}")
    for seed in SEEDS:
        case, truth, tier_of = build_world(seed)
        forecasts = {"ours": run_ours(case),
                     "persistence": baseline_persistence(case),
                     "curve": baseline_curve(case)}
        losses = {k: pinball_mean(forecasts[k], truth) for k in names}
        for k in names:
            totals[k] += losses[k]
            for tier in ("platinum", "gold", "silver", "bronze"):
                _tier_loss(forecasts[k], truth, tier_of, tier, k,
                           tier_totals, tier_counts)
        print(f"{seed:>5} {losses['ours']:>9.4f} "
              f"{losses['persistence']:>9.4f} {losses['curve']:>9.4f}")
    n = len(SEEDS)
    avg = {k: v / n for k, v in totals.items()}
    print(f"{'AVG':>5} {avg['ours']:>9.4f} {avg['persistence']:>9.4f} "
          f"{avg['curve']:>9.4f}")
    print("\nper-tier mean loss (ours vs persistence vs curve):")
    for tier in ("platinum", "gold", "silver", "bronze"):
        cells = []
        for k in names:
            v = tier_totals.get((tier, k), 0.0) / max(
                1, tier_counts.get((tier, k), 1))
            cells.append(f"{k} {v:.4f}")
        print(f"  {tier:<9} " + "   ".join(cells))

    # Dense-read spot check: reads every 10 minutes (~330 in history).
    dense_case, _truth, _tiers = build_world(SEEDS[0], read_step_h=1.0 / 6.0)
    t0 = time.perf_counter()
    run_ours(dense_case)
    dense_ms = (time.perf_counter() - t0) * 1000.0
    print(f"\ndense-read spot check: {len(dense_case['history'])} reads, "
          f"one run {dense_ms:.0f} ms")

    ok = avg["ours"] < avg["persistence"] and avg["ours"] < avg["curve"]
    print("\nVERDICT:", "PASS - ours beats both baselines on average"
          if ok else "FAIL - ours does not beat both baselines")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
