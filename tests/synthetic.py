#!/usr/bin/env python3
"""Deterministic synthetic final-like evaluation.

Builds fake cases that mimic the final's shape: launch-night rates taken
from the real p00 case (638 players tracked), then a page that refreshes
every few hours over 2.3 days of history while the tracked player count
grows into the tens of thousands and earners grow on saturating per-tier
progression curves (bronze story fast, silver/gold slower, platinum
near-flat).  House reads repeat the page between refreshes, exactly like
the real recorder.  The target sits 4.52 days after the history ends and
its answer is the page state at the last refresh at or before the target,
computed from the same curves.

Our program (../main.py, run as a subprocess on the generated case) is
scored against two baselines re-implemented here from the RULES.txt prose:
  * persistence: last rate, logit spread 0.5*sqrt(horizon days);
  * curve: per-trophy OLS of logit rate against ln(1 + hours since
    2026-11-19T00:00:00Z) over the case's last 24 h of reads, read at the
    target, spread = random-walk spread and fit residual combined in
    quadrature; fewer than 3 reads falls back to persistence.
Scoring is the exact competition loss: mean pinball loss on the logit
scale, rates clamped to 0.01..99.99, loss = 2*max(tau*d, (tau-1)*d) with
d = logit(y) - logit(q), over all 55 trophies x 5 levels.

Fixed seeds only; no wall clock.  Exit code 0 iff our program beats BOTH
baselines on the average over seeds.
"""

import json
import math
import os
import random
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
KIT_INPUT = os.path.expanduser(
    "~/workspace/makefun/gta6-trophy-rarity-forecast-01-kit/"
    "gta6-trophy-rarity-forecast-01-kit/input/p00.in")
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.645, -0.674, 0.0, 0.674, 1.645)
EPOCH_2026_11_19 = datetime(2026, 11, 19, tzinfo=timezone.utc)
START = datetime(2026, 11, 18, 23, 50, tzinfo=timezone.utc)
HISTORY_HOURS = 2.3 * 24.0
HORIZON_HOURS = 4.52 * 24.0
SEEDS = (11, 23, 37, 101)


def clamp_rate(r):
    return min(99.99, max(0.01, r))


def logit(r):
    p = clamp_rate(r) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(z):
    if z >= 0:
        return 100.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return 100.0 * e / (1.0 + e)


def pinball_mean(forecast, truth):
    """forecast: {id: [5 quantiles]}, truth: {id: rate}. Exact loss."""
    total = 0.0
    n = 0
    for tid, y in truth.items():
        ly = logit(y)
        for tau, q in zip(LEVELS, forecast[tid]):
            d = ly - logit(q)
            total += 2.0 * max(tau * d, (tau - 1.0) * d)
            n += 1
    return total / n


# ---------------------------------------------------------------------------
# world simulation
# ---------------------------------------------------------------------------

def build_world(seed):
    """Returns (case_dict, truth {id: rate}, tier_of {id: tier})."""
    rng = random.Random(seed)
    with open(KIT_INPUT, "r", encoding="utf-8") as fh:
        p00 = json.load(fh)
    trophies = p00["trophies"]
    launch_rates = p00["history"][0]["rates"]
    n0 = p00["history"][0]["players_tracked"]

    n_max = rng.uniform(42000.0, 88000.0)
    tau_n = rng.uniform(24.0, 40.0)

    def players(t):
        return n0 + (n_max - n0) * (1.0 - math.exp(-t / tau_n))

    # Per-trophy earner curves: e(t) = e0 + (A - e0) * (1 - exp(-t/tau)).
    curves = []
    for tr, r0 in zip(trophies, launch_rates):
        tier = tr["tier"]
        e0 = r0 * n0 / 100.0
        if tier == "platinum":
            a = e0 * rng.uniform(1.2, 2.0)          # near-flat earners
            tau = rng.uniform(120.0, 200.0)
        elif tier == "gold":
            u = rng.uniform(0.02, 0.08)             # ultimate rate share
            a = max(e0 * 1.5, u * n_max)
            tau = rng.uniform(70.0, 110.0)
        elif tier == "silver":
            u = rng.uniform(0.05, 0.18)
            a = max(e0 * 1.5, u * n_max)
            tau = rng.uniform(45.0, 75.0)
        else:  # bronze: story chapters fast, the rest medium
            if tr["id"] in (3, 4, 5, 6, 7):
                u = rng.uniform(0.30, 0.55)
                tau = rng.uniform(14.0, 26.0)
            elif r0 >= 99.0:
                u = rng.uniform(0.80, 0.92)         # the 100% launch trophy
                tau = rng.uniform(24.0, 36.0)
            else:
                u = rng.uniform(0.08, 0.30)
                tau = rng.uniform(26.0, 48.0)
            a = max(e0 * 1.5, u * n_max)
        curves.append((e0, a, tau))

    def page_at(t):
        n = players(t)
        rates = []
        for (e0, a, tau) in curves:
            e = e0 + (a - e0) * (1.0 - math.exp(-t / tau))
            if e > n:
                e = n
            rates.append(round(100.0 * e / n, 2))
        return int(round(n)), rates

    # Refresh times: every 4-7 hours from t=0 through the target.
    refresh_times = [0.0]
    t = 0.0
    end_all = HISTORY_HOURS + HORIZON_HOURS + 8.0
    while t < end_all:
        t += rng.uniform(4.0, 7.0)
        refresh_times.append(t)

    def page_shown_at(t):
        rt = max(x for x in refresh_times if x <= t)
        return page_at(rt)

    # House reads every 2 hours up to the history end; each read shows the
    # page as of the last refresh (so reads repeat between refreshes).
    history = []
    read_t = 0.0
    while read_t <= HISTORY_HOURS:
        n, rates = page_shown_at(read_t)
        stamp = START + timedelta(hours=read_t)
        history.append({
            "slot": stamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "at": stamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "players_tracked": n,
            "trophies_earned": 0,
            "rates": rates,
        })
        read_t += 2.0

    at_time = START + timedelta(hours=HISTORY_HOURS)
    target_time = at_time + timedelta(hours=HORIZON_HOURS)
    case = {
        "schema": "makefun/gta6-trophies-case/v1",
        "id": "synthetic",
        "at": at_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "horizon_minutes": int(HORIZON_HOURS * 60),
        "target": target_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "levels": list(LEVELS),
        "game": p00["game"],
        "trophies": trophies,
        "history": history,
        "notes": "synthetic final-like case",
    }
    _n, truth_rates = page_shown_at(HISTORY_HOURS + HORIZON_HOURS)
    truth = {tr["id"]: r for tr, r in zip(trophies, truth_rates)}
    tier_of = {tr["id"]: tr["tier"] for tr in trophies}
    return case, truth, tier_of


# ---------------------------------------------------------------------------
# baselines, re-implemented from the RULES.txt prose
# ---------------------------------------------------------------------------

def baseline_persistence(case):
    last = case["history"][-1]["rates"]
    d_days = case["horizon_minutes"] / 1440.0
    sigma = 0.5 * math.sqrt(d_days)
    out = {}
    for tr, r in zip(case["trophies"], last):
        r = clamp_rate(r if r is not None else 0.01)
        out[tr["id"]] = [round(inv_logit(logit(r) + z * sigma), 4) for z in Z]
    return out


def _ols(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return my, 0.0, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / n)
    return a, b, resid


def baseline_curve(case):
    fallback = baseline_persistence(case)
    d_days = case["horizon_minutes"] / 1440.0
    rw = 0.5 * math.sqrt(d_days)
    target_dt = datetime.strptime(case["target"], "%Y-%m-%dT%H:%M:%SZ")
    target_dt = target_dt.replace(tzinfo=timezone.utc)
    x_target = math.log1p((target_dt - EPOCH_2026_11_19).total_seconds() / 3600.0)
    last24 = []
    for row in case["history"]:
        dt = datetime.strptime(row["at"], "%Y-%m-%dT%H:%M:%SZ")
        dt = dt.replace(tzinfo=timezone.utc)
        age_h = (target_dt - dt).total_seconds() / 3600.0
        # "last 24 h of reads": reads within 24 h before the checkpoint.
        at_dt = datetime.strptime(case["at"], "%Y-%m-%dT%H:%M:%SZ")
        at_dt = at_dt.replace(tzinfo=timezone.utc)
        if (at_dt - dt).total_seconds() <= 24 * 3600:
            x = math.log1p((dt - EPOCH_2026_11_19).total_seconds() / 3600.0)
            last24.append((x, row["rates"]))
    out = {}
    for idx, tr in enumerate(case["trophies"]):
        if len(last24) < 3:
            out[tr["id"]] = fallback[tr["id"]]
            continue
        xs = [x for x, _ in last24]
        ys = [logit(rates[idx] if rates[idx] is not None else 0.01)
              for _, rates in last24]
        a, b, resid = _ols(xs, ys)
        med = a + b * x_target
        sigma = math.sqrt(rw * rw + resid * resid)
        out[tr["id"]] = [round(inv_logit(med + z * sigma), 4) for z in Z]
    return out


# ---------------------------------------------------------------------------
# our program, run end to end as a subprocess
# ---------------------------------------------------------------------------

def run_ours(case):
    with tempfile.TemporaryDirectory() as td:
        in_dir = os.path.join(td, "in")
        out_dir = os.path.join(td, "out")
        os.makedirs(in_dir)
        os.makedirs(out_dir)
        with open(os.path.join(in_dir, "synthetic.in"), "w",
                  encoding="utf-8") as fh:
            json.dump(case, fh)
        env = dict(os.environ)
        env["MAKEFUN_INPUT_DIR"] = in_dir
        env["MAKEFUN_OUTPUT_DIR"] = out_dir
        proc = subprocess.run([sys.executable, os.path.join(ROOT, "main.py")],
                              env=env, capture_output=True, text=True,
                              timeout=60)
        if proc.returncode != 0:
            raise RuntimeError("main.py failed: " + proc.stderr[-500:])
        with open(os.path.join(out_dir, "synthetic.out"), "r",
                  encoding="utf-8") as fh:
            data = json.load(fh)
    return {row["id"]: row["q"] for row in data["trophies"]}


def main():
    totals = {"ours": 0.0, "persistence": 0.0, "curve": 0.0}
    tier_totals = {}
    tier_counts = {}
    print(f"{'seed':>5} {'ours':>9} {'persist':>9} {'curve':>9}")
    for seed in SEEDS:
        case, truth, tier_of = build_world(seed)
        ours = run_ours(case)
        pers = baseline_persistence(case)
        curve = baseline_curve(case)
        losses = {"ours": pinball_mean(ours, truth),
                  "persistence": pinball_mean(pers, truth),
                  "curve": pinball_mean(curve, truth)}
        for k in totals:
            totals[k] += losses[k]
        print(f"{seed:>5} {losses['ours']:>9.4f} {losses['persistence']:>9.4f} "
              f"{losses['curve']:>9.4f}")
        # per-tier decomposition, ours vs persistence (summed pinball)
        for tid, y in truth.items():
            tier = tier_of[tid]
            for name, fc in (("ours", ours), ("persistence", pers)):
                s = 0.0
                for tau, q in zip(LEVELS, fc[tid]):
                    d = logit(y) - logit(q)
                    s += 2.0 * max(tau * d, (tau - 1.0) * d)
                key = (tier, name)
                tier_totals[key] = tier_totals.get(key, 0.0) + s / len(LEVELS)
                tier_counts[key] = tier_counts.get(key, 0) + 1
    n = len(SEEDS)
    avg = {k: v / n for k, v in totals.items()}
    print(f"{'AVG':>5} {avg['ours']:>9.4f} {avg['persistence']:>9.4f} "
          f"{avg['curve']:>9.4f}")
    print("\nper-tier mean loss (ours vs persistence):")
    for tier in ("platinum", "gold", "silver", "bronze"):
        o = tier_totals.get((tier, "ours"), 0.0) / max(1, tier_counts.get((tier, "ours"), 1))
        p = tier_totals.get((tier, "persistence"), 0.0) / max(1, tier_counts.get((tier, "persistence"), 1))
        print(f"  {tier:<9} ours {o:.4f}   persistence {p:.4f}")
    ok = avg["ours"] < avg["persistence"] and avg["ours"] < avg["curve"]
    print("\nVERDICT:", "PASS - ours beats both baselines on average"
          if ok else "FAIL - ours does not beat both baselines")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
