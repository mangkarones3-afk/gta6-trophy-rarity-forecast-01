#!/usr/bin/env python3
"""GTA VI trophy earn-rate forecaster.

Reads one make.fun case file and writes five quantiles (5/25/50/75/95) of
every trophy's earn rate at the case target.

Method (see MODEL.md for the full write-up):
  * The published rate is a ratio, earners / players tracked.  The history is
    first compressed to distinct snapshots (consecutive reads that repeat the
    same players count and the same rates are one observation of the page).
  * Four median candidates are computed per trophy:
      A - counts decomposition: log-time straight lines for ln(players) and
          ln(1 + earners), forecast separately, earners floored at the last
          observed earners and capped at the forecast players.  Both slopes
          are damped toward zero as the horizon grows relative to the
          observed span, because earner growth saturates while a log-log
          line does not (an undamped line reads saturation as a power law
          and explodes over a multi-day horizon);
      B - direct logit trend of the rate against ln(1 + hours), with the
          slope shrunk toward the tier pool and the global pool, the level
          anchored at the last rate, and the trend damped when the horizon
          is long compared with the observed span;
      C - persistence: the last observed rate;
      D - sustained trend: an undamped, unshrunk least-squares line of
          logit(rate) against ln(1 + hours) fitted on the last 24 hours of
          distinct snapshots (all of them when fewer than three fall in
          that window) and extrapolated straight to the target, intercept
          included.  Where the window holds fewer than three usable points
          for a trophy, or its time axis has no spread, D falls back to
          persistence for that trophy.
    Forms A and B fit on a recency window - the last 60% of the observed
    span (at least three snapshots): launch-night hours are a different
    regime (the initial surge), and a whole-history fit reads the early
    steep phase as the trend and overshoots a saturating curve for days.
  * An internal walk-forward backtest over up to three cut points inside the
    case's own history scores the four forms per trophy and turns the
    scores into blend weights (a softmax over losses, each cut weighted
    by its horizon so the longest, most forecast-like cut speaks
    loudest).  Each trophy's weight vector is then shrunk toward its
    tier's pooled vector and toward the global vector, the trophy's own
    evidence gaining weight as the number of backtest cuts grows: a
    trophy visibly mid-rise can lean on the trend forms without dragging
    the flat trophies along.  Finally, a trophy whose recent window shows
    a clean *accelerating* rise hands part of the damped forms' weight
    to Form D, whose lack of damping is exactly what a steepening
    S-curve rewards and what no in-history backtest cut can measure.
    With too few distinct snapshots a conservative default blend is used.
  * The spread combines, in quadrature, the house random-walk scale
    0.5*sqrt(horizon days), a scaled backtest residual sigma, and an observed
    refresh-jump sigma, clamped to [0.25x, 2.5x] of the house sigma.  The
    house scale enters at full strength when no backtest is possible; when
    the backtest proves the blended median's errors are far smaller than
    the house scale, it enters at 30% as a residual floor for the refresh
    timing risk a backtest cannot see.  The backtest sigma is robust
    (median absolute residual, not RMS): a single wild cut must not be
    allowed to widen every interval.  Quantiles are inverse-logit of
    median logit + z * sigma.

Standard library only, no network, no clock, no randomness: two runs on the
same case produce byte-identical output.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

Z_LEVELS = (-1.645, -0.674, 0.0, 0.674, 1.645)
RATE_LO = 0.01
RATE_HI = 99.99
DEFAULT_WEIGHTS = (0.30, 0.20, 0.35, 0.15)  # A, B, C, D
N_FORMS = 4


# --------------------------------------------------------------------------
# small numeric helpers
# --------------------------------------------------------------------------

def clamp(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def clamp_rate(r):
    return clamp(r, RATE_LO, RATE_HI)


def logit_rate(r):
    p = clamp_rate(r) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_rate(z):
    if z >= 0.0:
        e = math.exp(-z)
        return 100.0 / (1.0 + e)
    e = math.exp(z)
    return 100.0 * e / (1.0 + e)


def parse_time(text):
    """Parse an ISO-8601 UTC timestamp to epoch seconds (float)."""
    s = text.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def ols_slope(xs, ys):
    """Slope of the least-squares line y ~ x; 0.0 when x has no spread."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = 0.0
    sxy = 0.0
    for x, y in zip(xs, ys):
        dx = x - mx
        sxx += dx * dx
        sxy += dx * (y - my)
    if sxx <= 1e-12:
        return 0.0
    return sxy / sxx


def ols_line(xs, ys):
    """(intercept, slope) of the least-squares line y ~ x."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return ys[0], 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    slope = ols_slope(xs, ys)
    return my - slope * mx, slope


def damp_factor(span_h, horizon_h, floor):
    """Partial damping of an extrapolated slope: `floor` plus the remaining
    share scaled by span / (span + horizon).  A horizon much longer than the
    observed span pushes the factor toward `floor`."""
    if span_h + horizon_h <= 0:
        return 1.0
    raw = span_h / (span_h + horizon_h)
    return floor + (1.0 - floor) * raw


FIT_WINDOW_FRAC = 0.6


def fit_window(snaps):
    """The snapshots a trend fit may use: the last FIT_WINDOW_FRAC of the
    observed span, never fewer than three snapshots."""
    if len(snaps) <= 3:
        return snaps
    span = snaps[-1].hours - snaps[0].hours
    if span <= 0:
        return snaps
    cut = snaps[-1].hours - FIT_WINDOW_FRAC * span
    win = [s for s in snaps if s.hours >= cut - 1e-9]
    if len(win) < 3:
        win = snaps[-3:]
    return win


# --------------------------------------------------------------------------
# case preparation
# --------------------------------------------------------------------------

class Snapshot:
    __slots__ = ("hours", "players", "rates")

    def __init__(self, hours, players, rates):
        self.hours = hours          # hours since the first distinct snapshot
        self.players = players      # players tracked (int)
        self.rates = rates          # list of float-or-None, trophy order


def distinct_snapshots(case):
    """Compress the raw history to consecutive-distinct snapshots.

    A read that repeats the previous read's players count and every rate is
    the same page state seen again, not new information, so it is dropped.
    Returns (snapshots, target_hours) with hours measured from the first
    distinct snapshot.
    """
    history = case.get("history") or []
    trophies = case.get("trophies") or []
    n_tr = len(trophies)
    raw = []
    for row in history:
        rates_in = row.get("rates") or []
        rates = []
        for i in range(n_tr):
            v = rates_in[i] if i < len(rates_in) else None
            rates.append(None if v is None else float(v))
        stamp = row.get("at") or row.get("slot")
        raw.append((parse_time(stamp), int(row.get("players_tracked") or 0), rates))
    raw.sort(key=lambda item: item[0])

    snaps = []
    for t, players, rates in raw:
        if snaps and snaps[-1][1] == players and snaps[-1][2] == rates:
            continue
        snaps.append((t, players, rates))
    if not snaps:
        return [], 0.0

    t0 = snaps[0][0]
    out = [Snapshot((t - t0) / 3600.0, players, rates) for (t, players, rates) in snaps]

    # Target time on the same clock.
    target_hours = None
    if case.get("target"):
        try:
            target_hours = (parse_time(case["target"]) - t0) / 3600.0
        except (ValueError, TypeError):
            target_hours = None
    if target_hours is None:
        horizon_h = float(case.get("horizon_minutes") or 0) / 60.0
        target_hours = out[-1].hours + horizon_h
    if target_hours < out[-1].hours:
        target_hours = out[-1].hours
    return out, target_hours


def last_known_rates(snaps, n_tr):
    """Last non-null rate per trophy (falling back to RATE_LO)."""
    last = [RATE_LO] * n_tr
    for snap in snaps:
        for i in range(n_tr):
            v = snap.rates[i]
            if v is not None:
                last[i] = v
    return last


def earners_of(snap, i):
    r = snap.rates[i]
    if r is None:
        return None
    return r * snap.players / 100.0


def _last_rate(snaps, i):
    for s in reversed(snaps):
        v = s.rates[i]
        if v is not None:
            return v
    return RATE_LO


# --------------------------------------------------------------------------
# the four median forms.  Each returns a list of forecast rates (percent),
# one per trophy, given a training snapshot list and a target hour.
# --------------------------------------------------------------------------

def form_a(snaps, target_hours, n_tr):
    """Counts decomposition: forecast players and earners, divide.

    Both log-time slopes are damped by the horizon/span factor with a low
    floor: earner counts saturate as a trophy's reachable population is
    exhausted, and an undamped log-log line mistakes that bend for a power
    law and overshoots by orders of magnitude over a multi-day horizon.
    """
    if not snaps:
        return [RATE_LO] * n_tr
    last = snaps[-1]
    if len(snaps) == 1:
        return [clamp_rate(v) for v in last_known_rates(snaps, n_tr)]
    snaps = fit_window(snaps)

    xs = [math.log1p(s.hours) for s in snaps]
    x_t = math.log1p(target_hours)
    span_h = last.hours - snaps[0].hours
    horizon_h = max(0.0, target_hours - last.hours)
    damp = damp_factor(span_h, horizon_h, 0.15)

    # Players line: ln N against ln(1 + hours), anchored at the last point.
    ys_n = [math.log(max(1, s.players)) for s in snaps]
    _a_n, b_n = ols_line(xs, ys_n)
    b_n = clamp(b_n, 0.0, 2.0) * damp
    a_n = ys_n[-1] - b_n * xs[-1]
    n_pred = math.exp(a_n + b_n * x_t)
    n_pred = max(n_pred, float(last.players), 1.0)

    out = []
    for i in range(n_tr):
        pairs = [(x, e) for x, s in zip(xs, snaps)
                 for e in [earners_of(s, i)] if e is not None]
        last_rate = _last_rate(snaps, i)
        if len(pairs) < 2:
            out.append(clamp_rate(last_rate))
            continue
        px = [p[0] for p in pairs]
        py = [math.log1p(p[1]) for p in pairs]
        _a_e, b_e = ols_line(px, py)
        b_e = clamp(b_e, -0.5, 2.0) * damp
        a_e = py[-1] - b_e * px[-1]
        e_pred = math.expm1(a_e + b_e * x_t)
        e_last = pairs[-1][1]
        if e_pred < e_last:
            e_pred = e_last            # earners never fall
        if e_pred > n_pred:
            e_pred = n_pred            # earners cannot exceed players
        if e_pred < 0.0:
            e_pred = 0.0
        out.append(clamp_rate(100.0 * e_pred / n_pred))
    return out


def form_b(snaps, target_hours, tiers, n_tr):
    """Direct logit trend with tier/global slope shrinkage and damping."""
    if not snaps:
        return [RATE_LO] * n_tr
    last = snaps[-1]
    last_rates = [_last_rate(snaps, i) for i in range(n_tr)]
    if len(snaps) == 1:
        return [clamp_rate(v) for v in last_rates]

    win = fit_window(snaps)
    xs = [math.log1p(s.hours) for s in win]
    x_last = xs[-1]
    x_t = math.log1p(target_hours)
    span_h = win[-1].hours - win[0].hours
    horizon_h = max(0.0, target_hours - last.hours)
    damp = damp_factor(span_h, horizon_h, 0.25)

    slopes = []
    for i in range(n_tr):
        pairs = [(x, logit_rate(s.rates[i])) for x, s in zip(xs, win)
                 if s.rates[i] is not None]
        if len(pairs) < 2:
            slopes.append(0.0)
        else:
            slopes.append(clamp(ols_slope([p[0] for p in pairs],
                                          [p[1] for p in pairs]), -3.0, 3.0))
    global_slope = sum(slopes) / len(slopes) if slopes else 0.0
    tier_sum = {}
    tier_cnt = {}
    for i, tier in enumerate(tiers):
        tier_sum[tier] = tier_sum.get(tier, 0.0) + slopes[i]
        tier_cnt[tier] = tier_cnt.get(tier, 0) + 1
    tier_slope = {t: tier_sum[t] / tier_cnt[t] for t in tier_sum}

    m = float(len(win))
    out = []
    for i in range(n_tr):
        shrunk = (m * slopes[i] + 3.0 * tier_slope[tiers[i]]
                  + 2.0 * global_slope) / (m + 5.0)
        z = logit_rate(last_rates[i]) + shrunk * (x_t - x_last) * damp
        out.append(clamp_rate(inv_logit_rate(z)))
    return out


def form_c(snaps, n_tr):
    """Persistence: last known rate per trophy."""
    if not snaps:
        return [RATE_LO] * n_tr
    return [clamp_rate(_last_rate(snaps, i)) for i in range(n_tr)]


def recent_window(snaps, window_h=24.0):
    """Distinct snapshots inside the last `window_h` hours of the observed
    span; the whole list when fewer than three snapshots fall inside it."""
    if len(snaps) <= 3:
        return snaps
    cut = snaps[-1].hours - window_h
    win = [s for s in snaps if s.hours >= cut - 1e-9]
    if len(win) < 3:
        return snaps
    return win


def form_d(snaps, target_hours, n_tr):
    """Sustained trend: per-trophy OLS of logit(rate) against ln(1 + hours
    since the first snapshot) on the recent window, extrapolated to the
    target with no horizon damping and no slope shrinkage, the intercept
    taken from the fit rather than anchored at the last rate.  This is the
    form that keeps faith with a rise that is still in progress at the end
    of the history; the blend, not this form, decides how much faith each
    trophy has earned.  Trophies whose window holds fewer than three
    usable points, or whose window has no time spread, get persistence."""
    if not snaps:
        return [RATE_LO] * n_tr
    last_rates = [_last_rate(snaps, i) for i in range(n_tr)]
    if len(snaps) == 1:
        return [clamp_rate(v) for v in last_rates]
    win = recent_window(snaps)
    x_t = math.log1p(target_hours)
    out = []
    for i in range(n_tr):
        pairs = [(math.log1p(s.hours), logit_rate(s.rates[i]))
                 for s in win if s.rates[i] is not None]
        if len(pairs) < 3:
            out.append(clamp_rate(last_rates[i]))
            continue
        xs = [p[0] for p in pairs]
        if max(xs) - min(xs) <= 1e-12:
            out.append(clamp_rate(last_rates[i]))
            continue
        a, b = ols_line(xs, [p[1] for p in pairs])
        out.append(clamp_rate(inv_logit_rate(a + b * x_t)))
    return out


def trend_shift(snaps, i):
    """Fraction of the damped forms' (A + B) blend weight that should move
    to the sustained-trend form D for trophy i, judged from the recent
    window alone.  The backtest cannot see this: A and B damp their slopes
    by a factor that shrinks as the horizon grows, so at backtest horizons
    they look better than they will be at the real, longer horizon, while
    D transfers unchanged.  The shift fires only on a rise that is
    *accelerating* - the recent window's logit slope clearly above the
    slope over the earlier part of the history, the signature of an
    S-curve still in its steepening phase - with a clean fit (window rise
    large against the fit's residual scatter).  A saturating curve
    decelerates, its recent slope falls below its early slope, and the
    shift stays zero."""
    win = recent_window(snaps)
    pairs = [(math.log1p(s.hours), logit_rate(s.rates[i]))
             for s in win if s.rates[i] is not None]
    if len(pairs) < 3:
        return 0.0
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    x_spread = max(xs) - min(xs)
    if x_spread <= 1e-12:
        return 0.0
    a, b = ols_line(xs, ys)
    if b <= 0.15:
        return 0.0
    rise = b * x_spread
    if rise < 0.25:
        return 0.0
    resid = math.sqrt(sum((y - (a + b * x)) ** 2
                          for x, y in pairs) / len(pairs))
    # Acceleration check against the part of the history *before* the
    # recent window (the fit window overlaps the recent window too much
    # to separate their slopes).
    cut = snaps[-1].hours - 24.0
    epairs = [(math.log1p(s.hours), logit_rate(s.rates[i]))
              for s in snaps if s.hours < cut - 1e-9
              and s.rates[i] is not None]
    if len(epairs) >= 2:
        exs = [p[0] for p in epairs]
        if max(exs) - min(exs) > 1e-12:
            b_early = ols_slope(exs, [p[1] for p in epairs])
            if b_early > 0.0 and b < 1.25 * b_early:
                return 0.0
    s = rise / (resid + 0.10)
    if s < 1.2:
        return 0.0
    return clamp(0.30 * (s - 1.2), 0.0, 0.75)


# --------------------------------------------------------------------------
# internal walk-forward backtest -> blend weights and residual sigma
# --------------------------------------------------------------------------

SOFTMAX_TAU = 0.06


def _weights_from_losses(losses):
    """Backtest losses -> a weight vector over the forms: a softmax over
    the negative losses.  A plain inverse-loss rule cannot separate forms
    whose losses are all large and close (the long-cut regime, where every
    form is far off but the trend forms are least far off); the softmax
    keeps the *relative* gap decisive at any loss scale."""
    lo = min(losses)
    inv = [math.exp(-(l - lo) / SOFTMAX_TAU) for l in losses]
    total = sum(inv)
    return tuple(v / total for v in inv)


def backtest(snaps, tiers, n_tr):
    """Train each form on prefixes of the history and forecast the final
    snapshot.  Cut points at 40%, 65% and 85% of the span: the earliest cut
    makes the backtest extrapolate about as far, relative to its training
    span, as the real forecast must.

    Losses are kept per trophy, and each cut's losses are weighted by
    that cut's horizon: the earliest cut extrapolates about as far,
    relative to its training span, as the real forecast must, so it is
    the cut that separates the forms the way the real horizon will; the
    late cuts mostly reward whichever form sits closest to the last rate.
    Each trophy's raw weight vector is shrunk toward its tier's pooled
    vector and toward the global vector; the trophy's own share grows
    with the number of usable cuts, so three cuts of clean evidence
    outweigh the pools while a single noisy cut cannot.  Returns
    (weights, residuals, mean_bt_horizon_hours) with weights a list of
    one N_FORMS tuple per trophy, or (None, [], 0.0) when the history is
    too short to backtest.
    """
    if len(snaps) < 5:
        return None, [], 0.0
    first_h = snaps[0].hours
    last_h = snaps[-1].hours
    span = last_h - first_h
    if span <= 0:
        return None, [], 0.0
    actual = [logit_rate(_last_rate(snaps, i)) for i in range(n_tr)]

    loss_sum = [[0.0] * n_tr for _ in range(N_FORMS)]  # [form][trophy]
    counts = 0
    weight_sum = 0.0
    horizons = []
    per_cut = []  # (form logits per trophy, horizon)
    for frac in (0.4, 0.65, 0.85):
        cut_h = first_h + frac * span
        train = [s for s in snaps if s.hours <= cut_h + 1e-9]
        if len(train) < 2 or len(train) >= len(snaps):
            continue
        horizon = last_h - train[-1].hours
        if horizon <= 0:
            continue
        preds = (form_a(train, last_h, n_tr),
                 form_b(train, last_h, tiers, n_tr),
                 form_c(train, n_tr),
                 form_d(train, last_h, n_tr))
        logits = [[logit_rate(v) for v in p] for p in preds]
        for f in range(N_FORMS):
            row = loss_sum[f]
            lf = logits[f]
            for i in range(n_tr):
                row[i] += horizon * abs(lf[i] - actual[i])
        counts += 1
        weight_sum += horizon
        horizons.append(horizon)
        per_cut.append((logits, horizon))
    if counts == 0:
        return None, [], 0.0
    mean_loss = [[loss_sum[f][i] / weight_sum for i in range(n_tr)]
                 for f in range(N_FORMS)]

    global_w = _weights_from_losses(
        [sum(mean_loss[f]) / n_tr for f in range(N_FORMS)])
    tier_members = {}
    for i, tier in enumerate(tiers):
        tier_members.setdefault(tier, []).append(i)
    tier_w = {}
    for tier, idxs in tier_members.items():
        tier_w[tier] = _weights_from_losses(
            [sum(mean_loss[f][i] for i in idxs) / len(idxs)
             for f in range(N_FORMS)])

    # Pooling: the trophy's own vector carries counts/(counts+1) of the
    # blend; the remainder goes 60% to its tier pool, 40% to the global
    # pool.  With the usual three cuts that is 0.75 / 0.15 / 0.10.
    own = counts / (counts + 1.0)
    w_tier = (1.0 - own) * 0.6
    w_glob = (1.0 - own) * 0.4
    weights = []
    for i in range(n_tr):
        raw = _weights_from_losses([mean_loss[f][i] for f in range(N_FORMS)])
        tw = tier_w[tiers[i]]
        w = tuple(own * raw[f] + w_tier * tw[f] + w_glob * global_w[f]
                  for f in range(N_FORMS))
        s = sum(w)
        weights.append(tuple(v / s for v in w))

    residuals = []
    for logits, _horizon in per_cut:
        for i in range(n_tr):
            blend = sum(weights[i][f] * logits[f][i] for f in range(N_FORMS))
            residuals.append(blend - actual[i])
    mean_horizon = sum(horizons) / len(horizons) if horizons else 0.0
    return weights, residuals, mean_horizon


# --------------------------------------------------------------------------
# spread
# --------------------------------------------------------------------------

def jump_sigma(snaps, i):
    """Largest single logit jump between consecutive distinct snapshots for
    trophy i inside the fit window, divided by sqrt(window size).  Only the
    recent regime counts: the launch surge's cliff-sized first refreshes
    say nothing about refresh risk near the target, and importing them
    would widen every interval for no reason."""
    win = fit_window(snaps)
    vals = [logit_rate(s.rates[i]) for s in win if s.rates[i] is not None]
    if len(vals) < 2:
        return 0.0
    biggest = 0.0
    for a, b in zip(vals, vals[1:]):
        d = abs(b - a)
        if d > biggest:
            biggest = d
    return biggest / math.sqrt(len(win))


# --------------------------------------------------------------------------
# one case
# --------------------------------------------------------------------------

def forecast_case(case):
    trophies = case.get("trophies") or []
    n_tr = len(trophies)
    tiers = [t.get("tier", "") for t in trophies]
    ids = [t.get("id") for t in trophies]
    snaps, target_hours = distinct_snapshots(case)

    horizon_minutes = float(case.get("horizon_minutes") or 0)
    if snaps:
        horizon_h = max(0.0, target_hours - snaps[-1].hours)
    else:
        horizon_h = horizon_minutes / 60.0
    horizon_days = horizon_h / 24.0
    house_sigma = 0.5 * math.sqrt(horizon_days) if horizon_days > 0 else 0.0

    if not snaps:
        rows = []
        for tid in ids:
            rows.append({"id": tid, "q": [0.01] * 5})
        return {"trophies": rows}

    pred_a = form_a(snaps, target_hours, n_tr)
    pred_b = form_b(snaps, target_hours, tiers, n_tr)
    pred_c = form_c(snaps, n_tr)
    pred_d = form_d(snaps, target_hours, n_tr)

    weights, residuals, bt_horizon = backtest(snaps, tiers, n_tr)
    if weights is None:
        weights = [DEFAULT_WEIGHTS] * n_tr
        sigma_bt = 0.0
        floor_frac = 1.0
    # Horizon-transfer correction: on a clean accelerating rise, move part
    # of the damped forms' weight to the undamped trend form (see
    # trend_shift).  The total weight is preserved.
    weights = list(weights)
    for i in range(n_tr):
        sh = trend_shift(snaps, i)
        if sh > 0.0:
            wi = list(weights[i])
            move = sh * (wi[0] + wi[1])
            wi[0] *= (1.0 - sh)
            wi[1] *= (1.0 - sh)
            wi[3] += move
            weights[i] = tuple(wi)
    else:
        if residuals:
            # Dispersion around the backtest residuals' own median, not
            # around zero: a systematic backtest bias (every form missing
            # a steepening rise in the same direction) is a median
            # problem, and folding it into the spread as if it were
            # scatter roughly doubles the intervals in exactly the
            # worlds where the trend forms are already fixing the bias.
            srt = sorted(residuals)
            med_res = srt[len(srt) // 2]
            abs_dev = sorted(abs(r - med_res) for r in residuals)
            robust = 1.4826 * abs_dev[len(abs_dev) // 2]
        else:
            robust = 0.0
        if bt_horizon > 0 and horizon_h > 0:
            # Errors of a saturating process grow slower than sqrt(horizon);
            # cap the horizon scaling so a short backtest cannot inflate
            # the spread without bound.  The cap is tight (1.3x): the
            # pooled residuals already contain the longest cut's errors,
            # and measured swap tests put the useful total spread near
            # the house scale, not at twice it.
            scale = min(math.sqrt(horizon_h / bt_horizon), 1.3)
        else:
            scale = 1.0
        sigma_bt = robust * scale * 1.15
        floor_frac = 0.30

    logits_a = [logit_rate(v) for v in pred_a]
    logits_b = [logit_rate(v) for v in pred_b]
    logits_c = [logit_rate(v) for v in pred_c]
    logits_d = [logit_rate(v) for v in pred_d]
    form_logits = (logits_a, logits_b, logits_c, logits_d)

    rows = []
    for i in range(n_tr):
        wi = weights[i]
        med_logit = sum(wi[f] * form_logits[f][i] for f in range(N_FORMS))
        sig_c = jump_sigma(snaps, i)
        floor_sig = floor_frac * house_sigma
        sigma = math.sqrt(floor_sig * floor_sig + sigma_bt * sigma_bt
                          + sig_c * sig_c)
        if house_sigma > 0:
            sigma = clamp(sigma, 0.25 * house_sigma, 2.5 * house_sigma)
        qs = []
        for z in Z_LEVELS:
            v = inv_logit_rate(med_logit + z * sigma)
            v = clamp(v, 0.0, 100.0)
            qs.append(round(v, 4))
        # enforce non-decreasing after rounding
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": ids[i], "q": qs})
    return {"trophies": rows}


def main(argv=None):
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    try:
        names = sorted(os.listdir(in_dir))
    except FileNotFoundError:
        return 0
    for name in names:
        if not name.endswith(".in"):
            continue
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        out_name = name[:-3] + ".out"
        payload = json.dumps(result, separators=(",", ":"), ensure_ascii=False)
        with open(os.path.join(out_dir, out_name), "w", encoding="utf-8") as fh:
            fh.write(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
