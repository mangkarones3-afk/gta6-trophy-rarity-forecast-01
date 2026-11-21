# Model

## What is being forecast

For each of the 55 PlayStation 5 trophies of GTA VI, five quantiles
(5%, 25%, 50%, 75%, 95%) of the trophy's earn rate — the percent of
Exophase-tracked players who have earned it — at the case target time.
Scoring is mean pinball loss on the logit scale, so a miss from 1% to 2%
costs about as much as a miss from 33% to 50%: rare trophies matter as
much as common ones, and all modelling happens on the logit scale (or on
counts whose ratio is then read on that scale).

## The one idea everything follows from

A published rate is a ratio: earners / players tracked. Both sides move,
and they move differently. The tracked-player count jumps at every
Exophase refresh (which dilutes every rate even while earners rise), and
earners never decrease but saturate as a trophy's reachable population is
exhausted. A model that extrapolates the ratio directly, without
modelling the two sides, systematically misreads both effects — most
visibly for the platinum trophy, whose earners are nearly flat while the
denominator grows, so its rate falls by dilution alone.

## Pipeline

1. **Compression.** The history is reduced to distinct snapshots:
   consecutive reads with the same players count and identical rates are
   the same page state observed again, not new information. All fitting
   uses the distinct snapshots (a handful to a few dozen), which also
   keeps the runtime trivially inside the limit.

2. **Four median candidates** per trophy:
   - **Form A — counts decomposition.** Straight lines in log time,
     ln(1 + hours), are fitted for ln(players) and for ln(1 + earners)
     per trophy, each anchored at its last observed point. The forecast
     divides forecast earners by forecast players; earners are floored
     at the last observed earners and capped at the forecast players.
   - **Form B — direct logit trend.** A line of logit(rate) against
     ln(1 + hours) per trophy. The slope is shrunk toward the pooled
     slope of the trophy's tier and toward the global pooled slope
     (weights by snapshot count), and the level is anchored at the last
     observed rate, so only the slope extrapolates.
   - **Form C — persistence.** The last observed rate, unchanged.
   - **Form D — sustained trend.** A line of logit(rate) against
     ln(1 + hours) fitted on the recent window only (the last 24 hours
     of distinct snapshots), extrapolated with no horizon damping and
     no slope shrinkage, intercept from the fit. This is the candidate
     that speaks for S-curve worlds, where a trophy's earn rate is
     still mid-rise — or accelerating — when the history ends, as
     endgame trophies are in launch week while the first cohort of
     players reaches them. With fewer than three usable points it
     falls back to persistence for that trophy.

3. **Recency window.** Forms A and B fit on the last 60% of the observed
   span (at least three snapshots). Launch-night hours are a different
   regime — the initial surge — and a whole-history fit reads that steep
   phase as the trend. On saturating growth this overshoots for days;
   the window is what lets the fits see the bend.

4. **Horizon damping.** Both trend forms damp their slopes by a factor
   built from span / (span + horizon), applied partially (floors 0.15
   for Form A, 0.25 for Form B). Earner growth saturates while a
   log-time line does not; undamped, the counts form reads saturation
   as a power law and explodes over a 4.5-day horizon. Damping is the
   difference between a forecast and a fantasy.

5. **Self-calibration (blend weights).** The program backtests itself
   on the case's own history: at cut points of 40%, 65% and 85% of the
   span it trains all four forms on the prefix and forecasts the final
   snapshot; each cut's losses are weighted by its horizon, because the
   earliest cut is the one whose extrapolation most resembles the real
   forecast's. Losses become weights through a softmax (an inverse-loss
   rule cannot separate forms whose losses are all large and close,
   which is exactly the long-cut regime). Weights are computed **per
   trophy**, then shrunk toward the trophy's tier-pooled weights and
   toward the global weights (with three cuts the trophy keeps 75% of
   its own vector), so a trophy that is visibly mid-rise can lean on
   the trend forms without dragging the flat trophies along. One
   correction sits on top: the damped forms look better at backtest
   horizons than at the real horizon, because their damping factor
   shrinks as the horizon grows, while Form D does not change with
   horizon. When the recent window shows a clean, *accelerating* rise
   (recent slope at least 1.25× the early-history slope, with a large
   window gain relative to the fit residual), up to 75% of the damped
   forms' weight is shifted to Form D for that trophy. Saturating
   curves decelerate, so this gate stays closed in saturating worlds.
   With fewer than five distinct snapshots no backtest is possible and
   a conservative default blend (persistence-heavy) is used. The final
   median is the weighted blend of the four forms on the logit scale.

6. **Spread.** Sigma per trophy combines three components in quadrature:
   - the house random-walk scale, 0.5 * sqrt(horizon in days);
   - a backtest residual sigma: 1.4826 × the median absolute deviation
     of the blended backtest residuals **around their own median** (a
     robust, centred scale — a systematic median bias is a median
     problem, not a spread problem, and one wild cut must not widen
     every interval), scaled by sqrt(horizon ratio) capped at 1.3,
     times a 1.15 safety factor;
   - a refresh-jump sigma: the largest logit jump between consecutive
     distinct snapshots inside the fit window, divided by sqrt(window
     size). Only the recent regime counts; the launch surge's cliff
     jumps say nothing about refresh risk near the target.
   
   The house scale enters at full strength when no backtest is possible.
   When the backtest proves the blended median's errors are far smaller
   than the house scale, the house component enters at 30% — a residual
   floor for the one risk a backtest cannot measure, namely that the
   answer is the page state at Exophase's last refresh before the
   target, whenever that happens to fall. Sigma is clamped to
   [0.25×, 2.5×] of the house scale. Quantiles are
   inverse-logit(median logit + z·sigma) for
   z = −1.645, −0.674, 0, 0.674, 1.645, rounded to 4 decimals and made
   non-decreasing.

## Edge cases

- Rates of 0, 100 or null: rates are clamped to [0.01, 99.99] before
  any logit; a trophy whose last reads are null falls back to its last
  non-null rate, and to 0.01 if it was never observed.
- A history with a single distinct snapshot (both practice cases):
  all forms reduce to persistence and the spread is the house scale.
- No history at all: every trophy is forecast at 0.01 flat — schema
  valid, and the case cannot occur in this competition.

## Determinism and limits

No network, no wall clock, no randomness, sorted iteration only: two
runs on the same case are byte-identical. Runtime is a few milliseconds
per case (the modelling is O(snapshots × trophies) with at most three
backtest cuts), far inside the 8-second limit.

## Validation

`tests/validate.py` checks the exact output contract on the practice
cases p00/p01, byte-identical determinism, and runtime.
`tests/synthetic.py` builds deterministic final-like worlds (2.3 days
of history, players growing from 638 into the tens of thousands,
saturating per-tier earner curves, a target 4.52 days later) and scores
this program against the two house baselines re-implemented from the
rules text. Across four fixed seeds the program's mean pinball loss is
0.095 against 0.251 (persistence) and 0.234 (curve) — it wins every
seed and every trophy tier. `tests/adversarial.py` builds a second,
deliberately hostile family — earn rates as logistic S-curves whose
gold and silver inflections fall *after* the history ends — where the
program scores 0.392 against 0.826 (persistence) and 0.449 (curve);
inside that family the gold tier alone (2 trophies) still sits above
the curve baseline, while the mean over all 55 trophies wins.

## Known limits

- The synthetic worlds are two families of plausible dynamics
  (saturating exponentials; logistic S-curves). A real week-one surge
  pattern outside both families — for example a second player wave
  from a weekend or a sale — would be captured only through the
  spread, not the median.
- The house model forecast (a one-off model call made near the
  deadline) cannot be measured in advance; the margin above is against
  the two printed baselines only.
- The answer depends on Exophase's refresh timing, which no entrant
  controls. The spread prices that risk; it cannot remove it.
