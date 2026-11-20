# GTA VI trophy rarity forecast — model notes

Entry: `main.py` (Python 3.13, standard library only, no network, no
randomness). One entrypoint at the repo root; run as
`MAKEFUN_INPUT_DIR=<dir> MAKEFUN_OUTPUT_DIR=<dir> python3 main.py`.

## Model: "log-time counts + stale-refresh"

For each of the 55 trophies, the median at the target is built in logit
space from two views of the history, plus a correction for Exophase
refreshing in jumps:

1. **Counts view.** A rate is holders / players_tracked. Both counts are
   regressed in logs against `x = ln(1 + hours since release)` — log
   players and log(holders + 0.5) separately — over the recent history
   (last 36 h when the history is longer, so a saturating player count is
   not extrapolated as a power law), then read at the target:
   `rate = 100 * H(x_target) / N(x_target)`. This separates dilution
   (tracked players outgrowing holders, rates fall) from progression
   (holders outgrowing players, rates rise).
2. **Trend view.** Logit(rate) regressed against the same `x`, read at the
   target. Blended 50/50 with the counts view in logit space, then shrunk
   toward the last read when the history has few distinct times.
3. **Stale-refresh correction.** When the whole history is flat (tracked
   count and every rate unchanged, as on launch night), the page is stale
   and the next refresh lands inside the horizon. The house recording
   shows the first refresh jumping 638 → 3,868 tracked players (×6.06)
   with every rate but the 100% one falling. So for a flat history the
   median becomes `last_rate * H_i / R`, with `R ≈ 6` at a 90-minute
   horizon (growing slowly with horizon, capped at 20) and
   `H_i = 1 + g_i (R − 1)`, where `g_i` is a progression prior from the
   trophy itself: story/chapter 0.25, ordinary activities 0.15, grind
   0.05–0.10, gold 0.06, platinum 0.02. The 100% trophy stays at 100.
4. **Spread.** Sigma (logit) combines in quadrature: `0.5·sqrt(days)` (the
   baseline random-walk term), a stale term (0.85 when flat, 0.12
   otherwise — the baseline's 0.125 at 90 min missed every refreshed
   rate), the trophy's own logit-trend residual scatter, a small-count
   term `0.5/sqrt(holders+1)` (launch-night holders are single digits),
   plus a tier bump (platinum +0.25, gold +0.20, silver +0.10). Floored
   at 0.35, capped at 1.6. Quantiles are
   `sigmoid(logit(median) + z_tau · sigma)`, `z` = normal quantiles.

Guards: rates clamped to [0.01, 99.99] before any logit; total logit move
from the last read clamped to [−3.5, +3.0]; trophies never observed fall
back to their tier's median last rate with a wider spread; a one-read
history is treated as flat.

## Local test results (measured, 2026-11-20)

Commands (from the entry dir, kit inputs copied to a test dir):

```
mkdir -p out && MAKEFUN_INPUT_DIR=<test-in> MAKEFUN_OUTPUT_DIR=out python3 main.py
python3 check.py out   # check.py lives in gta6-forecast-test/, not the entry
```

- Shape: both `p00.out` and `p01.out` valid — 55 rows, ids 1–55 once
  each, quantiles non-decreasing in [0, 100], 2,765 bytes each (limit
  65,536). Runtime ~0.07–0.12 s per run (limit 8 s/case).
- Determinism: two runs byte-identical (`diff -r` clean).
- Baseline comparison (starter `main.py` run on the same inputs, same
  checker): baseline median never moves (mean |logit shift| 0.000).
  Mine moves a mean 1.26 logit units, in the direction of the refresh.
- Simulated refresh scores — **simulated, not real practice scores**
  (the true practice answers are not in the kit). Truth built from
  RULES.txt (N 638 → 3,868, every rate but 100% lower):
  - S1 holders frozen (`true = last × 638/3868`): mine 0.309, baseline 1.688.
  - S2 holders grow (story ×1.6, others ×1.15): mine 0.258, baseline
    1.490 — the baseline's S2 number reproduces the house's stated
    ~1.51 practice loss, which suggests the simulation is in the right
    ballpark. 90%-band coverage under S2: mine 1.00, baseline 0.00.
- Synthetic final-like case (60 h of reads, N growing, rates evolving;
  deliberately extreme, N to ~304k): runs clean, early-story medians
  rise (id 3: last 2.09% → median 2.92%), platinum falls (→ 0.01%).
- Edge case (single read, two null rates): runs clean; null trophies
  fall back to tier medians with wider spreads.

## Known weaknesses

- The stale-refresh multiplier `R ≈ 6` is calibrated on a single observed
  refresh (launch night); if the final's refreshes are smaller or the
  tracked count saturates, the flat-history median would overshoot the
  fall — but the final history is not flat, so this path should not fire
  there, and the wide spread covers the error.
- The log-time power-law extrapolation can over-extrapolate if player
  growth saturates sharply after the deadline; mitigated by the 36 h
  fit window and the [−3.5, +3.0] logit-move clamp.
- `g_i` progression priors are judgment calls from trophy text/tier, not
  fitted — they only matter on a flat history.
- Coverage 1.00 under the S2 simulation hints the spread may be a touch
  wide on a refresh scenario; the pinball loss still beats the baseline
  by ~5× there, so the trade-off is accepted.
