"""GTA VI trophy rarity forecast -- entry for gta6-trophy-rarity-forecast-01.

Model: "log-time counts + stale-refresh" forecast (original design).

For each trophy we forecast its Exophase earn rate at the case target as five
quantiles on the logit scale.  The median comes from two views of the same
history, blended in logit space, plus a correction for the one thing the
practice cases prove the house baselines miss: Exophase refreshes in jumps.

  1. Counts view.  A rate is holders / players_tracked.  Both counts grow
     roughly as powers of time since release, so we regress log(players) and
     log(holders + 0.5) separately against x = ln(1 + hours since release)
     over the recent history (the last 36 hours when the history is longer
     than that, so a saturating player count is not extrapolated as a power
     law; ordinary least squares, own code below) and read
     both lines at the target:  rate_counts = 100 * H(x_target) / N(x_target).
     This separates *dilution* (tracked players growing faster than holders,
     rates fall) from *progression* (holders growing faster, rates rise).

  2. Trend view.  We regress logit(rate) against the same x and read the
     line at the target.  This is the direct analogue of the counts view and
     agrees with it when the history is smooth; blending the two (50/50 in
     logit space) guards against either fit going wild on a noisy trophy.

  3. Stale-refresh correction.  When the whole history is flat -- the
     tracked count and every rate unchanged across all reads, as on launch
     night -- the page is stale: Exophase has not refreshed yet, and the
     next refresh lands inside the horizon.  The house recording shows the
     first refresh jumping 638 -> 3,868 tracked players (x6.06) while every
     rate but the 100% one fell.  So for a flat history we replace the
     (flat) trend median with  last_rate * H_i / R, where R is the expected
     refresh multiplier (6.0 for a ~90-minute horizon, growing slowly with
     horizon, capped at 20) and H_i = 1 + g_i * (R - 1) is how much that
     trophy's holders grow at the refresh.  g_i is a progression prior from
     the trophy itself: story/chapter trophies gain the most new holders
     (0.25), ordinary activities less (0.15), grind/silver/gold least
     (0.05-0.10), platinum almost none (0.02).  The 100% trophy is left at
     100: a refresh cannot lower it, and the rules say it did not fall.

  4. Spread (honest, per trophy).  sigma combines, in quadrature: the
     baseline random-walk term 0.5*sqrt(days); a stale term (0.85 logit
     when the history is flat, 0.12 otherwise -- the baseline's 0.125 at
     90 minutes missed every refreshed rate); the residual scatter of the
     trophy's own logit trend; a small-count term 0.5/sqrt(holders+1)
     (launch-night holders are single digits, so rates are noisy); plus a
     tier bump (platinum +0.25, gold +0.20, silver +0.10).  Sigma is
     floored at 0.35 and capped at 1.6.  Quantiles are
     sigmoid(logit(median) + z_tau * sigma) with z = standard normal
     quantiles, so the 90% band is wide enough to cover a refresh jump but
     tightens where the history is informative.

Guards: rates are clamped to [0.01, 99.99] before any logit; the total
logit move from the last read is clamped to [-3.5, +3.0]; trophies never
observed fall back to their tier's median last rate with a wider spread;
a history with a single read is treated as flat.  No randomness, no
network, standard library only; two runs are byte-identical.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

TIER_BUMP = {"platinum": 0.25, "gold": 0.20, "silver": 0.10, "bronze": 0.0}
TIER_FALLBACK_RATE = {"platinum": 0.5, "gold": 0.8, "silver": 0.8, "bronze": 1.0}


def _clamp_pct(x):
    return min(99.99, max(0.01, x))


def _logit(pct):
    p = _clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def _sigmoid_pct(logit_value):
    if logit_value >= 0:
        return 100.0 / (1.0 + math.exp(-logit_value))
    e = math.exp(logit_value)
    return 100.0 * e / (1.0 + e)


def _parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _round4(x):
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def _ols(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b, resid_std, n_distinct_x)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0
    if n == 1:
        return ys[0], 0.0, 0.0, 1
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        resid = math.sqrt(sum((y - my) ** 2 for y in ys) / n) if n else 0.0
        return my, 0.0, resid, len(set(xs))
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / n)
    return a, b, resid, len(set(xs))


def _progression_prior(trophy):
    """g_i: share of the refresh's new tracked players expected to have
    earned this trophy by the target. Story first, grind last."""
    tier = trophy.get("tier", "bronze")
    text = ((trophy.get("name") or "") + " " + (trophy.get("description") or "")).lower()
    if tier == "platinum":
        return 0.02
    if tier == "gold":
        return 0.06
    grind = ("100%" in text or "250 " in text or " all " in text
             or "every " in text or "each " in text)
    if tier == "silver":
        return 0.05 if grind else 0.10
    if "chapter" in text or "final mission" in text or "mission for" in text:
        return 0.25
    if grind:
        return 0.10
    return 0.15


def forecast_case(case):
    trophies = case["trophies"]
    history = case["history"]
    horizon_days = case["horizon_minutes"] / 1440.0
    horizon_hours = case["horizon_minutes"] / 60.0

    release = _parse_time(case["game"]["release_date"] + "T00:00:00Z") \
        if "T" not in str(case["game"].get("release_date", "")) \
        else _parse_time(case["game"]["release_date"])
    checkpoint = _parse_time(case["at"])
    target = _parse_time(case["target"])

    def hours_since_release(t):
        return max(0.0, (t - release).total_seconds() / 3600.0)

    def x_of(t):
        return math.log1p(hours_since_release(t))

    x_target = x_of(target)
    x_last = x_of(_parse_time(history[-1]["at"])) if history else x_of(checkpoint)

    # --- fits use the recent regime when history is long ------------------
    # Launch-night reads are a different regime from day-3 reads; when the
    # history spans more than 36 hours, fit on the last 36 hours only so a
    # saturating player count does not get extrapolated as a power law.
    # (Stale detection above still uses the full history.)
    fit_history = history
    if history:
        t_first = _parse_time(history[0]["at"])
        t_last = _parse_time(history[-1]["at"])
        if (t_last - t_first).total_seconds() > 36 * 3600:
            cutoff = t_last.timestamp() - 36 * 3600
            fit_history = [r for r in history
                           if _parse_time(r["at"]).timestamp() >= cutoff]

    # --- shared counts fit: log players_tracked vs x ---------------------
    xs_n, ys_n = [], []
    for read in fit_history:
        n = read.get("players_tracked")
        if n and n > 0:
            xs_n.append(x_of(_parse_time(read["at"])))
            ys_n.append(math.log(n))
    a_n, b_n, _, _ = _ols(xs_n, ys_n)
    log_n_last = math.log(history[-1]["players_tracked"]) if history and history[-1].get("players_tracked") else a_n
    log_n_pred = a_n + b_n * x_target

    # --- is the history flat (stale page, refresh pending)? ---------------
    all_n = [r["players_tracked"] for r in history if r.get("players_tracked")]
    flat_players = len(set(all_n)) <= 1 if all_n else True
    max_rate_move = 0.0
    for i in range(len(trophies)):
        vals = [r["rates"][i] for r in history
                if i < len(r["rates"]) and r["rates"][i] is not None]
        if len(vals) >= 2:
            max_rate_move = max(max_rate_move, max(vals) - min(vals))
    flat = flat_players and max_rate_move <= 0.05

    # Refresh multiplier for a flat history: ~6x at 90 minutes, growing
    # slowly with horizon (more backlog behind a longer-stale page).
    if flat:
        refresh_mult = 6.0 * math.sqrt(max(horizon_hours, 0.25) / 1.5)
        refresh_mult = min(20.0, max(1.5, refresh_mult))
    else:
        refresh_mult = 1.0

    # Tier fallback rates from the last read (for trophies never observed).
    last_rates = []
    if history:
        last_rates = history[-1]["rates"]
    tier_last = {}
    for idx, trophy in enumerate(trophies):
        if idx < len(last_rates) and last_rates[idx] is not None:
            tier_last.setdefault(trophy.get("tier", "bronze"), []).append(last_rates[idx])
    tier_median = {}
    for tier, vals in tier_last.items():
        vals = sorted(vals)
        tier_median[tier] = vals[len(vals) // 2]

    rows = []
    for idx, trophy in enumerate(trophies):
        tier = trophy.get("tier", "bronze")
        # Last observed rate and holders for this trophy.
        last_rate = None
        last_holders = 0.0
        if history and idx < len(last_rates) and last_rates[idx] is not None:
            last_rate = last_rates[idx]
            n_last = history[-1].get("players_tracked") or 0
            last_holders = last_rate * n_last / 100.0
        observed = last_rate is not None
        if not observed:
            last_rate = tier_median.get(tier, TIER_FALLBACK_RATE.get(tier, 1.0))

        # The 100% trophy cannot fall on a refresh; keep it pinned.
        if last_rate >= 99.5:
            rows.append({"id": trophy["id"], "q": [100, 100, 100, 100, 100]})
            continue

        # Per-trophy series: logit trend and log-holders, both vs x.
        xs_t, ys_logit, ys_logh = [], [], []
        for read in fit_history:
            if idx >= len(read["rates"]):
                continue
            r = read["rates"][idx]
            if r is None:
                continue
            n = read.get("players_tracked") or 0
            x = x_of(_parse_time(read["at"]))
            xs_t.append(x)
            ys_logit.append(_logit(r))
            ys_logh.append(math.log(max(0.0, r * n / 100.0) + 0.5))
        a_y, b_y, resid_y, n_distinct = _ols(xs_t, ys_logit)
        a_h, b_h, _, _ = _ols(xs_t, ys_logh)

        logit_last = _logit(last_rate)
        if flat:
            # Stale-refresh median: holders grow by H_i, players by R.
            g = _progression_prior(trophy)
            holders_mult = 1.0 + g * (refresh_mult - 1.0)
            median_rate = _clamp_pct(last_rate * holders_mult / refresh_mult)
            logit_med = _logit(median_rate)
            sigma_trend = 0.30
        else:
            # Counts view and trend view, blended in logit space.
            log_h_pred = a_h + b_h * x_target
            rate_counts = _clamp_pct(100.0 * math.exp(log_h_pred - log_n_pred))
            rate_trend = _clamp_pct(_sigmoid_pct(a_y + b_y * x_target))
            logit_med = 0.5 * _logit(rate_counts) + 0.5 * _logit(rate_trend)
            # Shrink toward the last read when the history is too short to
            # trust a slope (few distinct times).
            w = n_distinct / (n_distinct + 2.0) if n_distinct else 0.0
            logit_med = w * logit_med + (1.0 - w) * logit_last
            sigma_trend = resid_y
        # Never let the median run away from the last read.
        logit_med = logit_last + max(-3.5, min(3.0, logit_med - logit_last))

        # --- spread -------------------------------------------------------
        sigma_base = 0.5 * math.sqrt(horizon_days)
        sigma_stale = 0.85 if flat else 0.12
        sigma_binom = 0.5 / math.sqrt(last_holders + 1.0)
        sigma = math.sqrt(sigma_base ** 2 + sigma_stale ** 2
                          + min(sigma_trend, 0.8) ** 2 + sigma_binom ** 2)
        sigma += TIER_BUMP.get(tier, 0.0)
        if not observed:
            sigma += 0.30
        sigma = min(1.6, max(0.35, sigma))

        qs = []
        for z in Z:
            q = _sigmoid_pct(logit_med + z * sigma)
            qs.append(_round4(min(100.0, max(0.0, q))))
        # Enforce non-decreasing (rounding can only tie, but be safe).
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})

    rows.sort(key=lambda r: r["id"])
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = OUTPUT_DIR / (path.stem + ".out")
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
