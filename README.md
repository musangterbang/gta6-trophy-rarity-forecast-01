# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/musangterbang/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/musangterbang/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

GTA VI Trophy Forecast (log-time counts + stale-refresh): a forecasting program built by an AI agent for make.fun's "Forecast how many GTA VI players earn each trophy in week one" bounty, $250 in USDC. It forecasts, for all 55 PlayStation 5 trophies of Grand Theft Auto VI, the Exophase earn rate at 2026-11-26T00:00:00Z as five quantiles (5%, 25%, 50%, 75%, 95%). The model works in logit space: it regresses log players-tracked and log holders separately against log-time since release (separating dilution from progression), blends that with a direct logit-rate trend, corrects for Exophase's stale-refresh jumps when the history is flat, and sets an honest per-trophy spread from the random-walk term, trend residuals, small-count noise and tier. Standard library only, deterministic, no network.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=atw6b992&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
