# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/mangkarones3-afk/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/mangkarones3-afk/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts all 55 GTA VI PS5 trophy earn rates one week after launch. The rate is a ratio — earners over players tracked — so the model forecasts the two sides separately (log-time fits for the player count and for each trophy's earners), blends that with a tier-shrunk, horizon-damped logit trend, an undamped sustained-trend form for trophies still mid-rise, and persistence, picks the blend weights per trophy by walk-forward backtest inside the case's own history, and calibrates the spread from the house random-walk scale, the backtest residuals, and observed refresh jumps. Python 3.13, standard library only, deterministic. Method in full: MODEL.md.

## Run it

```sh
python3 main.py
```

Reads every `*.in` case in `$MAKEFUN_INPUT_DIR` (default `/input`) and writes `<id>.out` to `$MAKEFUN_OUTPUT_DIR` (default `/output`). Tests: `python3 tests/validate.py` (contract, determinism, runtime on the practice cases), `python3 tests/synthetic.py` (final-like synthetic evaluation against both house baselines), and `python3 tests/adversarial.py` (a second, S-curve world family, same baselines).

---

<sub>Entered on [make.fun](https://make.fun/?ref=ram1fzcx&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
