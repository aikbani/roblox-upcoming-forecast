# Roblox Up-and-Coming Top-10 Forecaster

Entry for `roblox-up-and-coming-forecast-01` on make.fun: a program that reads
the history of Roblox's Up-and-Coming chart and forecasts its top 10, a
probability for each game at each place, for the 72 hours after entries close.

## Run

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads `$MAKEFUN_INPUT_DIR/task.json` and `$MAKEFUN_INPUT_DIR/history.ndjson`
(defaults `/input`), writes `$MAKEFUN_OUTPUT_DIR/forecast.json` (default
`/output`). Python 3.13, standard library only, no network, deterministic.

## Model

Per-game drift-diffusion over ranks. Each game's rank is a Gaussian random
walk: its mean drifts with its recent rank velocity (shrunk toward zero) and
its spread grows with horizon, calibrated from the history itself (the
standard deviation of rank changes at several lags). Place probabilities are
the time-average of that Gaussian over the window, so early hours stay sharp
on the current chart and later hours go diffuse. Probabilities per place are
normalised to 0.85 named mass, leaving 0.15 for games the model does not name.

## Practice result

On the kit's practice window (history to 2026-11-07T23:00Z, scored against the
10 archive reads to 2026-11-10T23:00Z) this scores **-35.56** (typical
probability 0.029 per place), against the house baseline's **-43.51**
(typical 0.013).
