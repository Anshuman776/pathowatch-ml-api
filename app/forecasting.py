"""
Short-term forecasting of a monthly series (e.g. MDR cases per month in an area).

Same approach that won the back-test in the PathoWatch study: an ensemble of simple, robust methods
(mean of the last 3 periods, ARIMA, damped trend). Prediction intervals come from the errors the
ensemble made when it was re-run on the series' own past (rolling-origin back-test).
"""
import warnings

import numpy as np


def _mean3(y, h):
    return np.full(h, float(np.mean(y[-3:])))


def _arima(y, h):
    from statsmodels.tsa.arima.model import ARIMA
    best, aic = None, np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for order, trend in (((0, 1, 1), "n"), ((1, 0, 0), "c")):
            try:
                r = ARIMA(y, order=order, trend=trend).fit()
                if r.aic < aic:
                    best, aic = r, r.aic
            except Exception:
                pass
    return np.asarray(best.forecast(h)) if best is not None else _mean3(y, h)


def _damped(y, h, phi=0.8):
    """Damped linear trend on the last 6 points: the slope fades out so forecasts do not run away."""
    tail = np.asarray(y[-6:], float)
    slope = np.polyfit(np.arange(len(tail)), tail, 1)[0] if len(tail) >= 3 else 0.0
    level = float(np.mean(y[-2:]))
    return np.array([level + slope * sum(phi ** k for k in range(1, i + 2)) for i in range(h)])


METHODS = {"mean_last3": _mean3, "arima": _arima, "damped_trend": _damped}


def _ensemble(y, h):
    parts = {name: fn(y, h) for name, fn in METHODS.items()}
    parts["ensemble"] = np.mean(list(parts.values()), axis=0)
    return parts


def forecast(values: list[float], horizon: int = 3, upper_bound: float | None = None, backtest_points: int = 4) -> dict:
    y = np.asarray(values, float)
    if len(y) < 3:
        raise ValueError("At least 3 periods are needed to forecast")
    parts = _ensemble(y, horizon)

    # rolling-origin back-test on the series itself -> method errors + interval widths
    errors = {m: [[] for _ in range(horizon)] for m in parts}
    for cut in range(max(3, len(y) - backtest_points), len(y)):
        past, fut = y[:cut], y[cut:cut + horizon]
        for m, p in _ensemble(past, horizon).items():
            for i, v in enumerate(fut):
                errors[m][i].append(abs(p[i] - v))
    mae = {m: float(np.mean([e for hs in errs for e in hs])) if any(errs) else None for m, errs in errors.items()}

    point = parts["ensemble"]
    ens_err = errors["ensemble"]
    scale = float(np.std(np.diff(y))) if len(y) > 2 else float(np.std(y))
    width = []
    for i in range(horizon):
        e = ens_err[i] if i < len(ens_err) else []
        width.append(float(np.quantile(e, 0.9)) if len(e) >= 3 else 1.645 * scale * np.sqrt(i + 1))
    lo = np.maximum(point - width, 0)
    hi = point + width
    if upper_bound is not None:
        point, hi = np.minimum(point, upper_bound), np.minimum(hi, upper_bound)
    point = np.maximum(point, 0)
    return {
        "method": "ensemble (mean of last 3, ARIMA, damped trend)",
        "forecast": [round(float(v), 2) for v in point],
        "lower": [round(float(v), 2) for v in lo],
        "upper": [round(float(v), 2) for v in hi],
        "interval": "90%",
        "components": {m: [round(float(v), 2) for v in p] for m, p in parts.items() if m != "ensemble"},
        "backtest_mae": {m: (round(v, 2) if v is not None else None) for m, v in mae.items()},
    }
