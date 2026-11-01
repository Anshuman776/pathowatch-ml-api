"""
Area-level surveillance analysis (Objectives 3 & 4 of the PathoWatch synopsis), for monthly data.

Input : per area, a monthly series of {period, isolates, mdr_cases}.
Output: per area
    - latest month figures, change vs previous month, 3-month baseline
    - anomaly flags (Poisson excess over baseline; moving-average rule)
    - hotspot: species-free O/E vs the whole district this month, with random-effects funnel limits
    - forecast of MDR cases for the next months (+ 90% interval)
    - risk score 0..1 = 0.40 hotspot + 0.30 rise + 0.15 anomaly + 0.15 outlook  (same weights as the paper)
    - TWO wordings:
        staff   : risk_level High/Medium/Low + exact reasons (for the authority portal)
        public  : calm monitoring level + neutral message (for public pages, avoids alarm)
"""
import math

import numpy as np

from .forecasting import forecast

PUBLIC_LEVELS = [
    (0.25, "Stable", "Activity is within the usual range for this area."),
    (0.45, "Moderate Activity", "Activity is slightly above the usual range. Routine monitoring continues."),
    (0.70, "Increasing Activity", "Activity has increased recently. Health authorities are reviewing the data."),
    (1.01, "Requires Closer Monitoring", "Health authorities are monitoring this area more closely. "
                                         "No special action is needed by the public beyond usual precautions."),
]
PUBLIC_ADVICE = ("Take antibiotics only when prescribed, complete the course, keep hands clean, and stay up to date "
                 "with vaccines. Speak to a doctor if an infection is not improving.")


def _poisson_upper_z(obs: float, exp: float) -> float:
    return (obs - exp) / math.sqrt(exp) if exp > 0 else (3.0 if obs > 0 else 0.0)


def _funnel(areas_latest):
    """O/E of MDR cases vs district rate, with DerSimonian-Laird random-effects SD (as in the paper)."""
    tot_iso = sum(a["isolates"] for a in areas_latest)
    tot_mdr = sum(a["mdr_cases"] for a in areas_latest)
    p = tot_mdr / tot_iso if tot_iso else 0
    rows = []
    for a in areas_latest:
        E = a["isolates"] * p
        if E <= 0:
            rows.append(None)
            continue
        R = a["mdr_cases"] / E
        s2 = (1 - p) / E
        rows.append((R, s2))
    valid = [r for r in rows if r]
    tau2 = 0.0
    if len(valid) >= 3:
        w = np.array([1 / s2 for _, s2 in valid])
        z2 = np.array([(R - 1) ** 2 / s2 for R, s2 in valid])
        q = z2.sum()
        tau2 = max(0.0, (q - (len(valid) - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    out = []
    for r in rows:
        if r is None:
            out.append({"ratio": None, "z": 0.0, "z_random_effects": 0.0, "above_998": False})
            continue
        R, s2 = r
        z = (R - 1) / math.sqrt(s2)
        z_re = (R - 1) / math.sqrt(s2 + tau2)
        out.append({"ratio": round(R, 3), "z": round(z, 2), "z_random_effects": round(z_re, 2), "above_998": z > 3.09})
    return out, p, tau2


def analyze(areas: list[dict], horizon: int = 3) -> dict:
    prepared = []
    for a in areas:
        s = sorted(a["series"], key=lambda r: r["period"])
        if not s:
            continue
        prepared.append({**a, "series": s, "latest": s[-1]})
    funnel, district_rate, tau2 = _funnel([p["latest"] for p in prepared])

    results = []
    for a, fun in zip(prepared, funnel):
        s = a["series"]
        cases = [r["mdr_cases"] for r in s]
        iso = [r["isolates"] for r in s]
        last, prev = s[-1], (s[-2] if len(s) >= 2 else None)
        base_cases = float(np.mean(cases[-4:-1])) if len(s) >= 4 else (float(np.mean(cases[:-1])) if len(s) > 1 else None)
        base_iso = float(np.mean(iso[-4:-1])) if len(s) >= 4 else (float(np.mean(iso[:-1])) if len(s) > 1 else None)
        base_rate = (sum(cases[-4:-1]) / sum(iso[-4:-1])) if len(s) >= 4 and sum(iso[-4:-1]) else None
        rate = last["mdr_cases"] / last["isolates"] if last["isolates"] else 0.0
        change = ((last["mdr_cases"] - prev["mdr_cases"]) / prev["mdr_cases"] * 100) if prev and prev["mdr_cases"] else None

        # anomaly 1: more MDR cases than expected from the baseline MDR rate and this month's testing volume
        exp = base_rate * last["isolates"] if base_rate is not None else None
        z_pois = _poisson_upper_z(last["mdr_cases"], exp) if exp is not None else 0.0
        excess = (last["mdr_cases"] - exp) if exp is not None else 0
        poisson_flag = exp is not None and z_pois > 2.5 and excess >= 3
        # anomaly 2: moving-average rule on case counts
        sd = float(np.std(cases[-4:-1])) if len(s) >= 4 else 0.0
        ma_flag = base_cases is not None and last["mdr_cases"] > base_cases + max(3, 2 * sd)
        anomaly = poisson_flag or ma_flag

        fc = None
        if len(cases) >= 3:
            fc = forecast(cases, horizon)
            fc["periods"] = _next_periods(last["period"], horizon)

        s_hot = min(max(fun["z_random_effects"] / 1.96, 0), 1)
        growth = ((last["mdr_cases"] - base_cases) / base_cases * 100) if base_cases else 0.0
        s_rise = min(max(growth / 50, 0), 1)                      # +50% over baseline -> full score
        s_anom = 1.0 if anomaly else 0.0
        s_out = min(max(((fc["forecast"][0] - base_cases) / base_cases) / 0.5, 0), 1) if fc and base_cases else 0.0
        risk = round(0.40 * s_hot + 0.30 * s_rise + 0.15 * s_anom + 0.15 * s_out, 3)
        level = "High" if risk >= 0.7 else "Medium" if risk >= 0.45 else "Low"
        public_level, public_message = next((lbl, msg) for cut, lbl, msg in PUBLIC_LEVELS if risk < cut)

        reasons = []
        if fun["above_998"]:
            reasons.append(f"MDR cases are {fun['ratio']:.2f}x what the district rate predicts for this testing volume "
                           f"(above the 99.8% funnel limit).")
        if base_cases is not None and growth > 0:
            reasons.append(f"{last['mdr_cases']} MDR cases in {last['period']} vs a 3-month average of {base_cases:.1f} "
                           f"({growth:+.0f}%).")
        if poisson_flag:
            reasons.append(f"{excess:.0f} more MDR cases than expected from the recent MDR rate (z = {z_pois:.1f}).")
        if ma_flag:
            reasons.append("Monthly count is above the 3-month moving average by more than 2 SD.")
        if fc and base_cases and fc["forecast"][0] > base_cases * 1.1:
            reasons.append(f"Forecast for {fc['periods'][0]}: about {fc['forecast'][0]:.0f} cases "
                           f"(range {fc['lower'][0]:.0f}-{fc['upper'][0]:.0f}).")

        results.append({
            "area_id": a["area_id"], "area_name": a.get("area_name"),
            "latest": {"period": last["period"], "isolates": last["isolates"], "mdr_cases": last["mdr_cases"],
                       "mdr_rate": round(rate, 4)},
            "previous": ({"period": prev["period"], "isolates": prev["isolates"], "mdr_cases": prev["mdr_cases"]} if prev else None),
            "change_vs_previous_pct": round(change, 1) if change is not None else None,
            "baseline_cases": round(base_cases, 2) if base_cases is not None else None,
            "baseline_isolates": round(base_iso, 2) if base_iso is not None else None,
            "growth_vs_baseline_pct": round(growth, 1),
            "anomaly": {"flag": anomaly, "poisson_z": round(z_pois, 2), "expected_cases": round(exp, 2) if exp is not None else None,
                        "moving_average_flag": ma_flag},
            "hotspot": fun,
            "forecast": fc,
            "risk": {"score": risk, "level": level,
                     "components": {"hotspot": round(s_hot, 2), "rise": round(s_rise, 2),
                                    "anomaly": s_anom, "outlook": round(s_out, 2)},
                     "reasons": reasons},
            "public": {"level": public_level, "message": public_message, "advice": PUBLIC_ADVICE},
        })

    # district total
    periods = sorted({r["period"] for a in prepared for r in a["series"]})
    tot = [{"period": p, "isolates": sum(r["isolates"] for a in prepared for r in a["series"] if r["period"] == p),
            "mdr_cases": sum(r["mdr_cases"] for a in prepared for r in a["series"] if r["period"] == p)} for p in periods]
    district_fc = None
    if len(tot) >= 3:
        district_fc = forecast([t["mdr_cases"] for t in tot], horizon)
        district_fc["periods"] = _next_periods(periods[-1], horizon)
    return {"areas": sorted(results, key=lambda r: -r["risk"]["score"]),
            "district": {"series": tot, "forecast": district_fc, "mdr_rate_latest": round(district_rate, 4),
                         "between_area_sd": round(math.sqrt(tau2), 3)},
            "weights": {"hotspot": 0.40, "rise": 0.30, "anomaly": 0.15, "outlook": 0.15}}


def _next_periods(period: str, n: int) -> list[str]:
    y, m = map(int, period.split("-"))
    out = []
    for _ in range(n):
        m += 1
        if m > 12:
            y, m = y + 1, 1
        out.append(f"{y:04d}-{m:02d}")
    return out
