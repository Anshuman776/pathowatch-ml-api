"""
MDR risk prediction for a new isolate BEFORE susceptibility results are available.

Model: "adaptive" LightGBM (see research paper extension). Inputs are the routine metadata
(species, specimen, ward, patient type, age group, gender, year) plus the site's recent lab
history (MDR rate over the last 3 years). The site's own history is preferred; if the site has
fewer than MIN_LOCAL_ISOLATES isolates, the national reference history from ATLAS is used.
"""
import math

import numpy as np
import pandas as pd

from . import vocab
from .config import settings
from .model_store import LoadedModel


def _shrunk(mdr: float, n: float, prior: float, m: float) -> float:
    return (mdr + m * prior) / (n + m) if n else math.nan


def _reference_rates(lm: LoadedModel, row: dict, prior: float) -> dict:
    """History features from the ATLAS tables for the reference country (latest available window)."""
    B, H = lm.bundle, lm.history
    years = [k[-1] for k in H["h_c"]["t"]]
    year = max(years) + 1
    W, M = B["window"], B["m_prior"]
    out = {}
    for name, h in H.items():
        vals = [row[k] if k != "country" else settings.reference_country for k in h["keys"]]
        s = n = 0
        for y in range(year - W, year):
            c = h["t"].get((*vals, y))
            if c:
                s, n = s + c[0], n + c[1]
        out[f"{name}_rate"], out[f"{name}_n"] = _shrunk(s, n, prior, M), n
    last = H["h_cs"]["t"].get((settings.reference_country, row["species"], year - 1))
    out["h_cs_trend"] = (_shrunk(last[0], last[1], prior, M) - out["h_cs_rate"]) if last else math.nan
    return out


def _local_rates(hist: dict, prior: float, m: float) -> dict:
    """hist = counts the website computed from its own database (see schemas.LocalHistory)."""
    def pair(key):
        c = hist.get(key) or {}
        n, s = float(c.get("isolates", 0) or 0), float(c.get("mdr", 0) or 0)
        return _shrunk(s, n, prior, m), n
    out = {}
    for name, key in [("h_c", "location"), ("h_cs", "location_species"),
                      ("h_css", "location_species_source"), ("h_csp", "location_species_ward")]:
        out[f"{name}_rate"], out[f"{name}_n"] = pair(key)
    last_rate, last_n = pair("location_species_last_12_months")
    out["h_cs_trend"] = (last_rate - out["h_cs_rate"]) if last_n and out["h_cs_n"] else math.nan
    return out


def predict(lm: LoadedModel, req: dict) -> dict:
    B = lm.bundle
    sp = vocab.species(req["species"])
    if sp is None:
        return {"supported": False, "reason": "Risk model covers E. coli, K. pneumoniae, P. aeruginosa and A. baumannii only."}
    row = {
        "species": sp,
        "source_group": vocab.source_group(req.get("sample_type")),
        "speciality": vocab.speciality(req.get("ward")),
        "patient_type": vocab.patient_type(req.get("patient_type")),
        "age_group": vocab.age_group(req.get("age"), req.get("age_group")),
        "gender": vocab.gender(req.get("gender")),
        "year": int(req.get("year") or 2018),
        "country": settings.reference_country,
    }
    prior = B["species_prior"].get(sp, B["overall_prior"])
    local = req.get("local_history") or {}
    local_n = float((local.get("location") or {}).get("isolates", 0) or 0)
    if local_n >= settings.min_local_isolates:
        feats = _local_rates(local, prior, B["m_prior"])
        source = "local"
    else:
        feats = _reference_rates(lm, row, prior)
        source = "reference"
    row.update(feats)

    X = pd.DataFrame([row])[B["features"]]
    for c, cats in B["categories"].items():
        v = X[c].astype(str)
        X[c] = pd.Categorical(v.where(v.isin(cats)), categories=cats)
    p = float(B["model"].predict_proba(X)[0, 1])
    thr = float(B["threshold"])
    band = "High" if p >= 0.6 else "Moderate" if p >= thr else "Low"
    typical = float(prior)
    return {
        "supported": True,
        "mdr_probability": round(p, 4),
        "risk_band": band,
        "predicted_mdr": p >= thr,
        "threshold": round(thr, 4),
        "typical_for_species": round(typical, 4),
        "interpretation": (f"Estimated MDR likelihood {p:.0%} for {sp} "
                           f"({'higher' if p > typical else 'lower'} than the {typical:.0%} typical for this species)."),
        "inputs_used": {k: row[k] for k in ("species", "source_group", "speciality", "patient_type", "age_group", "gender")},
        "history_source": ("this site's own lab history" if source == "local" else
                           f"national reference ({settings.reference_country}, ATLAS) - site has < "
                           f"{settings.min_local_isolates} isolates in the last 3 years"),
        "disclaimer": "Surveillance estimate before susceptibility testing. Not a diagnosis; treat according to AST results.",
    }


def warmup(lm: LoadedModel) -> None:
    predict(lm, {"species": "E. coli", "sample_type": "Urine", "year": 2018})
    np.random.seed(0)
