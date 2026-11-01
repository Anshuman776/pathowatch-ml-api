"""
PathoWatch ML Service - the only place where trained models run.

The website never loads pickle/joblib files itself; it calls this service over HTTP.
That keeps the two parts independently deployable:
    ML owner   -> retrains, exports a new artifact, reloads this service
    Web owner  -> changes pages; talks to the stable /v1 API below

Run locally:  uvicorn app.main:app --port 8001
Docs:         http://127.0.0.1:8001/docs
"""
import hmac
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from . import mdr, risk, surveillance
from .config import settings
from .forecasting import forecast as run_forecast
from .model_store import store
from .schemas import AnalyzeRequest, ClassifyRequest, ForecastRequest, PredictRequest

logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("pathowatch.api")
API_VERSION = "1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.load()
    risk.warmup(store.current)
    if not settings.api_keys:
        log.warning("ML_API_KEYS is not set - the API is OPEN. Set it in production.")
    yield


app = FastAPI(title="PathoWatch ML Service", version=API_VERSION, lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1000)
if settings.allowed_origins:
    app.add_middleware(CORSMiddleware, allow_origins=settings.allowed_origins, allow_methods=["GET", "POST"],
                       allow_headers=["X-API-Key", "Content-Type"])


@app.middleware("http")
async def request_log(request: Request, call_next):
    rid = request.headers.get("X-Request-Id") or uuid.uuid4().hex[:12]
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("Unhandled error rid=%s path=%s", rid, request.url.path)
        response = JSONResponse({"error": "internal_error", "request_id": rid}, status_code=500)
    ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Request-Id"] = rid
    response.headers["X-Model-Version"] = store.current.version if store.ready else "none"
    log.info("%s %s %s %.0fms rid=%s", request.method, request.url.path, response.status_code, ms, rid)
    return response


def require_key(x_api_key: str = Header(default="")):
    if settings.api_keys and not any(hmac.compare_digest(x_api_key, k) for k in settings.api_keys):
        raise HTTPException(401, "Invalid or missing X-API-Key")


def require_admin(x_admin_key: str = Header(default="")):
    if not settings.admin_key or not hmac.compare_digest(x_admin_key, settings.admin_key):
        raise HTTPException(401, "Invalid or missing X-Admin-Key")


# ---------------------------------------------------------------- health
@app.get("/health", tags=["ops"])
def health():
    ok = store.ready
    body = {"status": "ok" if ok else "unavailable", "api_version": API_VERSION,
            "model_version": store.current.version if ok else None, "last_reload_error": store.error,
            "auth": "enabled" if settings.api_keys else "DISABLED"}
    return JSONResponse(body, status_code=200 if ok else 503)


@app.get("/v1/model-info", tags=["ops"], dependencies=[Depends(require_key)])
def model_info():
    m = store.current
    meta = m.bundle["meta"]
    t = meta.get("temporal", {})
    return {"version": meta["version"], "algorithm": meta["algorithm"], "feature_set": meta["feature_set"],
            "trained_on_years": meta.get("years"), "training_rows": meta.get("n_rows"), "created": meta.get("created"),
            "validation": {"auroc": t.get("auroc"), "calibration_error": t.get("ece"), "brier": t.get("brier"),
                           "test_years": meta.get("test_years")},
            "artifact": {"source": m.source, "sha256": m.sha256},
            "mdr_definition": "Magiorakos et al. 2012 (non-susceptible in >= 3 categories)"}


# ---------------------------------------------------------------- MDR labelling
@app.post("/v1/mdr/classify", tags=["mdr"], dependencies=[Depends(require_key)])
def classify(req: ClassifyRequest):
    """Label isolates as MDR / not MDR from their antibiotic results (standard definition)."""
    out = [{"id": i.id, **mdr.classify(i.pathogen, i.results)} for i in req.isolates]
    return {"results": out, "summary": {"isolates": len(out), "assessed": sum(r["assessed"] for r in out),
                                        "mdr": sum(bool(r["mdr"]) for r in out)}}


# ---------------------------------------------------------------- risk before AST
@app.post("/v1/risk/predict", tags=["risk"], dependencies=[Depends(require_key)])
def predict(req: PredictRequest):
    """MDR likelihood for a new isolate before lab results (for clinicians / authorities, not the public)."""
    m = store.current
    return risk.predict(m, req.model_dump()) | {"model_version": m.version}


# ---------------------------------------------------------------- surveillance
@app.post("/v1/surveillance/analyze", tags=["surveillance"], dependencies=[Depends(require_key)])
def analyze(req: AnalyzeRequest):
    """Per-area anomaly, hotspot, forecast and risk score from monthly counts."""
    body = surveillance.analyze([a.model_dump() for a in req.areas], req.horizon)
    return body | {"model_version": store.current.version, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


@app.post("/v1/forecast", tags=["surveillance"], dependencies=[Depends(require_key)])
def forecast(req: ForecastRequest):
    try:
        return run_forecast(req.values, req.horizon, req.upper_bound)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


# ---------------------------------------------------------------- ops
@app.post("/v1/admin/reload", tags=["ops"], dependencies=[Depends(require_admin)])
def reload_model():
    """Load the artifact again (after you replaced the file or changed MODEL_URL). Zero downtime."""
    old = store.current.version
    store.load()
    if store.error:
        raise HTTPException(500, f"Reload failed, still serving {old}: {store.error}")
    risk.warmup(store.current)
    return {"previous_version": old, "current_version": store.current.version}
