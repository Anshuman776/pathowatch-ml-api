"""
Loads the trained model artifact and keeps it in memory.

The artifact is ONE joblib file produced by `python ml/export_for_web.py` in the ML project. It holds:
    model          plain lightgbm.LGBMClassifier (no project code needed to unpickle)
    features       feature order used at training
    categories     category lists for categorical features
    threshold      probability cut-off for "predicted MDR"
    history        ATLAS lab-history tables (fallback when a site has little local data)
    species_prior, overall_prior, window, m_prior, meta

Hosting options (no code change needed, only env vars):
    MODEL_PATH=/models/model.joblib                      -> mounted volume / baked into the image
    MODEL_URL=https://.../model.joblib MODEL_SHA256=...  -> downloaded at startup (S3, GitHub release, GCS)
A new model can be activated without downtime via POST /v1/admin/reload.
"""
import hashlib
import logging
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import joblib

from .config import settings

log = logging.getLogger("pathowatch.model")


@dataclass
class LoadedModel:
    bundle: dict
    history: dict          # {name: {"keys": [...], "t": {(values..., year): (mdr, n)}}}
    source: str
    sha256: str

    @property
    def version(self) -> str:
        return self.bundle["meta"]["version"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_path() -> tuple[Path, str]:
    if settings.model_url:
        settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
        target = settings.model_cache_dir / Path(settings.model_url.split("?")[0]).name
        if not target.exists() or (settings.model_sha256 and _sha256(target) != settings.model_sha256):
            log.info("Downloading model from %s", settings.model_url)
            tmp = target.with_suffix(".part")
            urllib.request.urlretrieve(settings.model_url, tmp)  # noqa: S310 (URL comes from operator config)
            tmp.replace(target)
        return target, settings.model_url
    return settings.model_path, str(settings.model_path)


def _load() -> LoadedModel:
    path, source = _resolve_path()
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at {path}. Set MODEL_PATH or MODEL_URL.")
    digest = _sha256(path)
    if settings.model_sha256 and digest != settings.model_sha256:
        raise ValueError(f"Model checksum mismatch: expected {settings.model_sha256}, got {digest}")
    bundle = joblib.load(path)
    required = {"model", "features", "categories", "threshold", "history", "species_prior", "meta"}
    missing = required - set(bundle)
    if missing:
        raise ValueError(f"Model artifact is missing keys: {sorted(missing)}")
    history = {name: {"keys": h["keys"], "t": {tuple(r[:-2]): (r[-2], r[-1]) for r in h["rows"]}}
               for name, h in bundle["history"].items()}
    log.info("Loaded model %s (%s) from %s", bundle["meta"]["version"], bundle["meta"]["algorithm"], source)
    return LoadedModel(bundle=bundle, history=history, source=source, sha256=digest)


class ModelStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._current: LoadedModel | None = None
        self.error: str | None = None

    def load(self) -> None:
        try:
            m = _load()
            with self._lock:
                self._current = m
            self.error = None
        except Exception as e:  # keep serving the previous model if a reload fails
            self.error = f"{type(e).__name__}: {e}"
            log.exception("Model load failed")
            if self._current is None:
                raise

    @property
    def current(self) -> LoadedModel:
        if self._current is None:
            raise RuntimeError("Model not loaded")
        return self._current

    @property
    def ready(self) -> bool:
        return self._current is not None


store = ModelStore()
