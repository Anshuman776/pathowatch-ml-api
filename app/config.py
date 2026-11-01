"""Service settings, read once from environment variables (see .env.example)."""
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _list(name: str) -> list[str]:
    return [v.strip() for v in os.getenv(name, "").split(",") if v.strip()]


@dataclass(frozen=True)
class Settings:
    # Model artifact: a local path, or a URL that is downloaded once at startup into MODEL_CACHE_DIR
    model_path: Path = Path(os.getenv("MODEL_PATH", ROOT / "artifacts" / "model.joblib"))
    model_url: str = os.getenv("MODEL_URL", "")
    model_sha256: str = os.getenv("MODEL_SHA256", "").lower()
    model_cache_dir: Path = Path(os.getenv("MODEL_CACHE_DIR", ROOT / ".model_cache"))

    # Clients (the website) must send X-API-Key with one of these values
    api_keys: list[str] = field(default_factory=lambda: _list("ML_API_KEYS"))
    # Separate key for operational endpoints such as /v1/admin/reload
    admin_key: str = os.getenv("ML_ADMIN_KEY", "")

    # Location used for history when a site has too little local data
    reference_country: str = os.getenv("REFERENCE_COUNTRY", "India")
    min_local_isolates: int = int(os.getenv("MIN_LOCAL_ISOLATES", "20"))

    allowed_origins: list[str] = field(default_factory=lambda: _list("ALLOWED_ORIGINS"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
