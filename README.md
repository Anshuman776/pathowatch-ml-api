# PathoWatch ML Service

Trained models yahan chalte hain — website inhe HTTP API se use karti hai.
**Owner: ML team (aap).** Website team sirf `ML_SERVICE_URL` + `ML_SERVICE_API_KEY` use karti hai.

```
ML project (pathowatch/)                ML Service (ye folder)                 Website (Next.js)
 train_advanced.py ─► champion model    artifacts/model.joblib  ◄── load      src/lib/ml/client.js
 export_for_web.py ─► model.joblib ───► + model_card.json                     ──► POST /v1/...  (X-API-Key)
```

## Endpoints (`/docs` par try kar sakte ho)

| Endpoint | Kaam |
|---|---|
| `GET /health` | status + model version (no key) |
| `GET /v1/model-info` | version, algorithm, validation AUROC / calibration, checksum |
| `POST /v1/mdr/classify` | antibiotic results → MDR / not MDR (Magiorakos 2012 definition) |
| `POST /v1/risk/predict` | lab result se pehle MDR likelihood (adaptive LightGBM, site ki apni history) |
| `POST /v1/surveillance/analyze` | area-wise monthly counts → anomaly, hotspot (funnel), 3-month forecast, risk score, public level + message |
| `POST /v1/forecast` | koi bhi monthly series ka forecast (ensemble + 90% range) |
| `POST /v1/admin/reload` | naya model file load (header `X-Admin-Key`) — downtime nahi |

## Local run

```bash
python -m venv venv && venv\Scripts\activate        # Linux/Mac: source venv/bin/activate
pip install -r requirements-dev.txt
copy .env.example .env                               # Linux/Mac: cp
# PowerShell:  $env:ML_API_KEYS="web-dev-key"; $env:ML_ADMIN_KEY="admin-dev-key"
uvicorn app.main:app --port 8001
pytest -q                                            # 6 tests
```
`artifacts/model.joblib` pehle se hai. Naya banane ke liye ML project mein `python ml/export_for_web.py`.

## Model update karne ka tareeka (production)

1. ML project mein retrain: `python ml/train_advanced.py` (ya website ka Retrain flow).
2. Export: `python ml/export_for_web.py --service-dir ../pathowatch_ml_service/artifacts`
   → `model.joblib` + `model_card.json` (version, AUROC, **sha256**).
3. Host karo — do options:
   - **File:** server par `artifacts/model.joblib` replace karo (ya Docker volume).
   - **URL (recommended):** file ko GitHub Release / S3 / Google Cloud Storage par daalo, phir env:
     `MODEL_URL=https://.../model.joblib` aur `MODEL_SHA256=<model_card.json wala sha256>`
4. Activate: `curl -X POST https://<ml-service>/v1/admin/reload -H "X-Admin-Key: <ML_ADMIN_KEY>"`
   (ya service restart). Reload fail ho to purana model chalta rehta hai.
5. Check: `GET /v1/model-info` → naya version. Website ko kuch nahi karna.

Rollback: purana `model.joblib` (ya purana `MODEL_URL` + sha) wapas → reload.

## Deploy

**Docker**
```bash
docker build -t pathowatch-ml .
docker run -p 8001:8001 -v $(pwd)/artifacts:/srv/artifacts \
  -e ML_API_KEYS=<key-for-website> -e ML_ADMIN_KEY=<admin-key> pathowatch-ml
```
**Render / Railway:** "Docker" service, port 8001, env vars upar wale, health check `/health`.
Free tier par model `MODEL_URL` se download karna aasaan hai (disk persist nahi hoti).

## Security
- `ML_API_KEYS` set na ho to API **open** hai (`/health` mein `"auth": "DISABLED"` dikhega) — production mein hamesha set karo.
- Website server-side se call karti hai, isliye `ALLOWED_ORIGINS` khaali rakho (browser se direct call nahi).
- Model file sirf apne trusted source se load karo (joblib/pickle file code chala sakti hai). `MODEL_SHA256` isi liye hai.

## Honest notes (viva)
- MDR classify = standard rule, ML nahi — par yahi labels sab models ka base hain.
- Risk model AUROC ~0.71 (temporal test); naye locations par 0.70 (paper model 0.63). Population-level tool, treatment decision nahi.
- Monthly forecast = ensemble (mean-3, ARIMA, damped trend). Yearly ATLAS back-test mein ensemble sabse accha tha; LSTM akela nahi jeeta.
- Surveillance weights (0.40 hotspot / 0.30 rise / 0.15 anomaly / 0.15 outlook) research paper jaise hain.
