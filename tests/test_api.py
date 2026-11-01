"""Run: pytest -q   (needs artifacts/model.joblib)"""
import os

os.environ["ML_API_KEYS"] = "test-key"
os.environ["ML_ADMIN_KEY"] = "admin-key"

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

H = {"X-API-Key": "test-key"}


def client():
    return TestClient(app)


def test_health_and_auth():
    with client() as c:
        assert c.get("/health").json()["status"] == "ok"
        assert c.post("/v1/mdr/classify", json={"isolates": []}).status_code == 401
        assert c.get("/v1/model-info", headers=H).json()["algorithm"] == "LightGBM"


def test_mdr_definition():
    with client() as c:
        r = c.post("/v1/mdr/classify", headers=H, json={"isolates": [
            {"id": "a", "pathogen": "Klebsiella pneumoniae", "results": {"Meropenem": "R", "Ciprofloxacin": "R", "Ceftriaxone": "I"}},
            {"id": "b", "pathogen": "E. coli", "results": {"Amoxicillin": "R", "Ciprofloxacin": "S", "Ceftriaxone": "S", "Meropenem": "S"}},
            {"id": "c", "pathogen": "Klebsiella pneumoniae", "results": {"Amoxicillin": "R", "Meropenem": "R"}},
            {"id": "d", "pathogen": "Staphylococcus aureus", "results": {"Vancomycin": "S"}},
        ]}).json()["results"]
        by = {x["id"]: x for x in r}
        assert by["a"]["assessed"] and by["a"]["mdr"] is True
        assert by["b"]["assessed"] and by["b"]["mdr"] is False
        assert by["c"]["assessed"] is False            # amoxicillin not counted for K. pneumoniae -> 1 category
        assert by["d"]["assessed"] is False            # outside definition


def test_predict_local_vs_reference():
    with client() as c:
        base = {"species": "Klebsiella pneumoniae", "sample_type": "Blood", "ward": "Medicine ICU",
                "patient_type": "Inpatient", "age": 67, "gender": "M", "year": 2026}
        ref = c.post("/v1/risk/predict", headers=H, json=base).json()
        assert ref["supported"] and "national reference" in ref["history_source"]
        hi = c.post("/v1/risk/predict", headers=H, json=base | {"local_history": {
            "location": {"isolates": 400, "mdr": 320}, "location_species": {"isolates": 150, "mdr": 135},
            "location_species_source": {"isolates": 40, "mdr": 37}, "location_species_ward": {"isolates": 60, "mdr": 55},
            "location_species_last_12_months": {"isolates": 60, "mdr": 58}}}).json()
        lo = c.post("/v1/risk/predict", headers=H, json=base | {"local_history": {
            "location": {"isolates": 400, "mdr": 40}, "location_species": {"isolates": 150, "mdr": 12},
            "location_species_source": {"isolates": 40, "mdr": 3}, "location_species_ward": {"isolates": 60, "mdr": 5},
            "location_species_last_12_months": {"isolates": 60, "mdr": 4}}}).json()
        assert hi["history_source"].startswith("this site")
        assert hi["mdr_probability"] > lo["mdr_probability"] + 0.2


def test_surveillance_flags_outbreak():
    stable = [{"period": f"2026-{m:02d}", "isolates": 60, "mdr_cases": 12 + (m % 2)} for m in range(1, 9)]
    outbreak = stable[:-1] + [{"period": "2026-08", "isolates": 62, "mdr_cases": 34}]
    others = [{"area_id": f"x{i}", "series": stable} for i in range(5)]
    with client() as c:
        r = c.post("/v1/surveillance/analyze", headers=H, json={"areas": others + [
            {"area_id": "hot", "area_name": "Hot", "series": outbreak}]}).json()
        top = r["areas"][0]
        assert top["area_id"] == "hot" and top["anomaly"]["flag"] and top["risk"]["level"] == "High"
        assert top["public"]["level"] == "Requires Closer Monitoring"
        assert len(top["forecast"]["forecast"]) == 3
        assert all(a["risk"]["level"] == "Low" for a in r["areas"][1:])


def test_forecast_endpoint():
    with client() as c:
        r = c.post("/v1/forecast", headers=H, json={"values": [10, 12, 11, 13, 15, 14, 16], "horizon": 3}).json()
        assert len(r["forecast"]) == 3 and all(lo <= f <= hi for lo, f, hi in zip(r["lower"], r["forecast"], r["upper"]))


def test_reload_requires_admin():
    with client() as c:
        assert c.post("/v1/admin/reload", headers=H).status_code == 401
        r = c.post("/v1/admin/reload", headers={"X-Admin-Key": "admin-key"})
        assert r.status_code == 200 and r.json()["current_version"]
