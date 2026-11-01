"""Request bodies. Every endpoint is documented automatically at /docs."""
from pydantic import BaseModel, Field


class IsolateResults(BaseModel):
    id: str = Field(description="Your isolate/sample id; returned unchanged")
    pathogen: str = Field(examples=["Klebsiella pneumoniae"])
    results: dict[str, str] = Field(description="antibiotic -> S / I / R",
                                    examples=[{"Meropenem": "R", "Ciprofloxacin": "R", "Ceftriaxone": "R"}])


class ClassifyRequest(BaseModel):
    isolates: list[IsolateResults] = Field(max_length=20000)


class Counts(BaseModel):
    isolates: int = 0
    mdr: int = 0


class LocalHistory(BaseModel):
    """Counts the website computes from its own database for the LAST 36 MONTHS (and last 12 for the trend)."""
    location: Counts | None = None
    location_species: Counts | None = None
    location_species_source: Counts | None = None
    location_species_ward: Counts | None = None
    location_species_last_12_months: Counts | None = None


class PredictRequest(BaseModel):
    species: str = Field(examples=["Klebsiella pneumoniae"])
    sample_type: str | None = Field(None, examples=["Blood"])
    ward: str | None = Field(None, examples=["Medicine ICU"])
    patient_type: str | None = Field(None, examples=["Inpatient"])
    age: float | None = Field(None, ge=0, le=120, examples=[67])
    age_group: str | None = None
    gender: str | None = Field(None, examples=["Male"])
    year: int | None = Field(None, ge=2000, le=2100, examples=[2026])
    local_history: LocalHistory | None = None


class SeriesPoint(BaseModel):
    period: str = Field(pattern=r"^\d{4}-\d{2}$", examples=["2026-09"])
    isolates: int = Field(ge=0)
    mdr_cases: int = Field(ge=0)


class AreaSeries(BaseModel):
    area_id: str
    area_name: str | None = None
    series: list[SeriesPoint] = Field(min_length=1, max_length=240)


class AnalyzeRequest(BaseModel):
    areas: list[AreaSeries] = Field(min_length=1, max_length=500)
    horizon: int = Field(3, ge=1, le=12)


class ForecastRequest(BaseModel):
    values: list[float] = Field(min_length=3, max_length=500)
    horizon: int = Field(3, ge=1, le=24)
    upper_bound: float | None = Field(None, description="e.g. 100 for percentages")
