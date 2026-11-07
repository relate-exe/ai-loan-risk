"""FastAPI backend for the AI loan eligibility and risk assessment system.

Run:  uvicorn backend.main:app --reload --port 8000     (from the project root)
Docs: http://localhost:8000/docs
"""
import io
from typing import Literal

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import engine

app = FastAPI(title="AI Loan Risk API", version="1.0.0",
              description="Estimates probability of default, risk grade and loan decision (educational).")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
_state: dict = {}


def get_model() -> engine.Model:
    if "model" not in _state:
        _state["model"] = engine.train(engine.generate_applicants(8000))
    return _state["model"]


class Applicant(BaseModel):
    age: int = Field(..., ge=18, le=100)
    annual_income: float = Field(..., ge=60_000, le=100_000_000)
    employment_years: float = Field(..., ge=0, le=60)
    employment_type: Literal["Salaried", "Self-employed", "Business owner"]
    credit_score: int = Field(..., ge=300, le=900)
    existing_monthly_debt: float = Field(..., ge=0)
    credit_utilization: float = Field(..., ge=0, le=1)
    past_delinquencies: int = Field(..., ge=0, le=20)
    loan_amount: float = Field(..., ge=10_000, le=50_000_000)
    loan_term_months: int = Field(..., ge=6, le=120)
    loan_purpose: Literal["Personal", "Home improvement", "Education", "Vehicle", "Business", "Medical"]
    home_ownership: Literal["Owned", "Rented", "Family-owned"]
    dependents: int = Field(..., ge=0, le=10)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/model-info")
def model_info():
    """Training data summary, metrics, ROC curves, coefficients and the lending policy."""
    return get_model().info


@app.post("/api/assess")
def assess(a: Applicant):
    """Assess one applicant: decision, risk grade, probability of default, price and explanation."""
    return engine.assess(get_model(), a.model_dump())


def _validate_csv(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in engine.REQUIRED if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {', '.join(missing)}")
    df = df.dropna(subset=engine.REQUIRED)
    if not 1 <= len(df) <= 20000:
        raise HTTPException(status_code=400, detail="Provide between 1 and 20,000 valid rows.")
    for col, allowed in engine.CAT_COLS.items():
        bad = sorted(set(df[col]) - set(allowed))
        if bad:
            raise HTTPException(status_code=400, detail=f"Invalid {col} values {bad}. Allowed: {allowed}")
    try:
        for c in engine.REQUIRED:
            if c not in engine.CAT_COLS:
                df[c] = pd.to_numeric(df[c])
    except Exception:
        raise HTTPException(status_code=400, detail="All non-category columns must be numeric.")
    if (df["annual_income"] <= 0).any() or (df["loan_term_months"] <= 0).any():
        raise HTTPException(status_code=400, detail="annual_income and loan_term_months must be positive.")
    return df.reset_index(drop=True)


@app.post("/api/batch")
async def batch(file: UploadFile = File(...)):
    """Score a CSV of applicants (a portfolio view)."""
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (20 MB max).")
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the file as CSV.")
    return engine.assess_batch(get_model(), _validate_csv(df))


@app.get("/api/batch/sample")
def batch_sample(n: int = 500, seed: int = 7):
    """Score a freshly generated synthetic portfolio."""
    n = max(50, min(n, 5000))
    df = engine.generate_applicants(n, seed=seed)[engine.REQUIRED]
    return engine.assess_batch(get_model(), df)


@app.get("/api/meta")
def meta():
    return {"required_columns": engine.REQUIRED, "employment_types": engine.EMPLOYMENT,
            "home_ownership": engine.HOME, "loan_purposes": engine.PURPOSE}
