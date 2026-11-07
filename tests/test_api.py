"""Run with:  pytest -q      (synthetic data, no internet needed)"""
import pytest
from fastapi.testclient import TestClient

from backend import engine, main

GOOD = dict(age=35, annual_income=1_200_000, employment_years=8, employment_type="Salaried", credit_score=760,
            existing_monthly_debt=8000, credit_utilization=0.2, past_delinquencies=0, loan_amount=400_000,
            loan_term_months=36, loan_purpose="Personal", home_ownership="Owned", dependents=1)
BAD = dict(GOOD, annual_income=420_000, credit_score=560, existing_monthly_debt=16000, credit_utilization=0.85,
           past_delinquencies=3, loan_amount=900_000, employment_years=1, employment_type="Business owner")


@pytest.fixture(scope="module")
def client():
    return TestClient(main.app)


def test_emi_formula():
    assert round(engine.emi(100_000, 0.12, 12), 2) == 8884.88


def test_model_quality(client):
    info = client.get("/api/model-info").json()
    lr = info["metrics"]["Logistic Regression (decision model)"]
    assert lr["roc_auc"] > 0.85 and lr["ks"] > 0.5
    rates = [g["actual_default_rate"] for g in info["grade_stats"] if g["applicants"] > 30]
    assert rates == sorted(rates)          # riskier grades default more often


def test_good_vs_bad_applicant(client):
    g, b = client.post("/api/assess", json=GOOD).json(), client.post("/api/assess", json=BAD).json()
    assert g["decision"] == "Approve" and g["grade"] in ("A", "B")
    assert b["decision"] == "Reject" and b["rule_failures"]
    assert g["probability_of_default"] < b["probability_of_default"]
    assert g["factors"] and g["advice"]


def test_monotonic_effects(client):
    base = client.post("/api/assess", json=GOOD).json()["probability_of_default"]
    worse_score = client.post("/api/assess", json=dict(GOOD, credit_score=580)).json()["probability_of_default"]
    bigger_loan = client.post("/api/assess", json=dict(GOOD, loan_amount=3_000_000)).json()["probability_of_default"]
    assert worse_score > base and bigger_loan > base


def test_validation(client):
    assert client.post("/api/assess", json=dict(GOOD, credit_score=1200)).status_code == 422
    assert client.post("/api/assess", json=dict(GOOD, employment_type="Pilot")).status_code == 422


def test_batch(client):
    j = client.get("/api/batch/sample", params={"n": 200}).json()
    assert j["n"] == 200 and sum(j["decision_counts"].values()) == 200
    df = engine.generate_applicants(60, seed=3)[engine.REQUIRED]
    up = client.post("/api/batch", files={"file": ("a.csv", df.to_csv(index=False))})
    assert up.status_code == 200 and up.json()["n"] == 60
    bad = client.post("/api/batch", files={"file": ("a.csv", "x,y\n1,2\n")})
    assert bad.status_code == 400
