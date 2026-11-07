# AI Loan Eligibility & Risk Assessment System

AI for Finance CIA-2, Option 4: an application that analyses an applicant's income, credit history and
existing debts to assess loan eligibility and estimate credit risk.

> Educational project using synthetic data. Not for real lending decisions.

## Architecture

```
Streamlit frontend  --HTTP/JSON-->  FastAPI backend  -->  engine.py (models, grading, decisions)
frontend/app.py                     backend/main.py       backend/engine.py
```

- **Backend** (`backend/`): REST API. Generates training data, trains the models on first use, assesses
  single applicants and portfolios, explains each decision.
- **Frontend** (`frontend/`): Streamlit app. Calls the API only.
- **Tests** (`tests/`): API and engine tests (no internet needed).

## How AI is used

| Part | Technique |
|---|---|
| Probability of default | Logistic Regression (decision model, explainable) with Gradient Boosting as a challenger |
| Features | Income, credit score, existing EMIs, utilisation, missed payments, employment, loan size and term, plus derived debt-to-income and loan-to-income |
| Risk grade | Default probability mapped to grades A to E |
| Decision | Grades A/B approve, C approve with conditions, D manual review, E reject; hard policy rules override the model |
| Pricing | Base rate plus a premium by grade; EMI computed at the quoted rate |
| Explanation | Per-applicant factor contributions, a maximum approvable loan amount, and improvement advice |
| Evaluation | Held-out test set; ROC-AUC, KS, Brier score, and predicted-vs-actual default rate by grade |

## Run

Requires Python 3.10+.

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python run.py                     # backend on :8000, frontend on :8501
```

Open http://localhost:8501. API docs: http://localhost:8000/docs. The first request takes a few seconds while the models train.

Or in two terminals: `uvicorn backend.main:app --port 8000` and `streamlit run frontend/app.py`.

## Portfolio CSV

The Portfolio tab accepts a CSV with these columns (a template can be downloaded in the app):
`age, annual_income, employment_years, employment_type, credit_score, existing_monthly_debt, credit_utilization,
past_delinquencies, loan_amount, loan_term_months, loan_purpose, home_ownership, dependents`.

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/api/model-info` | Metrics, ROC curves, coefficients, lending policy |
| POST | `/api/assess` | Assess one applicant |
| POST | `/api/batch` | Score an uploaded CSV |
| GET | `/api/batch/sample` | Score a generated sample portfolio |
| GET | `/api/meta` | Required columns and allowed category values |

## Test

```bash
pytest -q
```

## Limitations

Data is synthetic, so real-world performance will differ. Protected attributes (gender, religion, caste) are not used,
but a real lender would still need fairness testing, bureau data, regulatory compliance (including RBI guidelines),
and ongoing monitoring.
