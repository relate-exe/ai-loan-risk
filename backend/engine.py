"""Loan risk engine: synthetic applicants, models, risk grading, decisions, explanations."""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

EMPLOYMENT = ["Salaried", "Self-employed", "Business owner"]
HOME = ["Owned", "Rented", "Family-owned"]
PURPOSE = ["Personal", "Home improvement", "Education", "Vehicle", "Business", "Medical"]
REQUIRED = ["age", "annual_income", "employment_years", "employment_type", "credit_score",
            "existing_monthly_debt", "credit_utilization", "past_delinquencies", "loan_amount",
            "loan_term_months", "loan_purpose", "home_ownership", "dependents"]
NUMERIC = ["age", "annual_income", "employment_years", "credit_score", "existing_monthly_debt",
           "credit_utilization", "past_delinquencies", "loan_amount", "loan_term_months",
           "dependents", "dti", "loan_to_income"]
CAT_COLS = {"employment_type": EMPLOYMENT, "home_ownership": HOME, "loan_purpose": PURPOSE}
FEATURES = NUMERIC + [f"{c}={v}" for c, vals in CAT_COLS.items() for v in vals]

REF_RATE = 0.12          # flat reference rate used to compute debt-to-income inside the model
BASE_RATE = 0.095        # pricing: base rate plus a grade premium
GRADES = [(0.05, "A"), (0.10, "B"), (0.18, "C"), (0.30, "D"), (1.01, "E")]
PREMIUM = {"A": 0.0, "B": 0.015, "C": 0.03, "D": 0.05, "E": 0.07}
MAX_DTI = 0.65
MIN_SCORE = 500

LABELS = {
    "age": "Age", "annual_income": "Annual income", "employment_years": "Years in current work",
    "credit_score": "Credit score", "existing_monthly_debt": "Existing monthly EMIs",
    "credit_utilization": "Credit card utilisation", "past_delinquencies": "Past missed payments",
    "loan_amount": "Loan amount", "loan_term_months": "Loan term", "dependents": "Dependents",
    "dti": "Debt-to-income after this loan", "loan_to_income": "Loan size vs annual income",
    "employment_type": "Employment type", "home_ownership": "Home ownership", "loan_purpose": "Loan purpose",
}


def emi(principal, annual_rate, months):
    r = annual_rate / 12
    return principal * r * (1 + r) ** months / ((1 + r) ** months - 1)


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    monthly = d["annual_income"] / 12
    d["dti"] = (d["existing_monthly_debt"] + emi(d["loan_amount"], REF_RATE, d["loan_term_months"])) / monthly
    d["loan_to_income"] = d["loan_amount"] / d["annual_income"]
    return d


def design(df: pd.DataFrame) -> pd.DataFrame:
    d = add_derived(df)
    X = d[NUMERIC].astype(float).copy()
    for c, vals in CAT_COLS.items():
        for v in vals:
            X[f"{c}={v}"] = (d[c] == v).astype(float)
    return X[FEATURES]


def generate_applicants(n: int = 8000, seed: int = 42) -> pd.DataFrame:
    """Synthetic Indian retail-loan applicants (INR, CIBIL-style 300-900 scores) with a hidden risk process."""
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 66, n)
    emp_years = np.clip(rng.normal((age - 21) * 0.6, 3), 0, age - 18).round(1)
    income = np.clip(rng.lognormal(13.3, 0.6, n), 180_000, 12_000_000).round(-3)
    emp_type = rng.choice(EMPLOYMENT, n, p=[0.6, 0.2, 0.2])
    score = np.clip(rng.normal(690, 80, n) + 6 * np.minimum(emp_years, 10), 300, 900).astype(int)
    util = rng.beta(2, 3, n).round(2)
    delinq = np.clip(rng.poisson(0.25 + np.clip(700 - score, 0, None) / 250, n), 0, 6)
    debt = (income / 12 * rng.beta(1.5, 6, n)).round(-2)
    amount = np.clip(income * rng.uniform(0.2, 2.5, n), 50_000, 5_000_000).round(-3)
    term = rng.choice([12, 24, 36, 48, 60, 84], n)
    deps = np.clip(rng.poisson(1.2, n), 0, 5)
    df = pd.DataFrame({
        "age": age, "annual_income": income, "employment_years": emp_years, "employment_type": emp_type,
        "credit_score": score, "existing_monthly_debt": debt, "credit_utilization": util,
        "past_delinquencies": delinq, "loan_amount": amount, "loan_term_months": term,
        "loan_purpose": rng.choice(PURPOSE, n, p=[0.35, 0.15, 0.1, 0.15, 0.15, 0.1]),
        "home_ownership": rng.choice(HOME, n, p=[0.25, 0.45, 0.30]), "dependents": deps})
    d = add_derived(df)
    z = (-4.5 + 3.5 * (d["dti"] - 0.40) - 0.011 * (d["credit_score"] - 700) + 1.4 * (d["credit_utilization"] - 0.35)
         + 0.45 * d["past_delinquencies"] - 0.05 * np.minimum(d["employment_years"], 15)
         + 0.30 * (d["loan_to_income"] - 1.0) + 0.08 * d["dependents"]
         + 0.30 * (d["employment_type"] == "Business owner") + rng.normal(0, 0.5, n))
    df["default"] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return df


# ------------------------------------------------------------------ model
@dataclass
class Model:
    scaler: StandardScaler
    logit: LogisticRegression
    gb: GradientBoostingClassifier
    info: dict


def grade_for(pd_: float) -> str:
    return next(g for cut, g in GRADES if pd_ < cut)


def train(df: pd.DataFrame, seed: int = 42) -> Model:
    X, y = design(df), df["default"].to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    scaler = StandardScaler().fit(Xtr)
    logit = LogisticRegression(C=1.0, max_iter=2000).fit(scaler.transform(Xtr), ytr)
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                    subsample=0.8, random_state=seed).fit(Xtr, ytr)
    p_lr = logit.predict_proba(scaler.transform(Xte))[:, 1]
    p_gb = gb.predict_proba(Xte)[:, 1]
    cut = GRADES[3][0]  # reject at or above 30% (grade E)

    def metrics(p):
        fpr, tpr, _ = roc_curve(yte, p)
        pred = (p >= cut).astype(int)
        return {"roc_auc": roc_auc_score(yte, p), "pr_auc": average_precision_score(yte, p),
                "ks": float(np.max(tpr - fpr)), "brier": brier_score_loss(yte, p),
                "precision_at_cut": precision_score(yte, pred, zero_division=0),
                "recall_at_cut": recall_score(yte, pred, zero_division=0),
                "accuracy_at_cut": accuracy_score(yte, pred)}

    def roc(p):
        fpr, tpr, _ = roc_curve(yte, p)
        step = max(1, len(fpr) // 100)
        return [{"fpr": float(a), "tpr": float(b)} for a, b in zip(fpr[::step], tpr[::step])]

    test = pd.DataFrame({"p": p_lr, "y": yte})
    test["grade"] = test["p"].map(grade_for)
    gs = test.groupby("grade").agg(applicants=("y", "size"), actual_default_rate=("y", "mean"),
                                   avg_predicted_pd=("p", "mean")).reindex(list("ABCDE")).fillna(0)
    cm = confusion_matrix(yte, (p_lr >= cut).astype(int))

    # grouped logistic coefficients (per one standard deviation) for the global explanation
    coef = pd.Series(logit.coef_[0], index=FEATURES)
    grouped = {}
    for f in NUMERIC:
        grouped[LABELS[f]] = float(coef[f])
    for c, vals in CAT_COLS.items():
        grouped[LABELS[c]] = float(coef[[f"{c}={v}" for v in vals]].abs().max())
    info = {
        "n_total": len(df), "n_train": len(Xtr), "n_test": len(Xte), "default_rate": float(y.mean()),
        "metrics": {"Logistic Regression (decision model)": metrics(p_lr),
                    "Gradient Boosting (challenger)": metrics(p_gb)},
        "roc": {"Logistic Regression": roc(p_lr), "Gradient Boosting": roc(p_gb)},
        "coefficients": dict(sorted(grouped.items(), key=lambda kv: kv[1])),
        "gb_importances": dict(sorted(zip(FEATURES, gb.feature_importances_.tolist()), key=lambda kv: kv[1])[-12:]),
        "grade_stats": [{"grade": g, **{k: float(v) for k, v in r.items()}} for g, r in gs.iterrows()],
        "confusion": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
        "policy": {"grades": [{"grade": g, "pd_below": min(c, 1.0), "decision": d, "premium": PREMIUM[g]}
                              for (c, g), d in zip(GRADES, ["Approve", "Approve", "Approve with conditions",
                                                            "Manual review", "Reject"])],
                   "base_rate": BASE_RATE, "max_dti": MAX_DTI, "min_credit_score": MIN_SCORE},
    }
    return Model(scaler, logit, gb, _jsonable(info))


# ------------------------------------------------------------------ scoring
def predict_pd(m: Model, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = design(df)
    return m.logit.predict_proba(m.scaler.transform(X))[:, 1], m.gb.predict_proba(X)[:, 1]


def hard_rule_failures(a: dict, dti: float) -> list[str]:
    out = []
    if a["age"] < 21 or a["age"] > 65:
        out.append("Applicant age must be between 21 and 65")
    if a["credit_score"] < MIN_SCORE:
        out.append(f"Credit score is below the minimum of {MIN_SCORE}")
    if dti > MAX_DTI:
        out.append(f"Total EMIs would exceed {MAX_DTI:.0%} of monthly income")
    if a["past_delinquencies"] >= 5:
        out.append("Five or more past missed payments")
    return out


def decide(grade: str, failures: list[str]) -> str:
    if failures:
        return "Reject"
    return {"A": "Approve", "B": "Approve", "C": "Approve with conditions",
            "D": "Manual review", "E": "Reject"}[grade]


def explain(m: Model, a: dict, top: int = 6) -> list[dict]:
    df = pd.DataFrame([a])
    X = design(df)
    xs = m.scaler.transform(X)[0]
    contrib = pd.Series(m.logit.coef_[0] * xs, index=FEATURES)   # log-odds vs the average applicant
    d = add_derived(df).iloc[0]
    rows = []
    for f in NUMERIC:
        v = d[f]
        rows.append((LABELS[f], float(contrib[f]), v, f))
    for c, vals in CAT_COLS.items():
        rows.append((LABELS[c], float(contrib[[f"{c}={v}" for v in vals]].sum()), a[c], c))
    rows.sort(key=lambda r: abs(r[1]), reverse=True)
    out = []
    for label, c, v, f in rows[:top]:
        out.append({"feature": f, "label": label, "contribution": c,
                    "effect": "raises risk" if c > 0 else "lowers risk", "value": _fmt(f, v)})
    return out


def _fmt(f, v):
    if f in ("annual_income", "loan_amount", "existing_monthly_debt"):
        return f"₹{float(v):,.0f}"
    if f in ("dti", "credit_utilization"):
        return f"{float(v):.0%}"
    if f == "loan_to_income":
        return f"{float(v):.2f}x"
    if f == "loan_term_months":
        return f"{int(v)} months"
    if isinstance(v, (int, float, np.integer, np.floating)):
        return f"{float(v):g}"
    return str(v)


def max_approvable(m: Model, a: dict):
    """Largest loan amount (same term) that still lands in grade A-C with no hard-rule failure."""
    hi = float(min(5_000_000, max(a["annual_income"] * 3, 100_000)))
    grid = np.linspace(50_000, hi, 60).round(-3)
    df = pd.DataFrame([{**a, "loan_amount": float(x)} for x in grid])
    p, _ = predict_pd(m, df)
    dti = add_derived(df)["dti"].to_numpy()
    ok = [(g, x) for g, x, pp, dd in zip(grid, grid, p, dti)
          if grade_for(pp) in ("A", "B", "C") and not hard_rule_failures(a, dd)]
    return float(max(x for _, x in ok)) if ok else None


def assess(m: Model, a: dict) -> dict:
    df = pd.DataFrame([a])
    p_lr, p_gb = predict_pd(m, df)
    pd_, gb = float(p_lr[0]), float(p_gb[0])
    d = add_derived(df).iloc[0]
    grade = grade_for(pd_)
    failures = hard_rule_failures(a, float(d["dti"]))
    decision = decide(grade, failures)
    rate = BASE_RATE + PREMIUM[grade]
    q_emi = float(emi(a["loan_amount"], rate, a["loan_term_months"]))
    monthly = a["annual_income"] / 12
    dti_quoted = (a["existing_monthly_debt"] + q_emi) / monthly
    factors = explain(m, a)
    mx = max_approvable(m, a)

    advice = []
    if a["credit_utilization"] > 0.6:
        advice.append("Pay down credit card balances: utilisation above 60% raises risk.")
    if a["past_delinquencies"] > 0:
        advice.append("Avoid any new missed payments; a clean record for 12 months improves the profile.")
    if a["credit_score"] < 700:
        advice.append("Improving the credit score above 700 would lower the risk grade.")
    if d["dti"] > 0.40:
        advice.append("Reduce existing EMIs or choose a smaller or longer loan to bring debt-to-income below 40%.")
    if mx is not None and decision != "Approve" and mx < a["loan_amount"]:
        advice.append(f"A loan of up to ₹{mx:,.0f} over {a['loan_term_months']} months fits approval criteria.")
    if decision == "Reject" and mx is None:
        advice.append("No loan amount in the tested range meets approval criteria right now.")
    if not advice:
        advice.append("Profile looks healthy; no changes needed.")

    return _jsonable({
        "decision": decision, "grade": grade, "probability_of_default": pd_, "challenger_pd": gb,
        "health_score": round((1 - pd_) * 100), "rule_failures": failures,
        "quoted_rate": rate, "quoted_emi": q_emi, "dti_quoted": dti_quoted,
        "dti_reference": float(d["dti"]), "loan_to_income": float(d["loan_to_income"]),
        "max_approvable_amount": mx, "factors": factors, "advice": advice,
    })


def assess_batch(m: Model, df: pd.DataFrame) -> dict:
    p_lr, p_gb = predict_pd(m, df)
    d = add_derived(df)
    out = df.copy()
    out["pd"] = p_lr
    out["grade"] = [grade_for(x) for x in p_lr]
    out["dti"] = d["dti"].to_numpy()
    out["decision"] = [decide(g, hard_rule_failures(r, dd)) for g, (_, r), dd in
                       zip(out["grade"], out.iterrows(), out["dti"])]
    counts = out["decision"].value_counts().to_dict()
    approved = out[out["decision"].isin(["Approve", "Approve with conditions"])]
    show = out.head(200)
    return _jsonable({
        "n": len(out), "decision_counts": counts, "grade_counts": out["grade"].value_counts().reindex(list("ABCDE"), fill_value=0).to_dict(),
        "avg_pd": float(out["pd"].mean()), "requested_total": float(out["loan_amount"].sum()),
        "approved_total": float(approved["loan_amount"].sum()),
        "expected_loss_on_approved": float((approved["pd"] * approved["loan_amount"]).sum()),
        "rows": show[["age", "annual_income", "credit_score", "loan_amount", "loan_term_months", "dti", "pd", "grade", "decision"]].to_dict(orient="records"),
    })


def _jsonable(obj):
    def default(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return None if np.isnan(o) else float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        raise TypeError(type(o))
    return json.loads(json.dumps(obj, default=default))
