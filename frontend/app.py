"""Streamlit frontend for the AI loan eligibility and risk assessment system. Calls the FastAPI backend only.

Run:  streamlit run frontend/app.py        (backend must be running)
"""
import os
import re

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
GREEN, RED, AMBER, INK, GREY = "#0F6B5C", "#B3392F", "#E0A100", "#16211F", "#9AA6A3"
GRADE_COLOR = {"A": "#0F6B5C", "B": "#4C9A6A", "C": "#E0A100", "D": "#D9772B", "E": "#B3392F"}

st.set_page_config(page_title="AI Loan Risk", page_icon="🏦", layout="wide")


def inr(x):
    x = int(round(x))
    s = str(abs(x))
    if len(s) > 3:
        s = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", s[:-3]) + "," + s[-3:]
    return ("-" if x < 0 else "") + "₹" + s


class ApiError(Exception):
    pass


def _json(r):
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise ApiError(str(detail))
    return r.json()


def _call(fn, *a, **k):
    try:
        return fn(*a, **k)
    except requests.ConnectionError:
        raise ApiError(f"Cannot reach the backend at {API_URL}. Start it with: uvicorn backend.main:app --port 8000")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_info():
    return _json(_call(requests.get, f"{API_URL}/api/model-info", timeout=300))


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_meta():
    return _json(_call(requests.get, f"{API_URL}/api/meta", timeout=30))


@st.cache_data(show_spinner=False)
def fetch_sample_batch(n):
    return _json(_call(requests.get, f"{API_URL}/api/batch/sample", params={"n": n}, timeout=300))


@st.cache_data(show_spinner=False)
def fetch_batch(name, content):
    return _json(_call(requests.post, f"{API_URL}/api/batch", files={"file": (name, content, "text/csv")}, timeout=300))


def post_assess(payload):
    return _json(_call(requests.post, f"{API_URL}/api/assess", json=payload, timeout=60))


try:
    with st.spinner("Starting up (the backend trains the models on first use)..."):
        info, meta = fetch_info(), fetch_meta()
except ApiError as e:
    st.error(str(e))
    st.stop()

st.title("AI Loan Eligibility & Risk Assessment")
st.caption("Estimates probability of default, assigns a risk grade, and explains the decision.")
tab_apply, tab_perf, tab_port, tab_how = st.tabs(["Assess an applicant", "Model performance", "Portfolio", "How it works"])

# ------------------------------------------------------------------ assess
with tab_apply:
    with st.form("applicant"):
        a, b, c = st.columns(3)
        age = a.number_input("Age", 18, 80, 32)
        income = a.number_input("Annual income (₹)", 60_000, 50_000_000, 900_000, 50_000)
        emp_years = a.number_input("Years in current work", 0.0, 50.0, 6.0, 0.5)
        emp_type = a.selectbox("Employment type", meta["employment_types"])
        score = b.slider("Credit score (300 to 900)", 300, 900, 720)
        debt = b.number_input("Existing monthly EMIs (₹)", 0, 2_000_000, 12_000, 1000)
        util = b.slider("Credit card utilisation", 0.0, 1.0, 0.30, 0.05, format="%.2f")
        delinq = b.number_input("Past missed payments", 0, 20, 0)
        amount = c.number_input("Loan amount (₹)", 10_000, 50_000_000, 500_000, 25_000)
        term = c.select_slider("Loan term (months)", [12, 24, 36, 48, 60, 84, 120], value=36)
        purpose = c.selectbox("Loan purpose", meta["loan_purposes"])
        home = c.selectbox("Home ownership", meta["home_ownership"], index=1)
        deps = c.number_input("Dependents", 0, 10, 1)
        submit = st.form_submit_button("Assess applicant", type="primary")

    if submit:
        payload = {"age": int(age), "annual_income": float(income), "employment_years": float(emp_years),
                   "employment_type": emp_type, "credit_score": int(score), "existing_monthly_debt": float(debt),
                   "credit_utilization": float(util), "past_delinquencies": int(delinq),
                   "loan_amount": float(amount), "loan_term_months": int(term), "loan_purpose": purpose,
                   "home_ownership": home, "dependents": int(deps)}
        try:
            r = post_assess(payload)
        except ApiError as e:
            st.error(str(e))
            st.stop()

        box = {"Approve": st.success, "Approve with conditions": st.warning,
               "Manual review": st.warning, "Reject": st.error}[r["decision"]]
        box(f"**Decision: {r['decision']}**  |  Risk grade {r['grade']}")
        for f in r["rule_failures"]:
            st.write(f"- Policy rule failed: {f}")

        k = st.columns(4)
        k[0].metric("Probability of default", f"{r['probability_of_default']*100:.1f}%")
        k[1].metric("Risk grade", r["grade"])
        k[2].metric(f"Quoted EMI at {r['quoted_rate']*100:.1f}%", inr(r["quoted_emi"]))
        k[3].metric("Debt-to-income after loan", f"{r['dti_quoted']*100:.0f}%")

        left, right = st.columns(2)
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=r["probability_of_default"] * 100, number={"suffix": "%"},
            title={"text": "Probability of default"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": INK},
                   "steps": [{"range": [0, 5], "color": "#CFE8E1"}, {"range": [5, 10], "color": "#E1EFD9"},
                             {"range": [10, 18], "color": "#F6E3A8"}, {"range": [18, 30], "color": "#F2CFA6"},
                             {"range": [30, 100], "color": "#EBC0BB"}]}))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=0))
        left.plotly_chart(fig, width="stretch")

        fx = pd.DataFrame(r["factors"]).iloc[::-1]
        fig = go.Figure(go.Bar(x=fx["contribution"], y=fx["label"] + " (" + fx["value"] + ")", orientation="h",
                               marker_color=[RED if v > 0 else GREEN for v in fx["contribution"]]))
        fig.update_layout(title="What drove this result (red raises risk, green lowers it)", height=300,
                          template="plotly_white", margin=dict(l=0, r=0, t=50, b=0),
                          xaxis_title="Effect on default risk vs an average applicant")
        right.plotly_chart(fig, width="stretch")

        mx = r["max_approvable_amount"]
        st.write("**Maximum loan that would be approved (same term):** " + (inr(mx) if mx else "none in the tested range"))
        st.write("**Suggestions**")
        for line in r["advice"]:
            st.write(f"- {line}")
        st.caption(f"Challenger model (Gradient Boosting) estimates {r['challenger_pd']*100:.1f}%. "
                   "Decisions use the logistic model because its reasoning can be explained.")

# ------------------------------------------------------------------ performance
with tab_perf:
    st.write(f"Trained on **{info['n_train']:,}** synthetic applicants and tested on **{info['n_test']:,}** unseen ones. "
             f"Overall default rate in the data: **{info['default_rate']*100:.1f}%**.")
    mdf = pd.DataFrame(info["metrics"]).T
    mdf = mdf[["roc_auc", "pr_auc", "ks", "brier", "precision_at_cut", "recall_at_cut", "accuracy_at_cut"]]
    mdf.columns = ["ROC-AUC", "PR-AUC", "KS", "Brier (lower is better)", "Precision at reject cut-off",
                   "Recall at reject cut-off", "Accuracy at reject cut-off"]
    st.dataframe(mdf.style.format("{:.3f}"), width="stretch")
    st.caption("ROC-AUC and KS measure how well the model separates good from bad borrowers. "
               "Brier measures how accurate the probabilities are.")

    l, r_ = st.columns(2)
    fig = go.Figure()
    for name, col in (("Logistic Regression", GREEN), ("Gradient Boosting", AMBER)):
        pts = pd.DataFrame(info["roc"][name])
        fig.add_trace(go.Scatter(x=pts["fpr"], y=pts["tpr"], name=name, line=dict(color=col, width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random guess", line=dict(color=GREY, dash="dot")))
    fig.update_layout(title="ROC curve (unseen data)", xaxis_title="False positive rate",
                      yaxis_title="True positive rate", height=360, template="plotly_white",
                      margin=dict(l=0, r=0, t=40, b=0), legend=dict(orientation="h", y=-0.25))
    l.plotly_chart(fig, width="stretch")

    gs = pd.DataFrame(info["grade_stats"])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=gs["grade"], y=gs["avg_predicted_pd"] * 100, name="Predicted default rate",
                         marker_color=GREY))
    fig.add_trace(go.Bar(x=gs["grade"], y=gs["actual_default_rate"] * 100, name="Actual default rate",
                         marker_color=GREEN))
    fig.update_layout(title="Does each risk grade behave as predicted? (%)", barmode="group", height=360,
                      template="plotly_white", margin=dict(l=0, r=0, t=40, b=0), legend=dict(orientation="h", y=-0.25))
    r_.plotly_chart(fig, width="stretch")
    st.caption("A well-calibrated model has predicted and actual rates close together in every grade.")

    co = pd.Series(info["coefficients"])
    fig = go.Figure(go.Bar(x=co.values, y=co.index, orientation="h",
                           marker_color=[RED if v > 0 else GREEN for v in co.values]))
    fig.update_layout(title="Logistic model: effect of each factor (per 1 standard deviation; red raises risk)",
                      height=420, template="plotly_white", margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, width="stretch")

    st.write("**Lending policy used**")
    pol = pd.DataFrame(info["policy"]["grades"])
    pol["pd_below"] = pol["pd_below"].map(lambda x: f"{x*100:.0f}%" if x < 1 else "above 30%")
    pol["premium"] = pol["premium"].map(lambda x: f"+{x*100:.1f}%")
    pol.columns = ["Grade", "Default probability below", "Decision", "Rate premium"]
    st.dataframe(pol, hide_index=True, width="stretch")
    st.caption(f"Base rate {info['policy']['base_rate']*100:.1f}%. Hard rules: reject if total EMIs exceed "
               f"{info['policy']['max_dti']:.0%} of income or credit score is below {info['policy']['min_credit_score']}.")

# ------------------------------------------------------------------ portfolio
with tab_port:
    st.write("Score many applicants at once.")
    c1, c2 = st.columns([1, 1])
    n_sample = c1.select_slider("Sample portfolio size", [100, 250, 500, 1000, 2500], value=500)
    run_sample = c1.button("Run on a sample portfolio")
    up = c2.file_uploader("...or upload your own CSV", type="csv")
    template = pd.DataFrame([[32, 900000, 6, "Salaried", 720, 12000, 0.3, 0, 500000, 36, "Personal", "Rented", 1]],
                            columns=meta["required_columns"])
    c2.download_button("Download CSV template", template.to_csv(index=False), "applicants_template.csv", "text/csv")

    res = None
    try:
        if up is not None:
            res = fetch_batch(up.name, up.getvalue())
        elif run_sample:
            res = fetch_sample_batch(n_sample)
    except ApiError as e:
        st.error(str(e))

    if res:
        k = st.columns(4)
        k[0].metric("Applicants", f"{res['n']:,}")
        k[1].metric("Average default probability", f"{res['avg_pd']*100:.1f}%")
        k[2].metric("Requested", inr(res["requested_total"]))
        k[3].metric("Approved (incl. conditions)", inr(res["approved_total"]),
                    f"expected loss {inr(res['expected_loss_on_approved'])}", delta_color="off")
        l, r_ = st.columns(2)
        dc = pd.Series(res["decision_counts"])
        colors = {"Approve": GREEN, "Approve with conditions": AMBER, "Manual review": "#D9772B", "Reject": RED}
        fig = go.Figure(go.Bar(x=dc.index, y=dc.values, marker_color=[colors[i] for i in dc.index]))
        fig.update_layout(title="Decisions", height=320, template="plotly_white", margin=dict(l=0, r=0, t=40, b=0))
        l.plotly_chart(fig, width="stretch")
        gc = pd.Series(res["grade_counts"])
        fig = go.Figure(go.Bar(x=gc.index, y=gc.values, marker_color=[GRADE_COLOR[g] for g in gc.index]))
        fig.update_layout(title="Applicants by risk grade", height=320, template="plotly_white",
                          margin=dict(l=0, r=0, t=40, b=0))
        r_.plotly_chart(fig, width="stretch")
        rows = pd.DataFrame(res["rows"])
        rows.columns = ["Age", "Annual income", "Credit score", "Loan amount", "Term (months)",
                        "Debt-to-income", "Default probability", "Grade", "Decision"]
        st.dataframe(rows.style.format({"Annual income": "₹{:,.0f}", "Loan amount": "₹{:,.0f}",
                                        "Debt-to-income": "{:.0%}", "Default probability": "{:.1%}"}),
                     hide_index=True, width="stretch")
        st.caption("First 200 applicants shown.")

# ------------------------------------------------------------------ how it works
with tab_how:
    st.markdown("""
**Inputs:** income, credit score, existing EMIs, credit utilisation, past missed payments, employment, loan size and term.
From these the system derives debt-to-income after the new loan and loan size relative to income.

**Two models**
- **Logistic Regression (decision model):** gives a probability of default and a clear, explainable reason for each factor.
  Lenders favour this kind of model because its decisions can be justified.
- **Gradient Boosting (challenger):** a more flexible model used to cross-check the result.

**From probability to decision:** default probability maps to a risk grade (A to E). Grades A and B are approved,
C is approved with conditions at a higher rate, D goes to manual review, E is rejected. Hard policy rules
(very high debt burden, very low credit score) override the model.

**Explanations:** each assessment shows which factors pushed risk up or down, a suggested loan amount that would be
approved, and plain-English advice to improve the application.

**Testing:** models are scored on applicants they never saw, using ROC-AUC, KS, and a grade-by-grade check that predicted
and actual default rates agree.

**Fairness and limits:** gender, religion, caste and similar attributes are not used. Data is synthetic, so real-world results
will differ. A real lender needs bureau data, regulatory compliance and ongoing monitoring.
""")
