"""Interactive dashboard for the hospital readmission project.

Run it from the PyCharm terminal with:
    streamlit run dashboard.py
It opens in your browser at http://localhost:8501
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import data_cleaning
from model import CATEGORICAL, NUMERIC, TARGET, build_preprocessor

st.set_page_config(page_title="Hospital Readmission Dashboard", page_icon="🏥", layout="wide")

BLUE, RED, GREY = "#3a7ca5", "#c0392b", "#8a8a8a"


# --------------------------------------------------------------------------- #
# Data and model (cached so they only run once, not on every click)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Cleaning the raw data…")
def load_data():
    return data_cleaning.clean(verbose=False, return_report=True)


@st.cache_resource(show_spinner="Training the risk model…")
def train_model(df: pd.DataFrame):
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ]).fit(X_train, y_train)
    test_scores = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, test_scores)
    # Decile boundaries + actual readmission rate per decile, to place a new patient
    deciles, edges = pd.qcut(test_scores, 10, labels=range(1, 11), retbins=True)
    decile_rates = y_test.groupby(deciles, observed=True).mean() * 100
    return pipe, auc, edges, decile_rates


def rate_by(df: pd.DataFrame, column: str, min_count: int = 50) -> pd.DataFrame:
    t = (df.groupby(column, observed=True)[TARGET]
         .agg(rate="mean", patients="size").reset_index())
    t["rate"] = (t.rate * 100).round(1)
    return t[t.patients >= min_count]


def bar_chart(t: pd.DataFrame, column: str, title: str, overall: float, sort: bool = True):
    if sort:
        t = t.sort_values("rate")
    t = t.assign(colour=["Above average" if r > overall else "At or below average" for r in t.rate])
    fig = px.bar(t, x="rate", y=column, orientation="h", color="colour", title=title,
                 color_discrete_map={"Above average": RED, "At or below average": BLUE},
                 hover_data={"patients": ":,", "rate": ":.1f", "colour": False},
                 labels={"rate": "30-day readmission rate (%)", column: ""})
    fig.add_vline(x=overall, line_dash="dash", line_color=GREY,
                  annotation_text=f"Average {overall:.1f}%", annotation_position="top")
    fig.update_layout(legend_title_text="", height=120 + 38 * len(t), margin=dict(t=60, b=10),
                      yaxis={"categoryorder": "array", "categoryarray": t[column].tolist()})
    return fig


df, report = load_data()
model, model_auc, decile_edges, decile_rates = train_model(df)
df["age_band"] = df.age_years.map(lambda a: f"{a - 5}-{a + 5}")
AGE_BANDS = [f"{a}-{a + 10}" for a in range(0, 100, 10)]

# --------------------------------------------------------------------------- #
# Sidebar filters
# --------------------------------------------------------------------------- #
st.sidebar.header("Filters")
st.sidebar.caption("Filters apply to the Overview and Patient groups tabs.")
age_range = st.sidebar.select_slider("Age band", options=AGE_BANDS, value=(AGE_BANDS[0], AGE_BANDS[-1]))
genders = st.sidebar.multiselect("Gender", sorted(df.gender.unique()), default=sorted(df.gender.unique()))
adm_types = st.sidebar.multiselect("Admission type", sorted(df.admission_type.unique()),
                                   default=sorted(df.admission_type.unique()))
diagnoses = st.sidebar.multiselect("Primary diagnosis", sorted(df.diag_1_group.unique()),
                                   default=sorted(df.diag_1_group.unique()))

lo, hi = AGE_BANDS.index(age_range[0]), AGE_BANDS.index(age_range[1])
filtered = df[
    df.age_band.isin(AGE_BANDS[lo:hi + 1])
    & df.gender.isin(genders)
    & df.admission_type.isin(adm_types)
    & df.diag_1_group.isin(diagnoses)
]

# --------------------------------------------------------------------------- #
# Header + KPIs
# --------------------------------------------------------------------------- #
st.title("🏥 Hospital Readmission Dashboard")
st.caption("Diabetes 130-US Hospitals dataset (UCI, 1999–2008) · one row per patient after cleaning")

if filtered.empty:
    st.warning("No patients match these filters. Widen the filters in the sidebar.")
    st.stop()

overall_all = df[TARGET].mean() * 100
overall = filtered[TARGET].mean() * 100
k1, k2, k3, k4 = st.columns(4)
k1.metric("Patients", f"{len(filtered):,}",
          help="Patients matching the current filters")
k2.metric("30-day readmission rate", f"{overall:.1f}%",
          delta=f"{overall - overall_all:+.1f} pts vs all patients" if len(filtered) < len(df) else None,
          delta_color="inverse")
k3.metric("Avg length of stay", f"{filtered.length_of_stay.mean():.1f} days")
k4.metric("Avg medications", f"{filtered.num_medications.mean():.1f}")

tab_overview, tab_groups, tab_risk, tab_cleaning = st.tabs(
    ["📊 Overview", "👥 Patient groups", "🩺 Risk calculator", "🧹 Data cleaning"])

# --------------------------------------------------------------------------- #
# Tab 1: Overview
# --------------------------------------------------------------------------- #
with tab_overview:
    c1, c2 = st.columns(2)
    with c1:
        prior = filtered.assign(prior_stays=filtered.number_inpatient.clip(upper=4).map(
            lambda n: "4+" if n == 4 else str(n)))
        t = rate_by(prior, "prior_stays", min_count=1)
        fig = px.line(t, x="prior_stays", y="rate", markers=True, text="rate",
                      title="Readmission by inpatient stays in the previous year",
                      hover_data={"patients": ":,"},
                      labels={"prior_stays": "Inpatient stays in previous year", "rate": "Readmission rate (%)"})
        fig.update_traces(line_color=RED, texttemplate="%{text:.1f}%", textposition="top center")
        fig.update_yaxes(range=[0, t.rate.max() * 1.25])
        # Treat "0".."4+" as labels, not numbers (otherwise plotly drops "4+")
        fig.update_xaxes(type="category", categoryorder="array", categoryarray=["0", "1", "2", "3", "4+"])
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.plotly_chart(bar_chart(rate_by(filtered, "discharge_to"), "discharge_to",
                                  "Readmission by discharge destination", overall), width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        t = rate_by(filtered, "age_band", min_count=20)
        t["age_band"] = pd.Categorical(t.age_band, categories=AGE_BANDS, ordered=True)
        fig = px.bar(t.sort_values("age_band"), x="age_band", y="rate", title="Readmission by age",
                     hover_data={"patients": ":,"}, labels={"age_band": "Age", "rate": "Readmission rate (%)"})
        fig.update_traces(marker_color=BLUE)
        fig.add_hline(y=overall, line_dash="dash", line_color=GREY)
        st.plotly_chart(fig, width="stretch")
    with c4:
        t = rate_by(filtered, "length_of_stay", min_count=20)
        fig = px.bar(t, x="length_of_stay", y="rate", title="Readmission by length of stay",
                     hover_data={"patients": ":,"},
                     labels={"length_of_stay": "Days in hospital", "rate": "Readmission rate (%)"})
        fig.update_traces(marker_color=BLUE)
        fig.add_hline(y=overall, line_dash="dash", line_color=GREY)
        st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------------------- #
# Tab 2: Patient groups (explore any column)
# --------------------------------------------------------------------------- #
with tab_groups:
    options = {
        "Primary diagnosis": "diag_1_group", "Admission type": "admission_type",
        "Admission source": "admission_source", "Discharge destination": "discharge_to",
        "Medical specialty": "medical_specialty", "HbA1c result": "a1c_result",
        "Glucose test result": "glucose_result", "Race": "race", "Gender": "gender",
        "On insulin": "on_insulin", "Medication changed": "med_changed",
    }
    choice = st.selectbox("Compare readmission rates by", list(options))
    col = options[choice]
    t = rate_by(filtered, col).astype({col: str})
    left, right = st.columns([3, 2])
    left.plotly_chart(bar_chart(t, col, f"Readmission by {choice.lower()}", overall), width="stretch")
    right.dataframe(t.sort_values("rate", ascending=False).rename(
        columns={col: choice, "rate": "Readmission rate (%)", "patients": "Patients"}),
        hide_index=True, width="stretch")
    right.caption("Groups with fewer than 50 patients are hidden.")

# --------------------------------------------------------------------------- #
# Tab 3: Risk calculator
# --------------------------------------------------------------------------- #
with tab_risk:
    st.subheader("Estimate a patient's 30-day readmission risk")
    st.caption(f"Logistic Regression trained on 75% of patients · ROC-AUC on the other 25%: {model_auc:.3f}. "
               "For learning purposes only, not for clinical use.")

    def mode(col):
        return df[col].mode()[0]

    with st.form("patient"):
        a, b, c = st.columns(3)
        age = a.select_slider("Age band", AGE_BANDS, value="70-80")
        gender = a.radio("Gender", ["Female", "Male"], horizontal=True)
        diag = a.selectbox("Primary diagnosis", sorted(df.diag_1_group.unique()),
                           index=sorted(df.diag_1_group.unique()).index("Circulatory"))
        adm_type = b.selectbox("Admission type", sorted(df.admission_type.unique()),
                               index=sorted(df.admission_type.unique()).index("Emergency"))
        discharge = b.selectbox("Discharged to", sorted(df.discharge_to.unique()),
                                index=sorted(df.discharge_to.unique()).index("Home"))
        a1c = b.selectbox("HbA1c result", ["Not tested", "Norm", ">7", ">8"])
        stay = c.slider("Days in hospital", 1, 14, 4)
        inpatient = c.slider("Inpatient stays in previous year", 0, 10, 0)
        emergency = c.slider("Emergency visits in previous year", 0, 10, 0)
        meds = c.slider("Number of medications", 1, 60, 15)
        submitted = st.form_submit_button("Calculate risk", type="primary")

    # Start from a "typical" patient, then overwrite with the form values
    patient = {col: df[col].median() for col in NUMERIC}
    patient.update({col: mode(col) for col in CATEGORICAL})
    patient.update({
        "age_years": AGE_BANDS.index(age) * 10 + 5, "gender": gender, "diag_1_group": diag,
        "admission_type": adm_type, "discharge_to": discharge, "a1c_result": a1c,
        "length_of_stay": stay, "number_inpatient": inpatient, "number_emergency": emergency,
        "num_medications": meds,
    })
    score = model.predict_proba(pd.DataFrame([patient]))[:, 1][0]
    # Which of the 10 risk groups does this score fall into?
    decile = int(np.searchsorted(decile_edges[1:-1], score, side="right")) + 1
    actual = decile_rates.loc[decile]

    r1, r2 = st.columns([1, 2])
    with r1:
        level = "🔴 High" if decile >= 9 else "🟠 Medium" if decile >= 6 else "🟢 Low"
        st.metric("Risk level", level)
        st.metric("Risk group", f"Decile {decile} of 10")
        st.metric("Readmission rate for similar patients", f"{actual:.1f}%",
                  delta=f"{actual - overall_all:+.1f} pts vs average", delta_color="inverse")
        st.caption("Based on the actual readmission rate of test-set patients the model "
                   "placed in the same risk group.")
    with r2:
        d = decile_rates.reset_index()
        d.columns = ["decile", "rate"]
        d["this patient"] = d.decile == decile
        fig = px.bar(d, x="decile", y="rate", color="this patient",
                     color_discrete_map={True: RED, False: "#c9d6e3"},
                     title="Actual readmission rate by predicted risk group",
                     labels={"decile": "Risk group (10 = highest)", "rate": "Readmission rate (%)"})
        fig.add_hline(y=overall_all, line_dash="dash", line_color=GREY, annotation_text="Average")
        fig.update_layout(showlegend=False, xaxis=dict(tickmode="linear"))
        st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------------------- #
# Tab 4: Data cleaning
# --------------------------------------------------------------------------- #
with tab_cleaning:
    st.subheader("From raw data to a clean dataset")
    steps = pd.DataFrame(report["steps"], columns=["Step", "Rows left"])
    steps["Rows removed"] = (-steps["Rows left"].diff()).fillna(0).astype(int)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.funnel(steps, x="Rows left", y="Step", title="Rows remaining after each step")
        fig.update_traces(marker_color=BLUE, texttemplate="%{x:,}")
        st.plotly_chart(fig, width="stretch")
    with c2:
        m = pd.Series(report["missing"]).rename_axis("Column").reset_index(name="% missing")
        fig = px.bar(m.sort_values("% missing"), x="% missing", y="Column", orientation="h", text="% missing",
                     title="Missing values in the raw data")
        fig.update_traces(marker_color=[RED if v > 30 else "#e9a23b" for v in m.sort_values("% missing")["% missing"]],
                          texttemplate="%{text:.1f}%")
        fig.update_xaxes(range=[0, 110])
        st.plotly_chart(fig, width="stretch")
    st.dataframe(steps, hide_index=True, width="stretch")
    st.markdown(
        "**Other fixes:** `?` read as missing · lab results `\"None\"` kept as *Not tested* · "
        "`weight` and `payer_code` dropped · ID codes translated with `IDS_mapping.csv` · "
        "900+ ICD-9 codes grouped into 9 diagnosis categories · age bands converted to numbers.")
    with st.expander("Preview the cleaned data"):
        st.dataframe(df.drop(columns="age_band").head(200), width="stretch")
