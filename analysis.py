"""Exploratory analysis of 30-day readmissions on the cleaned dataset.

Prints key statistics and saves charts to the images/ folder.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

IMAGES = Path(__file__).parent / "images"
sns.set_theme(style="whitegrid")
BLUE, RED = "#3a7ca5", "#c0392b"


def save(fig, name: str) -> None:
    IMAGES.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(IMAGES / name, dpi=120)
    plt.close(fig)


def rate_table(df: pd.DataFrame, column, min_count: int = 100) -> pd.DataFrame:
    """Readmission rate and patient count per group (small groups dropped)."""
    t = df.groupby(column, observed=True)["readmitted_30d"].agg(rate="mean", patients="size")
    t["rate"] *= 100
    return t[t.patients >= min_count]


def bar_by_group(df, column, title, filename, sort=True, xlabel=None):
    t = rate_table(df, column)
    if sort:
        t = t.sort_values("rate")
    overall = df.readmitted_30d.mean() * 100
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(t) + 1.6))
    ax.barh(t.index.astype(str), t.rate, color=[RED if r > overall else BLUE for r in t.rate])
    ax.axvline(overall, ls="--", color="grey", label=f"Overall {overall:.1f}%")
    ax.set(xlabel=xlabel or "30-day readmission rate (%)", title=title)
    ax.legend(loc="lower right")
    save(fig, filename)
    return t


def show(t: pd.DataFrame, label: str) -> None:
    print(f"\n{label}:")
    print(t.assign(rate=t.rate.round(1).astype(str) + "%").to_string())


def run(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("EXPLORATORY ANALYSIS")
    print("=" * 60)
    print(f"Patients: {len(df):,}   Readmitted within 30 days: {df.readmitted_30d.sum():,} "
          f"({df.readmitted_30d.mean():.1%})")
    print(f"Average length of stay: {df.length_of_stay.mean():.1f} days | "
          f"average medications: {df.num_medications.mean():.1f}")

    # 1. Discharge destination
    show(bar_by_group(df, "discharge_to", "Readmission rate by discharge destination",
                      "readmission_by_discharge.png"), "By discharge destination")

    # 2. Previous inpatient stays (the strongest signal)
    capped = df.number_inpatient.clip(upper=4).rename("prior_inpatient_stays")
    t = rate_table(df.assign(prior_inpatient_stays=capped), "prior_inpatient_stays", min_count=1)
    show(t, "By inpatient stays in the previous year (4 = 4+)")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t.index, t.rate, marker="o", lw=2.5, color=RED)
    for x, (r, n) in zip(t.index, t.values):
        ax.annotate(f"{r:.1f}%\n(n={int(n):,})", (x, r), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8)
    ax.set_xticks(range(5), ["0", "1", "2", "3", "4+"])
    ax.set(xlabel="Inpatient stays in the previous year", ylabel="Readmission rate (%)",
           title="Frequent past admissions predict readmission", ylim=(0, t.rate.max() * 1.3))
    save(fig, "readmission_by_prior_stays.png")

    # 3. Age
    bands = df.age_years.map(lambda a: f"{a - 5}-{a + 5}")
    bands = pd.Categorical(bands, categories=[f"{a}-{a + 10}" for a in range(0, 100, 10)], ordered=True)
    show(bar_by_group(df.assign(age_band=bands), "age_band", "Readmission rate by age",
                      "readmission_by_age.png", sort=False), "By age band")

    # 4. Primary diagnosis
    show(bar_by_group(df, "diag_1_group", "Readmission rate by primary diagnosis",
                      "readmission_by_diagnosis.png"), "By primary diagnosis group")

    # 5. HbA1c testing — a known finding from the original study
    show(bar_by_group(df, "a1c_result", "Readmission rate by HbA1c test result",
                      "readmission_by_a1c.png"), "By HbA1c result")
    print(f"→ HbA1c was only measured for {(df.a1c_result != 'Not tested').mean():.1%} of patients")

    # 6. Length of stay
    t = rate_table(df, "length_of_stay", min_count=1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(t.index, t.rate, color=BLUE)
    ax.set(xlabel="Length of stay (days)", ylabel="Readmission rate (%)",
           title="Longer hospital stays, higher readmission risk")
    save(fig, "readmission_by_length_of_stay.png")

    # 7. Correlation of numeric features with readmission
    numeric = ["length_of_stay", "num_lab_procedures", "num_procedures", "num_medications",
               "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
               "age_years", "n_diabetes_meds", "total_prior_visits", "readmitted_30d"]
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(df[numeric].corr(), annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax, annot_kws={"size": 7})
    ax.set_title("Correlation between numeric features")
    save(fig, "correlation_heatmap.png")

    print(f"\nCharts saved to {IMAGES.name}/")
