"""Clean the raw "Diabetes 130-US Hospitals" dataset for readmission analysis.

Input : data/raw/diabetic_data.csv  (101,766 hospital encounters, 50 columns)
        data/raw/IDS_mapping.csv    (meanings of the numeric ID columns)
Output: data/processed/admissions_clean.csv

Each cleaning step is logged so we can see exactly how many rows it removed.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
RAW_DATA = ROOT / "data" / "raw" / "diabetic_data.csv"
ID_MAPPING = ROOT / "data" / "raw" / "IDS_mapping.csv"
CLEAN_DATA = ROOT / "data" / "processed" / "admissions_clean.csv"
IMAGES = ROOT / "images"

# The 23 diabetes medication columns. Each holds "No", "Steady", "Up" or "Down".
DRUG_COLUMNS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide",
    "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol",
    "troglitazone", "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
]
UNKNOWN_WORDS = ("NULL", "Not Available", "Not Mapped", "Unknown/Invalid")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_raw() -> pd.DataFrame:
    # The file marks missing values with "?". We also turn OFF pandas' default
    # missing-value detection, because the lab columns use the word "None" to
    # mean "test not done" and pandas would silently turn that into NaN.
    return pd.read_csv(RAW_DATA, na_values="?", keep_default_na=False, low_memory=False)


def load_id_mapping() -> dict[str, dict[int, str]]:
    """IDS_mapping.csv holds three tables stacked in one file, separated by
    blank lines. Split them into {column_name: {id: description}}."""
    mappings, current = {}, None
    for line in ID_MAPPING.read_text().splitlines():
        key, _, value = line.partition(",")
        if not key.strip():
            current = None                   # blank separator line
        elif current is None:
            current = key.strip()            # header line, e.g. "admission_type_id"
            mappings[current] = {}
        else:
            mappings[current][int(key)] = value.strip().strip('"')
    return mappings


# --------------------------------------------------------------------------- #
# Helpers for grouping messy categories
# --------------------------------------------------------------------------- #
def is_unknown(desc: str) -> bool:
    return any(word.lower() in desc.lower() for word in UNKNOWN_WORDS)


def group_admission_type(desc: str) -> str:
    if is_unknown(desc):
        return "Unknown"
    if desc == "Trauma Center":
        return "Emergency"
    return desc if desc in ("Emergency", "Urgent", "Elective") else "Other"


def group_discharge(desc: str) -> str:
    d = desc.lower()
    if is_unknown(desc):
        return "Unknown"
    if "home" in d and ("health" in d or "iv provider" in d):
        return "Home with care"
    if d == "discharged to home":
        return "Home"
    if "left ama" in d:
        return "Left against advice"
    if d.startswith("discharged/transferred"):
        return "Transferred to facility"
    return "Other"


def group_admission_source(desc: str) -> str:
    d = desc.lower()
    if is_unknown(desc):
        return "Unknown"
    if "emergency room" in d:
        return "Emergency Room"
    if "referral" in d:
        return "Referral"
    if "transfer" in d:
        return "Transfer"
    return "Other"


def group_icd9(code) -> str:
    """Group raw ICD-9 diagnosis codes into broad categories
    (same groups as Strack et al., 2014, the paper that published this dataset)."""
    if pd.isna(code):
        return "Missing"
    code = str(code)
    if code[0] in "VE":                      # V = health-status codes, E = external causes
        return "Other"
    n = float(code)
    if int(n) == 250:
        return "Diabetes"
    if 390 <= n <= 459 or int(n) == 785:
        return "Circulatory"
    if 460 <= n <= 519 or int(n) == 786:
        return "Respiratory"
    if 520 <= n <= 579 or int(n) == 787:
        return "Digestive"
    if 580 <= n <= 629 or int(n) == 788:
        return "Genitourinary"
    if 800 <= n <= 999:
        return "Injury"
    if 710 <= n <= 739:
        return "Musculoskeletal"
    if 140 <= n <= 239:
        return "Neoplasms"
    return "Other"


# --------------------------------------------------------------------------- #
# Main cleaning function
# --------------------------------------------------------------------------- #
def plot_missing(df: pd.DataFrame) -> None:
    pct = (df.isna().mean() * 100).sort_values()
    pct = pct[pct > 0]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bars = ax.barh(pct.index, pct.values, color=["#c0392b" if v > 30 else "#e9a23b" for v in pct.values])
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set(xlabel="% of rows missing", title="Missing values in the raw data", xlim=(0, 110))
    fig.tight_layout()
    IMAGES.mkdir(exist_ok=True)
    fig.savefig(IMAGES / "missing_values.png", dpi=120)
    plt.close(fig)


def clean(verbose: bool = True, return_report: bool = False):
    """Return the cleaned DataFrame. With return_report=True, also return a
    dict with the row count after each step and the % missing per raw column
    (used by the dashboard)."""
    df = load_raw()
    mapping = load_id_mapping()
    log = [("Raw data", len(df))]
    missing = (df.isna().mean() * 100).round(1)

    if verbose:
        print("=" * 60)
        print("DATA CLEANING")
        print("=" * 60)
        print(f"Loaded {len(df):,} encounters x {df.shape[1]} columns "
              f"({df.patient_nbr.nunique():,} unique patients)")
        print("Columns with missing values (%):", missing[missing > 0].sort_values(ascending=False).to_dict())
    plot_missing(df)

    # 1. Invalid gender values ---------------------------------------------
    df = df[df.gender != "Unknown/Invalid"]
    log.append(("Removed invalid gender", len(df)))

    # 2. One encounter per patient -------------------------------------------
    # Repeat visits by the same patient are not independent, and would let the
    # model "see" a patient in training and again in testing. Keep each
    # patient's first recorded encounter (lowest encounter_id).
    df = df.sort_values("encounter_id").drop_duplicates("patient_nbr", keep="first")
    log.append(("Kept first visit per patient", len(df)))

    # 3. Patients who died or went to hospice can't be readmitted ------------
    discharge_desc = df.discharge_disposition_id.map(mapping["discharge_disposition_id"]).fillna("Unknown")
    can_return = ~discharge_desc.str.contains("Expired|Hospice", case=False)
    df, discharge_desc = df[can_return], discharge_desc[can_return]
    log.append(("Removed deaths / hospice", len(df)))

    # 4. Translate ID codes into readable groups -----------------------------
    df = df.assign(
        admission_type=df.admission_type_id.map(mapping["admission_type_id"]).fillna("NULL").map(group_admission_type),
        discharge_to=discharge_desc.map(group_discharge),
        admission_source=df.admission_source_id.map(mapping["admission_source_id"]).fillna("NULL").map(group_admission_source),
    )

    # 5. Missing values --------------------------------------------------------
    df["race"] = df.race.fillna("Unknown")
    df["medical_specialty"] = df.medical_specialty.fillna("Unknown")
    top_specialties = df.medical_specialty.value_counts().index[:10]
    df["medical_specialty"] = df.medical_specialty.where(df.medical_specialty.isin(top_specialties), "Other")
    df["a1c_result"] = df.A1Cresult.replace({"None": "Not tested"})
    df["glucose_result"] = df.max_glu_serum.replace({"None": "Not tested"})

    # 6. Diagnosis codes -> categories ----------------------------------------
    for i in (1, 2, 3):
        df[f"diag_{i}_group"] = df[f"diag_{i}"].map(group_icd9)

    # 7. Age: "[70-80)" -> 75 ----------------------------------------------------
    df["age_years"] = df.age.str.extract(r"\[(\d+)-").astype(int)[0] + 5

    # 8. New features ----------------------------------------------------------
    drugs = df[DRUG_COLUMNS]
    df["n_diabetes_meds"] = (drugs != "No").sum(axis=1)
    df["n_med_changes"] = drugs.isin(["Up", "Down"]).sum(axis=1)
    df["total_prior_visits"] = df.number_outpatient + df.number_emergency + df.number_inpatient
    df["on_insulin"] = (df.insulin != "No").astype(int)
    df["med_changed"] = (df.change == "Ch").astype(int)
    df["on_diabetes_med"] = (df.diabetesMed == "Yes").astype(int)

    # 9. Target: readmitted within 30 days? ("<30" vs ">30" or "NO") --------
    df["readmitted_30d"] = (df.readmitted == "<30").astype(int)

    # 10. Drop columns we no longer need ---------------------------------------
    df = df.drop(columns=[
        "weight",          # 97% missing
        "payer_code",      # 40% missing, insurance type isn't a clinical factor
        "encounter_id", "admission_type_id", "discharge_disposition_id", "admission_source_id",
        "diag_1", "diag_2", "diag_3", "age", "A1Cresult", "max_glu_serum", "change", "diabetesMed",
        "readmitted", *DRUG_COLUMNS,
    ]).rename(columns={"time_in_hospital": "length_of_stay"})

    assert df.isna().sum().sum() == 0, "There should be no missing values left"
    assert df.patient_nbr.is_unique

    if verbose:
        print("\nCleaning steps:")
        prev = None
        for step, rows in log:
            removed = "" if prev is None else f"(-{prev - rows:,})"
            print(f"  {step:<32} {rows:>8,} rows {removed}")
            prev = rows
        print(f"\nFinal dataset: {len(df):,} patients x {df.shape[1]} columns, no missing values")
        print(f"30-day readmission rate: {df.readmitted_30d.mean():.1%}")

    CLEAN_DATA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_DATA, index=False)
    if return_report:
        return df, {"steps": log, "missing": missing[missing > 0].sort_values(ascending=False).to_dict()}
    return df


if __name__ == "__main__":
    clean()
