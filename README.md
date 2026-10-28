# Hospital Readmission Analysis

Hospital readmissions within 30 days of discharge are costly and often preventable. This project cleans a real, messy dataset of **101,766 diabetic patient encounters from 130 US hospitals**, explores what drives readmission, and builds machine learning models that rank patients by readmission risk.

**Tools:** Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · Streamlit · Plotly

## Dataset

[Diabetes 130-US Hospitals for Years 1999–2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008), from the UCI Machine Learning Repository (also on Kaggle). It has 50 columns covering demographics, admission details, lab tests, diagnoses (ICD-9 codes), 23 diabetes medications and whether the patient was readmitted.

> Strack, B. et al. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records.* BioMed Research International. Licensed under CC BY 4.0.

## 1. Data cleaning

The raw data had a lot of problems:

![Missing values](images/missing_values.png)

| Problem | How it was handled |
|---|---|
| Missing values stored as `?` instead of blanks | Read with `na_values="?"` |
| **Hidden trap:** the lab columns use the word `"None"` to mean *test not done*, and pandas silently converts it to NaN | Turned off pandas' default NA detection (`keep_default_na=False`) and relabelled it as "Not tested" |
| `weight` is 97% missing, `payer_code` 40% missing | Dropped both columns |
| `medical_specialty` is 49% missing and has 70+ values | Filled with "Unknown" and kept the top 10 (the rest became "Other") |
| `race` is 2% missing | Filled with "Unknown" |
| 3 rows with gender "Unknown/Invalid" | Removed |
| 71,518 patients but 101,766 rows, because of repeat visits | Kept each patient's **first** visit, so no patient appears in both the training and test sets |
| Patients who **died or went to hospice** can't be readmitted | Removed, using the descriptions in `IDS_mapping.csv` |
| Admission and discharge columns are numeric codes (1–30) | Translated with `IDS_mapping.csv` and grouped into readable categories |
| Diagnoses are raw ICD-9 codes (900+ unique, e.g. `428`, `250.83`, `V45`) | Grouped into 9 categories (Circulatory, Respiratory, Diabetes and so on), following the original paper |
| Age stored as text bands like `[70-80)` | Converted to a number (75) |
| 23 separate medication columns | Summarised into `n_diabetes_meds`, `n_med_changes` and `on_insulin` |
| Target has 3 values: `NO`, `>30`, `<30` | Converted to `readmitted_30d` = 1 only for `<30` |

**Result:**

| Step | Rows left | Removed |
|---|---|---|
| Raw data | 101,766 | |
| Removed invalid gender | 101,763 | 3 |
| Kept first visit per patient | 71,515 | 30,248 |
| Removed deaths / hospice | **69,970** | 1,545 |

The final dataset has 69,970 patients, 28 columns, zero missing values, and a 30-day readmission rate of **9.0%**.

## 2. Key findings

| Insight | Detail |
|---|---|
| **Past admissions are the strongest signal** | Readmission rises from 8.1% (no stays in the past year) to **29.3%** (4+ stays) |
| **Discharge destination matters** | Patients transferred to another facility were readmitted at **15.1%**, versus 6.9% for those sent home |
| **Age** | Under-60s: about 7%. Over-70s: above 10% |
| **Primary diagnosis** | Injury (10.8%) and Circulatory (9.7%) are highest. Respiratory is lowest (7.3%) |
| **HbA1c testing is rare** | This key diabetes blood test was done for only **18.4%** of patients. Tested patients had slightly lower readmission rates (8.2–8.6% vs 9.1%) |

![Prior stays](images/readmission_by_prior_stays.png)

![Discharge](images/readmission_by_discharge.png)
![Age](images/readmission_by_age.png)

## 3. Predictive models

- 75/25 stratified train/test split (52,477 / 17,493 patients)
- Scaling and one-hot encoding inside a scikit-learn `Pipeline`, to avoid data leakage
- `class_weight="balanced"` because only 9% of patients are readmitted

**Why not accuracy?** A model that always predicts "not readmitted" is 91% accurate and catches nobody.

| Model | ROC-AUC | PR-AUC | Recall (readmitted) | Precision (readmitted) |
|---|---|---|---|---|
| Logistic Regression | 0.642 | 0.151 | 0.541 | 0.140 |
| **Random Forest** | **0.647** | **0.153** | 0.540 | 0.141 |

An AUC of about 0.65 is in line with published results on this dataset. Predicting readmission from administrative data alone is hard, because a lot of what happens after discharge isn't recorded.

### Risk stratification: where the model is useful

The model can't perfectly predict individual patients, but it **ranks** them well. When patients are sorted into ten groups by predicted risk, the highest-risk 10% are actually readmitted at **18.9%**. That's 2× the average and almost 5× the lowest-risk group (3.9%).

![Risk deciles](images/risk_deciles.png)

![Feature importance](images/feature_importance.png)

## 4. Recommendations

1. **Use the model as a ranking tool, not a yes/no predictor.** Focus follow-up calls and care coordination on the top 10–20% risk group.
2. **Prioritise patients with prior inpatient stays.** Each previous admission adds roughly 5 percentage points of risk.
3. **Improve handovers for patients transferred to other facilities**, the highest-risk discharge group.
4. **Test HbA1c more often.** It was measured in fewer than 1 in 5 diabetic patients.

## 5. Interactive dashboard

The project includes a Streamlit dashboard (`dashboard.py`) for exploring the data in the browser:

- **Overview:** key charts that update when you change the sidebar filters (age, gender, admission type, diagnosis)
- **Patient groups:** compare readmission rates across any category
- **Risk calculator:** enter a patient's details and see which risk group the model puts them in
- **Data cleaning:** the cleaning funnel and missing-values chart

![Dashboard overview](images/dashboard_overview.png)

![Risk calculator](images/dashboard_risk_calculator.png)

## Limitations

- The data is from 1999–2008, so care practices have changed since.
- These are correlations, not causes. For example, transferred patients are likely sicker to begin with.
- It covers only diabetic patients in US hospitals.

## Project structure

```
hospital-readmission-analysis/
├── data/
│   ├── raw/
│   │   ├── diabetic_data.csv     # original dataset (101,766 rows)
│   │   └── IDS_mapping.csv       # meanings of the ID code columns
│   └── processed/                # cleaned data (created when you run the project)
├── images/                       # all charts
├── data_cleaning.py              # step 1: cleaning + missing-values chart
├── analysis.py                   # step 2: exploratory analysis + charts
├── model.py                      # step 3: models, evaluation, risk deciles
├── main.py                       # runs all three steps
├── dashboard.py                  # interactive Streamlit dashboard
└── requirements.txt
```

## How to run (PyCharm)

1. **File → Open** and select the project folder.
2. Create a virtual environment and install `requirements.txt`. PyCharm offers this automatically, or run `pip install -r requirements.txt` in the terminal.
3. Open `main.py` and click **Run** ▶. It takes about 30 seconds.

Results print in the Run console, the cleaned data is saved to `data/processed/`, and charts go to `images/`.

### Run the dashboard

In the PyCharm terminal (**Alt+F12**, or **⌥F12** on a Mac), type:

```bash
streamlit run dashboard.py
```

It opens in your browser at `http://localhost:8501`. The first load takes about 20 seconds while it cleans the data and trains the model; after that it's instant. Press **Ctrl+C** in the terminal to stop it.
