"""Predict 30-day readmission with Logistic Regression and Random Forest."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (RocCurveDisplay, average_precision_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

IMAGES = Path(__file__).parent / "images"

CATEGORICAL = ["race", "gender", "admission_type", "discharge_to", "admission_source", "medical_specialty",
               "a1c_result", "glucose_result", "diag_1_group", "diag_2_group", "diag_3_group"]
NUMERIC = ["age_years", "length_of_stay", "num_lab_procedures", "num_procedures", "num_medications",
           "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
           "n_diabetes_meds", "n_med_changes", "on_insulin", "med_changed", "on_diabetes_med"]
TARGET = "readmitted_30d"


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), CATEGORICAL),
    ])


def run(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 60)
    print("READMISSION PREDICTION MODELS")
    print("=" * 60)

    X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    print(f"Train: {len(X_train):,} patients | Test: {len(X_test):,} patients | "
          f"Readmitted in test set: {y_test.sum():,} ({y_test.mean():.1%})")
    print(f"Baseline: always predicting 'not readmitted' would be {1 - y_test.mean():.1%} accurate "
          f"but catch 0 readmissions.")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=30,
                                                class_weight="balanced", random_state=42, n_jobs=-1),
    }
    results, fitted = {}, {}
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, clf in models.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)]).fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = pipe.predict(X_test)
        results[name] = {"roc_auc": roc_auc_score(y_test, proba), "pr_auc": average_precision_score(y_test, proba)}
        fitted[name] = pipe
        print(f"\n--- {name} --- ROC-AUC {results[name]['roc_auc']:.3f} | PR-AUC {results[name]['pr_auc']:.3f}")
        print(classification_report(y_test, pred, target_names=["Not readmitted", "Readmitted"], digits=3))
        print("Confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(y_test, pred))
        RocCurveDisplay.from_predictions(y_test, proba, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], ls="--", color="grey", label="Random guess")
    ax.set_title("ROC curves — 30-day readmission")
    ax.legend(loc="lower right")
    fig.tight_layout()
    IMAGES.mkdir(exist_ok=True)
    fig.savefig(IMAGES / "roc_curves.png", dpi=120)
    plt.close(fig)

    # Who does the model flag? Readmission rate in each risk decile
    best = max(results, key=lambda m: results[m]["roc_auc"])
    proba = fitted[best].predict_proba(X_test)[:, 1]
    deciles = pd.qcut(proba, 10, labels=range(1, 11))
    by_decile = y_test.groupby(deciles, observed=True).mean() * 100
    print(f"\nActual readmission rate by predicted-risk decile ({best}):")
    print(by_decile.round(1).to_string())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(by_decile.index.astype(str), by_decile.values,
           color=["#c0392b" if d >= 9 else "#3a7ca5" for d in by_decile.index])
    ax.axhline(y_test.mean() * 100, ls="--", color="grey", label="Overall rate")
    ax.set(xlabel="Predicted risk decile (10 = highest risk)", ylabel="Actual readmission rate (%)",
           title=f"Risk stratification — {best}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMAGES / "risk_deciles.png", dpi=120)
    plt.close(fig)

    # Feature importance from the Random Forest
    rf = fitted["Random Forest"]
    names = rf.named_steps["prep"].get_feature_names_out()
    importances = (pd.Series(rf.named_steps["clf"].feature_importances_, index=names)
                   .rename(lambda s: s.split("__", 1)[1]).sort_values().tail(15))
    fig, ax = plt.subplots(figsize=(7, 6))
    importances.plot.barh(ax=ax, color="#3a7ca5")
    ax.set(title="Top predictors of readmission (Random Forest)", xlabel="Importance")
    fig.tight_layout()
    fig.savefig(IMAGES / "feature_importance.png", dpi=120)
    plt.close(fig)
    print("\nTop 5 predictors:", ", ".join(importances.sort_values(ascending=False).index[:5]))
    return results
