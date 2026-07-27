"""
08 -- Risk stratification.

Splits patients into low, medium and high risk tertiles by predicted risk and
plots Kaplan-Meier event-free survival for each group, with a log-rank test.

Risk scores are out-of-fold: each patient is scored by a model that never saw
them during training, so the separation is not an artefact of the model
memorising its own training data.

    python 08_risk_stratification.py
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

import config as cfg
import pipeline as pl

cfg.ensure_dirs()
cfg.check_inputs(("PATIENT_TABLE_CSV",))

# Tier C is the point where the modalities that measurably help are all present.
tier = pd.read_csv(f"{cfg.OUTPUT_DIR}/tier_c_features.csv", low_memory=False)
feature_cols = pl.feature_columns(tier)
tier = pl.coerce_numeric(tier, feature_cols)

X = tier[feature_cols].values.astype(float)
time = tier["efs_time_days"].values
event = tier["efs_event"].values.astype(bool)

# ------------------------------------------------------------
# Out-of-fold risk scores
# ------------------------------------------------------------
print("Generating out-of-fold risk scores...")
oof_risk = np.full(len(tier), np.nan)
outer_cv = StratifiedKFold(n_splits=cfg.OUTER_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE)

for train_idx, test_idx in outer_cv.split(X, event):
    X_train, X_test = pl.impute_from_train(X[train_idx], X[test_idx])
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    cox = CoxPHSurvivalAnalysis(alpha=1.0)
    cox.fit(X_train, Surv.from_arrays(event=event[train_idx], time=time[train_idx]))
    oof_risk[test_idx] = cox.predict(X_test)

# ------------------------------------------------------------
# Tertiles
# ------------------------------------------------------------
cutoffs = np.quantile(oof_risk, [1 / 3, 2 / 3])
group_idx = np.digitize(oof_risk, cutoffs)
group_names = {0: "Low risk", 1: "Medium risk", 2: "High risk"}

km_df = pd.DataFrame({
    "time_years": time / 365.25,
    "event": event.astype(int),
    "risk_group": [group_names[g] for g in group_idx],
})

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
plt.figure(figsize=(9, 6))
kmf = KaplanMeierFitter()
colours = {"Low risk": "#2E7D32", "Medium risk": "#F9A825", "High risk": "#C62828"}

for label in ["Low risk", "Medium risk", "High risk"]:
    mask = km_df["risk_group"] == label
    kmf.fit(km_df.loc[mask, "time_years"], km_df.loc[mask, "event"],
            label=f"{label} (n={int(mask.sum())})")
    kmf.plot_survival_function(color=colours[label], ci_show=True)

plt.xlabel("Time (years)")
plt.ylabel("Event-free survival probability")
plt.title("Event-Free Survival by Predicted Risk Group")
plt.legend(loc="lower left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{cfg.FIGURE_DIR}/kaplan_meier_risk_groups.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved figure to {cfg.FIGURE_DIR}/kaplan_meier_risk_groups.png")

# ------------------------------------------------------------
# Log-rank test
# ------------------------------------------------------------
test = multivariate_logrank_test(km_df["time_years"], km_df["risk_group"], km_df["event"])
print(f"\nLog-rank test across risk groups:")
print(f"  statistic = {test.test_statistic:.2f}")
print(f"  p-value   = {test.p_value:.2e}")

print(f"\nEvent rate by risk group:")
group_stats = {}
for label in ["Low risk", "Medium risk", "High risk"]:
    sub = km_df[km_df["risk_group"] == label]
    rate = float(sub["event"].mean())
    group_stats[label] = {"n": int(len(sub)), "events": int(sub["event"].sum()), "rate": rate}
    print(f"  {label:<14} {int(sub['event'].sum())}/{len(sub)} ({rate:.1%})")

with open(f"{cfg.RESULTS_DIR}/risk_stratification.json", "w") as f:
    json.dump({
        "logrank_statistic": float(test.test_statistic),
        "logrank_p": float(test.p_value),
        "groups": group_stats,
    }, f, indent=2)
print(f"\nSaved results to {cfg.RESULTS_DIR}/risk_stratification.json")

print("\nNext: 09_nodal_prediction.py")
