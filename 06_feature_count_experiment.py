"""
06 -- How many radiomic features should be retained?

Sweeps the number of radiomic features (10, 20, 40, 80) and measures C-index at
each setting, for every model. Also compares selection strategies: supervised
Cox-based selection against using all 529 features unselected.

LASSO is not used here because it chooses its own sparsity and cannot be forced
to return an exact count. Selection stops after correlation removal instead,
which permits exact truncation.

Slow -- 5 settings (4 counts + no selection) x 5 folds x 4 models, plus a univariate Cox screen over ~450
features per fold.

    python 06_feature_count_experiment.py
"""

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

import config as cfg
import pipeline as pl

cfg.ensure_dirs()
cfg.check_inputs(("IMAGING_FEATURES_XLSX",))

N_VALUES = [10, 20, 40, 80]

tier_d = pd.read_csv(f"{cfg.OUTPUT_DIR}/tier_d_features.csv", low_memory=False)
tier_c = pd.read_csv(f"{cfg.OUTPUT_DIR}/tier_c_features.csv", low_memory=False)
fixed_cols = pl.feature_columns(tier_c)
radiomics_cols = [c for c in tier_d.columns if c not in tier_c.columns and c != cfg.ID_COL]

time_all = tier_d["efs_time_days"].values
event_all = tier_d["efs_event"].values.astype(bool)
outer_cv = StratifiedKFold(n_splits=cfg.OUTER_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE)

print(f"N={len(tier_d)}, {len(radiomics_cols)} radiomic candidates")


def evaluate_with_features(selector, label):
    """Run all four models using a given per-fold feature selector."""
    results = {"CoxPH": [], "RSF": [], "GradientBoosting": [], "DeepSurv": []}

    for train_pos, test_pos in outer_cv.split(np.arange(len(tier_d)), event_all):
        selected = selector(train_pos)
        cols = fixed_cols + selected
        X = tier_d[cols].apply(pd.to_numeric, errors="coerce").values.astype(float)
        X_train, X_test = pl.impute_from_train(X[train_pos], X[test_pos])
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        time_train, time_test = time_all[train_pos], time_all[test_pos]
        event_train, event_test = event_all[train_pos], event_all[test_pos]
        y_train = Surv.from_arrays(event=event_train, time=time_train)

        cox = CoxPHSurvivalAnalysis(alpha=1.0)
        cox.fit(X_train, y_train)
        results["CoxPH"].append(
            concordance_index_censored(event_test, time_test, cox.predict(X_test))[0])

        rsf = RandomSurvivalForest(n_estimators=200, max_depth=5,
                                   random_state=cfg.RANDOM_STATE, n_jobs=-1)
        rsf.fit(X_train, y_train)
        results["RSF"].append(
            concordance_index_censored(event_test, time_test, rsf.predict(X_test))[0])

        gb = GradientBoostingSurvivalAnalysis(n_estimators=100, learning_rate=0.1,
                                              max_depth=3, random_state=cfg.RANDOM_STATE)
        gb.fit(X_train, y_train)
        results["GradientBoosting"].append(
            concordance_index_censored(event_test, time_test, gb.predict(X_test))[0])

        fit_idx, es_idx = train_test_split(np.arange(len(X_train)), test_size=0.2,
                                           stratify=event_train, random_state=cfg.RANDOM_STATE)
        ds = pl.train_deepsurv(X_train[fit_idx], time_train[fit_idx], event_train[fit_idx],
                               X_train[es_idx], time_train[es_idx], event_train[es_idx],
                               hidden=32)
        pred = pl.predict_torch(ds, X_test)
        results["DeepSurv"].append(
            0.5 if np.isnan(pred).any()
            else concordance_index_censored(event_test, time_test, pred)[0])

    print(f"\n{label}")
    for model, scores in results.items():
        print(f"  {model:<20} {np.mean(scores):.3f} +/- {np.std(scores):.3f}")
    return results


# ------------------------------------------------------------
# Feature-count sweep
# ------------------------------------------------------------
print(f"\n{'=' * 70}\nFEATURE COUNT SWEEP\n{'=' * 70}")

sweep = {}
for n in N_VALUES:
    sweep[n] = evaluate_with_features(
        lambda tp, n=n: pl.select_top_n_by_significance(tier_d, tp, radiomics_cols, n),
        f"--- {n} radiomic features ---")

# ------------------------------------------------------------
# No selection at all
# ------------------------------------------------------------
print(f"\n{'=' * 70}\nNO SELECTION (all {len(radiomics_cols)} features)\n{'=' * 70}")
no_selection = evaluate_with_features(
    lambda tp: radiomics_cols, f"--- all {len(radiomics_cols)} features ---")

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------
models = ["CoxPH", "RSF", "GradientBoosting", "DeepSurv"]
print(f"\n{'=' * 70}\nSUMMARY: C-INDEX BY FEATURE COUNT\n{'=' * 70}")
print(f"{'N':<12}" + "".join(f"{m:<16}" for m in models))
for n in N_VALUES:
    print(f"{n:<12}" + "".join(f"{np.mean(sweep[n][m]):<16.3f}" for m in models))
print(f"{'all ' + str(len(radiomics_cols)):<12}"
      + "".join(f"{np.mean(no_selection[m]):<16.3f}" for m in models))

payload = {
    "sweep": {str(n): {m: list(map(float, v)) for m, v in sweep[n].items()} for n in N_VALUES},
    "no_selection": {m: list(map(float, v)) for m, v in no_selection.items()},
    "n_candidates": len(radiomics_cols),
}
with open(f"{cfg.RESULTS_DIR}/feature_count_experiment.json", "w") as f:
    json.dump(payload, f, indent=2)
print(f"\nSaved results to {cfg.RESULTS_DIR}/feature_count_experiment.json")

print("\nNext: 07_modified_deepsurv_baseline.py")
