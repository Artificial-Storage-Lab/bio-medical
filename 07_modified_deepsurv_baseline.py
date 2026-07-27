"""
07 -- Reference configuration baseline (Lei et al. 2023).

Modified-DeepSurv is DeepSurv with fixed published hyperparameters: one hidden
layer of 450 units, SELU, a single tanh output, Adam at lr=0.07 with decay
0.003, no tuning and no early stopping. It is a configuration, not a distinct
architecture.

Running it across the same tiers on the same folds shows what changes when
hyperparameters are selected by cross-validation rather than taken from a
paper fitted to a different cohort.

    python 07_modified_deepsurv_baseline.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored

import config as cfg
import pipeline as pl

cfg.ensure_dirs()

tier_c = pd.read_csv(f"{cfg.OUTPUT_DIR}/tier_c_features.csv", low_memory=False)
fixed_cols_c = pl.feature_columns(tier_c)


def evaluate_reference(df, fixed_cols, tier_name, radiomics_cols=None, emb_cols=None):
    """Fit the published configuration across outer folds."""
    time_all = df["efs_time_days"].values
    event_all = df["efs_event"].values.astype(bool)
    outer_cv = StratifiedKFold(n_splits=cfg.OUTER_FOLDS, shuffle=True,
                               random_state=cfg.RANDOM_STATE)

    scores, n_diverged = [], 0
    for fold, (train_pos, test_pos) in enumerate(outer_cv.split(np.arange(len(df)), event_all)):
        X_train, X_test, _ = pl.build_fold_matrices(
            df, train_pos, test_pos, fixed_cols, radiomics_cols, emb_cols)

        model, diverged = pl.train_modified_deepsurv(
            X_train, time_all[train_pos], event_all[train_pos])
        n_diverged += int(diverged)

        pred = pl.predict_torch(model, X_test)
        c = 0.5 if np.isnan(pred).any() else concordance_index_censored(
            event_all[test_pos], time_all[test_pos], pred)[0]
        scores.append(c)
        print(f"  Fold {fold + 1}: {c:.3f}" + ("   (training diverged)" if diverged else ""))

    note = f"   [{n_diverged}/{cfg.OUTER_FOLDS} folds diverged]" if n_diverged else ""
    print(f"  {tier_name}: {np.mean(scores):.3f} +/- {np.std(scores):.3f}{note}")
    return scores, n_diverged


# ------------------------------------------------------------
# Assemble whichever tiers are available
# ------------------------------------------------------------
tier_a = pl.build_feature_block(pl.build_endpoints(pl.load_clinical()), pl.CLINICAL_SPEC)
clinical = pl.build_endpoints(pl.load_clinical())
tier_a = tier_a.merge(clinical[[cfg.ID_COL, "efs_time_days", "efs_event"]],
                      on=cfg.ID_COL, how="inner").dropna(subset=["efs_time_days", "efs_event"])
tier_a = pl.coerce_numeric(tier_a, pl.feature_columns(tier_a))

tier_b = pl.build_feature_block(clinical, pl.MOLECULAR_SPEC, base=tier_a)
tier_b = pl.coerce_numeric(tier_b, pl.feature_columns(tier_b))

runs = [
    ("A: Clinical", tier_a, pl.feature_columns(tier_a), None, None),
    ("B: +Molecular", tier_b, pl.feature_columns(tier_b), None, None),
    ("C: +Treatment", tier_c, fixed_cols_c, None, None),
]

import os

if os.path.exists(f"{cfg.OUTPUT_DIR}/tier_d_features.csv"):
    tier_d = pd.read_csv(f"{cfg.OUTPUT_DIR}/tier_d_features.csv", low_memory=False)
    radiomics_cols = [c for c in tier_d.columns if c not in tier_c.columns and c != cfg.ID_COL]
    runs.append(("D: +Radiomics", tier_d, fixed_cols_c, radiomics_cols, None))

if os.path.exists(cfg.EMBEDDINGS_CSV):
    embeddings = pd.read_csv(cfg.EMBEDDINGS_CSV)
    emb_cols = [c for c in embeddings.columns if c.startswith("medimg_emb_")]
    tier_e = tier_c.merge(embeddings, on=cfg.ID_COL, how="inner").dropna(
        subset=["efs_time_days", "efs_event"])
    runs.append(("E: +Embeddings", tier_e, fixed_cols_c, None, emb_cols))

    if os.path.exists(f"{cfg.OUTPUT_DIR}/tier_d_features.csv"):
        tier_f = tier_d.merge(embeddings, on=cfg.ID_COL, how="inner").dropna(
            subset=["efs_time_days", "efs_event"])
        runs.append(("F: All combined", tier_f, fixed_cols_c, radiomics_cols, emb_cols))

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
reference_results = {}
for tier_name, df, fcols, rcols, ecols in runs:
    print(f"\n{'=' * 70}\n{tier_name}\n{'=' * 70}")
    scores, n_div = evaluate_reference(df, fcols, tier_name, rcols, ecols)
    reference_results[tier_name] = {"scores": scores, "n_diverged": n_div}

# ------------------------------------------------------------
# Against the tuned configuration
# ------------------------------------------------------------
tuned_lookup = {
    "A: Clinical": "tier_A_clinical",
    "B: +Molecular": "tier_B_molecular",
    "C: +Treatment": "tier_C_treatment",
    "D: +Radiomics": "tier_D_radiomics",
    "E: +Embeddings": "tier_E_embeddings",
    "F: All combined": "tier_F_combined",
}

print(f"\n{'=' * 78}\nPUBLISHED CONFIGURATION vs CROSS-VALIDATED CONFIGURATION\n{'=' * 78}")
print(f"{'Tier':<20}{'Reference':<16}{'Tuned DeepSurv':<18}{'Difference':<12}")

comparison = []
for tier_name in reference_results:
    ref_mean = float(np.mean(reference_results[tier_name]["scores"]))
    saved = pl.load_results(tuned_lookup.get(tier_name, ""))
    if saved and "DeepSurv" in saved["summary"]:
        tuned = saved["summary"]["DeepSurv"]["mean"]
        print(f"{tier_name:<20}{ref_mean:<16.3f}{tuned:<18.3f}{tuned - ref_mean:<+12.3f}")
        comparison.append({"tier": tier_name, "reference": ref_mean, "tuned": tuned})
    else:
        print(f"{tier_name:<20}{ref_mean:<16.3f}{'(not run)':<18}{'--':<12}")

print("\nBoth use identical folds, features and endpoint. The only difference is")
print("whether hyperparameters were fixed from the publication or selected by the")
print("inner cross-validation loop.")

pl.save_results("modified_deepsurv_reference",
                {k: v["scores"] for k, v in reference_results.items()},
                extra={"comparison": comparison,
                       "divergences": {k: v["n_diverged"] for k, v in reference_results.items()}})

print("\nNext: 08_risk_stratification.py")
