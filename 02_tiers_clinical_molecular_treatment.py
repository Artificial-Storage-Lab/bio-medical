"""
02 -- Tiers A, B and C: clinical, molecular, treatment.

Each tier adds one modality on top of the previous, so the change in C-index
measures what that modality contributes given everything already available.

    python 02_tiers_clinical_molecular_treatment.py
"""

import numpy as np

import config as cfg
import pipeline as pl

cfg.ensure_dirs()
cfg.check_inputs(("PATIENT_TABLE_CSV",))

clinical = pl.load_clinical()
clinical = pl.build_endpoints(clinical)

# Assemble the three tiers
tier_a = pl.build_feature_block(clinical, pl.CLINICAL_SPEC)
tier_a = tier_a.merge(
    clinical[[cfg.ID_COL, "efs_time_days", "efs_event", "os_time_days", "os_event"]],
    on=cfg.ID_COL, how="inner")

# Patients with no recorded follow-up have no computable survival time. This is
# missingness in the target, so no amount of feature imputation addresses it.
n_before = len(tier_a)
tier_a = tier_a.dropna(subset=["efs_time_days", "efs_event"])
print(f"Dropped {n_before - len(tier_a)} patients with no follow-up recorded")

tier_b = pl.build_feature_block(clinical, pl.MOLECULAR_SPEC, base=tier_a)
tier_c = pl.build_feature_block(clinical, pl.TREATMENT_SPEC, base=tier_b)

tiers = {"A_clinical": tier_a, "B_molecular": tier_b, "C_treatment": tier_c}
for name, df in tiers.items():
    cols = pl.feature_columns(df)
    tiers[name] = pl.coerce_numeric(df, cols)
    print(f"  Tier {name}: N={len(df)}, features={len(cols)}")

# EVALUATE
labels = {
    "A_clinical": "TIER A: Clinical",
    "B_molecular": "TIER B: Clinical + Molecular",
    "C_treatment": "TIER C: Clinical + Molecular + Treatment",
}

all_results = {}
for name, df in tiers.items():
    results, _ = pl.run_nested_cv(df, pl.feature_columns(df), labels[name])
    all_results[name] = results
    pl.save_results(f"tier_{name}", results,
                    extra={"n_patients": len(df),
                           "n_features": len(pl.feature_columns(df)),
                           "n_events": int(df["efs_event"].sum())})

# Persist Tier C -- later tiers build on it, and rebuilding is wasted work.
tier_c.to_csv(f"{cfg.OUTPUT_DIR}/tier_c_features.csv", index=False)
print(f"\nSaved Tier C feature matrix to {cfg.OUTPUT_DIR}/tier_c_features.csv")

# INCREMENTAL CHANGE 
print(f"\n{'=' * 70}\nINCREMENTAL CHANGE\n{'=' * 70}")
print(f"{'Model':<20}{'A':<10}{'B':<10}{'C':<10}{'A->B':<10}{'B->C':<10}")
for model in ["CoxPH", "RSF", "GradientBoosting", "DeepSurv", "Ensemble"]:
    a, b, c = (np.mean(all_results[t][model]) for t in ["A_clinical", "B_molecular", "C_treatment"])
    print(f"{model:<20}{a:<10.3f}{b:<10.3f}{c:<10.3f}{b - a:<+10.3f}{c - b:<+10.3f}")

print("\nNext: 03_tier_d_radiomics.py")
