"""
03 -- Tier D: adds hand-crafted radiomic features.

529 pre-extracted DCE-MRI features from Saha et al. (2018), computed from
radiologist-drawn tumour bounding boxes. Selection runs inside each outer fold
on training patients only: near-zero-variance removal, univariate Cox ranking,
correlation filtering, then LASSO-Cox.

Also records which features are chosen in each fold, since features selected
repeatedly are more trustworthy than fold-specific ones.

    python 03_tier_d_radiomics.py
"""

from collections import Counter

import numpy as np
import pandas as pd

import config as cfg
import pipeline as pl

cfg.ensure_dirs()
cfg.check_inputs(("IMAGING_FEATURES_XLSX",))

tier_c = pd.read_csv(f"{cfg.OUTPUT_DIR}/tier_c_features.csv", low_memory=False)
fixed_cols = pl.feature_columns(tier_c)
print(f"Tier C: N={len(tier_c)}, features={len(fixed_cols)}")

# Radiomics
print("\nLoading radiomic features...")
radiomics = pd.read_excel(cfg.IMAGING_FEATURES_XLSX, sheet_name="Imaging Features")
radiomics = radiomics.rename(columns={"Patient ID": cfg.ID_COL})

candidate_cols = [c for c in radiomics.columns if c != cfg.ID_COL]
for c in candidate_cols:
    radiomics[c] = pd.to_numeric(radiomics[c], errors="coerce")

coverage = radiomics[candidate_cols].notna().mean()
candidate_cols = coverage[coverage >= cfg.RADIOMICS_COVERAGE_MIN].index.tolist()
print(f"  {len(candidate_cols)} features meet the {cfg.RADIOMICS_COVERAGE_MIN:.0%} coverage threshold")

tier_d = tier_c.merge(radiomics[[cfg.ID_COL] + candidate_cols], on=cfg.ID_COL, how="inner")
tier_d = tier_d.dropna(subset=["efs_time_days", "efs_event"])
print(f"  Tier D: N={len(tier_d)} after merge")

# Evaluate
results, selection_history = pl.run_nested_cv(
    tier_d, fixed_cols, "TIER D: + Radiomics", radiomics_cols=candidate_cols)

# Selection stability
counts = Counter(f for fold in selection_history for f in fold)
print(f"\n{'=' * 70}\nRADIOMIC FEATURE STABILITY\n{'=' * 70}")
print(f"Features selected in all {cfg.OUTER_FOLDS} folds:")
for feature, count in counts.most_common():
    if count == cfg.OUTER_FOLDS:
        print(f"  {feature}")
print(f"\nTop 15 by selection frequency:")
for feature, count in counts.most_common(15):
    print(f"  {count}/{cfg.OUTER_FOLDS}  {feature}")

pl.save_results("tier_D_radiomics", results, extra={
    "n_patients": len(tier_d),
    "n_radiomics_candidates": len(candidate_cols),
    "features_per_fold": [len(f) for f in selection_history],
    "selection_counts": dict(counts.most_common(30)),
})

tier_d.to_csv(f"{cfg.OUTPUT_DIR}/tier_d_features.csv", index=False)
print(f"Saved Tier D feature matrix to {cfg.OUTPUT_DIR}/tier_d_features.csv")

# Comparison against Tier C
tier_c_results = pl.load_results("tier_C_treatment")
if tier_c_results:
    print(f"\n{'=' * 70}\nDOES RADIOMICS IMPROVE ON TIER C?\n{'=' * 70}")
    print(f"{'Model':<20}{'Tier C':<12}{'Tier D':<12}{'Change':<10}")
    for model in results:
        c = tier_c_results["summary"][model]["mean"]
        d = np.mean(results[model])
        print(f"{model:<20}{c:<12.3f}{d:<12.3f}{d - c:<+10.3f}")

print("\nNext: 04_tier_e_embeddings.py")
