"""
05 -- Tier F: all modalities combined.

Clinical, molecular, treatment, radiomics and embeddings together. Both imaging
sources are reduced within each fold on training data only, so the combined
matrix stays proportionate to the sample size.

    python 05_tier_f_combined.py
"""

import numpy as np
import pandas as pd

import config as cfg
import pipeline as pl

cfg.ensure_dirs()
cfg.check_inputs(("EMBEDDINGS_CSV",))

tier_d = pd.read_csv(f"{cfg.OUTPUT_DIR}/tier_d_features.csv", low_memory=False)
tier_c = pd.read_csv(f"{cfg.OUTPUT_DIR}/tier_c_features.csv", low_memory=False)
fixed_cols = pl.feature_columns(tier_c)
radiomics_cols = [c for c in tier_d.columns
                  if c not in tier_c.columns and c != cfg.ID_COL]

embeddings = pd.read_csv(cfg.EMBEDDINGS_CSV)
emb_cols = [c for c in embeddings.columns if c.startswith("medimg_emb_")]

tier_f = tier_d.merge(embeddings, on=cfg.ID_COL, how="inner")
tier_f = tier_f.dropna(subset=["efs_time_days", "efs_event"])

print(f"Tier F: N={len(tier_f)}")
print(f"  fixed features:        {len(fixed_cols)}")
print(f"  radiomics candidates:  {len(radiomics_cols)}")
print(f"  embedding dimensions:  {len(emb_cols)}")

results, selection_history = pl.run_nested_cv(
    tier_f, fixed_cols, "TIER F: All modalities combined",
    radiomics_cols=radiomics_cols, emb_cols=emb_cols)

pl.save_results("tier_F_combined", results, extra={
    "n_patients": len(tier_f),
    "n_radiomics_candidates": len(radiomics_cols),
    "n_embedding_dims": len(emb_cols),
    "radiomics_per_fold": [len(f) for f in selection_history],
})

# ------------------------------------------------------------
# Full progression
# ------------------------------------------------------------
tier_names = ["tier_A_clinical", "tier_B_molecular", "tier_C_treatment",
              "tier_D_radiomics", "tier_E_embeddings"]
labels = ["A clinical", "B +molecular", "C +treatment", "D +radiomics", "E +embeddings"]

print(f"\n{'=' * 82}\nFULL TIER PROGRESSION (mean C-index [95% CI])\n{'=' * 82}")
models = ["CoxPH", "RSF", "GradientBoosting", "DeepSurv", "Ensemble"]
print(f"{'Tier':<16}" + "".join(f"{m:<13}" for m in models))

for name, label in zip(tier_names, labels):
    saved = pl.load_results(name)
    if saved:
        row = f"{label:<16}"
        for m in models:
            row += f"{saved['summary'][m]['mean']:<13.3f}" if m in saved["summary"] else f"{'--':<13}"
        print(row)

row = f"{'F all combined':<16}"
for m in models:
    row += f"{np.mean(results[m]):<13.3f}"
print(row)

print(f"\n95% bootstrap confidence intervals, Tier F:")
for m in models:
    mean, lo, hi = pl.bootstrap_ci(results[m])
    print(f"  {m:<18} {mean:.3f}  [{lo:.3f}, {hi:.3f}]")
print("\nOverlapping intervals across tiers indicate the differences are not")
print("statistically distinguishable at this sample size.")

print("\nNext: 06_feature_count_experiment.py")
