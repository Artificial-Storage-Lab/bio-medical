"""
04 -- Tier E: adds MedImageInsight embeddings.

1024-dimensional embeddings of the cropped tumour region, produced by a
pretrained medical imaging foundation model (vision-transformer based, frozen
-- no training here). See imaging/ for how they are extracted.

1024 features on roughly 540 training patients would overfit badly, so the
embeddings are reduced by PCA fit on the training fold only. PCA suits these
dimensions: they are dense and individually uninterpretable, so the
interpretability cost that argues against PCA for radiomics does not apply.

    python 04_tier_e_embeddings.py
"""

import numpy as np
import pandas as pd

import config as cfg
import pipeline as pl

cfg.ensure_dirs()
cfg.check_inputs(("EMBEDDINGS_CSV",))

tier_c = pd.read_csv(f"{cfg.OUTPUT_DIR}/tier_c_features.csv", low_memory=False)
fixed_cols = pl.feature_columns(tier_c)

print("Loading embeddings...")
embeddings = pd.read_csv(cfg.EMBEDDINGS_CSV)
emb_cols = [c for c in embeddings.columns if c.startswith("medimg_emb_")]
print(f"  {len(emb_cols)} dimensions, {len(embeddings)} patients")

tier_e = tier_c.merge(embeddings, on=cfg.ID_COL, how="inner")
tier_e = tier_e.dropna(subset=["efs_time_days", "efs_event"])
print(f"  Tier E: N={len(tier_e)} after merge "
      f"(lost {len(tier_c) - len(tier_e)} without embeddings)")

results, _ = pl.run_nested_cv(
    tier_e, fixed_cols, "TIER E: + MedImageInsight embeddings", emb_cols=emb_cols)

pl.save_results("tier_E_embeddings", results, extra={
    "n_patients": len(tier_e),
    "n_embedding_dims": len(emb_cols),
    "pca_components": cfg.PCA_N_COMPONENTS,
})

tier_e.to_csv(f"{cfg.OUTPUT_DIR}/tier_e_features.csv", index=False)
print(f"Saved Tier E feature matrix to {cfg.OUTPUT_DIR}/tier_e_features.csv")

# ------------------------------------------------------------
# Learned embeddings against hand-crafted radiomics
# ------------------------------------------------------------
tier_c_results = pl.load_results("tier_C_treatment")
tier_d_results = pl.load_results("tier_D_radiomics")

if tier_c_results and tier_d_results:
    print(f"\n{'=' * 70}\nEMBEDDINGS vs RADIOMICS vs BASELINE\n{'=' * 70}")
    print(f"{'Model':<20}{'Tier C':<14}{'Tier D (rad)':<16}{'Tier E (emb)':<16}")
    for model in results:
        c = tier_c_results["summary"][model]["mean"]
        d = tier_d_results["summary"][model]["mean"]
        e = np.mean(results[model])
        print(f"{model:<20}{c:<14.3f}{d:<16.3f}{e:<16.3f}")

print("\nNext: 05_tier_f_combined.py")
