"""
09 -- Paired imaging comparison on a common cohort.

Tiers D, E and F each keep only the patients who have that imaging record, so
their cohorts and folds differ from Tier C and from each other. A difference
between tiers can then reflect who was included as much as what the imaging
adds. This script removes that confound.

It restricts everything to the patients who have BOTH radiomics and an
embedding, and runs four configurations on exactly the same patients and the
same outer folds:

    C              clinical + molecular + treatment
    C+Radiomics    ... + per-fold selected radiomics      (Tier D pipeline)
    C+Embeddings   ... + per-fold PCA of embeddings       (Tier E pipeline)
    C+Both         ... + both                             (Tier F pipeline)

Uncertainty comes from a patient-level bootstrap: patients are resampled within
each outer fold, and the same resample scores every configuration, so the
differences are paired. This is much tighter and more defensible than
resampling the five fold scores.

Four comparisons are reported per model, so treat individual p-values as
descriptive (or apply a Holm correction if you report them as tests).

    python 09_common_cohort_imaging.py
"""

import json
import os

import numpy as np
import pandas as pd

import config as cfg
import pipeline as pl

cfg.ensure_dirs()
cfg.check_inputs(("EMBEDDINGS_CSV",))
for name in ("tier_c_features.csv", "tier_d_features.csv"):
    if not os.path.exists(f"{cfg.OUTPUT_DIR}/{name}"):
        raise SystemExit(f"{name} not found -- run scripts 02 and 03 first.")

tier_c = pd.read_csv(f"{cfg.OUTPUT_DIR}/tier_c_features.csv", low_memory=False)
tier_d = pd.read_csv(f"{cfg.OUTPUT_DIR}/tier_d_features.csv", low_memory=False)
fixed_cols = pl.feature_columns(tier_c)
radiomics_cols = [c for c in tier_d.columns if c not in tier_c.columns and c != cfg.ID_COL]

embeddings = pd.read_csv(cfg.EMBEDDINGS_CSV)
emb_cols = [c for c in embeddings.columns if c.startswith("medimg_emb_")]

common = tier_d.merge(embeddings[[cfg.ID_COL] + emb_cols], on=cfg.ID_COL, how="inner")
common = common.dropna(subset=["efs_time_days", "efs_event"]).reset_index(drop=True)
if common[cfg.ID_COL].duplicated().any():
    raise SystemExit("Duplicate patient IDs after merging -- check the embeddings file.")

# ------------------------------------------------------------
# Cohort composition: who is lost by requiring both imaging records?
# ------------------------------------------------------------
excluded = tier_c[~tier_c[cfg.ID_COL].isin(common[cfg.ID_COL])]
print(f"{'=' * 70}\nCOMMON COHORT\n{'=' * 70}")
print(f"  Tier C cohort:   N={len(tier_c)}, events={int(tier_c['efs_event'].sum())} "
      f"({tier_c['efs_event'].mean():.1%})")
print(f"  Common cohort:   N={len(common)}, events={int(common['efs_event'].sum())} "
      f"({common['efs_event'].mean():.1%})")
print(f"  Excluded:        N={len(excluded)}, events={int(excluded['efs_event'].sum())}"
      + (f" ({excluded['efs_event'].mean():.1%})" if len(excluded) else ""))
print(f"  Radiomics candidates={len(radiomics_cols)}, embedding dims={len(emb_cols)}")

# ------------------------------------------------------------
# Four configurations, identical patients and folds
# ------------------------------------------------------------
CONFIGS = {
    "C": {},
    "C+Radiomics": {"radiomics_cols": radiomics_cols},
    "C+Embeddings": {"emb_cols": emb_cols},
    "C+Both": {"radiomics_cols": radiomics_cols, "emb_cols": emb_cols},
}
COMPARISONS = [
    ("C", "C+Radiomics"),
    ("C", "C+Embeddings"),
    ("C+Radiomics", "C+Embeddings"),
    ("C", "C+Both"),
]
MODELS = ["CoxPH", "RSF", "GradientBoosting", "DeepSurv", "Ensemble"]

runs = {}
for label, kwargs in CONFIGS.items():
    results, selection, preds = pl.run_nested_cv(
        common, fixed_cols, f"COMMON COHORT -- {label}", return_predictions=True, **kwargs)
    runs[label] = {"results": results, "selection": selection, **preds}

fold_id = runs["C"]["fold_id"]
for label in CONFIGS:
    # Same data frame + same seed => same folds. Fail loudly if that ever breaks.
    assert np.array_equal(runs[label]["fold_id"], fold_id), f"fold mismatch in {label}"

time = common["efs_time_days"].values
event = common["efs_event"].values.astype(bool)

# ------------------------------------------------------------
# Patient-level paired bootstrap
# ------------------------------------------------------------
print(f"\nPatient-level bootstrap ({cfg.BOOTSTRAP_N} replicates, paired across configurations)...")
summary = {}
for model in MODELS:
    preds = {label: runs[label]["oof"][model] for label in CONFIGS}
    point = {label: pl.mean_fold_cindex(p, time, event, fold_id) for label, p in preds.items()}
    boot = pl.patient_bootstrap(preds, time, event, fold_id)
    summary[model] = {
        "c_index": {label: pl.summarise_bootstrap(point[label], boot[label]) for label in CONFIGS},
        "delta": {f"{b} minus {a}": pl.paired_delta(point[a], point[b], boot[a], boot[b])
                  for a, b in COMPARISONS},
    }
    print(f"  {model} done")

# ------------------------------------------------------------
# Tables
# ------------------------------------------------------------
print(f"\n{'=' * 86}\nC-INDEX ON THE COMMON COHORT (N={len(common)}), 95% patient-level bootstrap CI\n{'=' * 86}")
print(f"{'Model':<18}" + "".join(f"{label:<17}" for label in CONFIGS))
for model in MODELS:
    row = f"{model:<18}"
    for label in CONFIGS:
        s = summary[model]["c_index"][label]
        row += f"{s['estimate']:.3f} [{s['ci_low']:.2f},{s['ci_high']:.2f}] "
    print(row)

print(f"\n{'=' * 86}\nPAIRED DIFFERENCES IN C-INDEX (95% CI, bootstrap p)\n{'=' * 86}")
for model in MODELS:
    print(f"{model}")
    for key, s in summary[model]["delta"].items():
        print(f"  {key:<32} {s['estimate']:+.3f}  [{s['ci_low']:+.3f}, {s['ci_high']:+.3f}]"
              f"  p={s['p_value']:.3f}")

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------
payload = {
    "n_patients": int(len(common)),
    "n_events": int(event.sum()),
    "n_excluded_from_tier_c": int(len(excluded)),
    "n_events_excluded": int(excluded["efs_event"].sum()),
    "n_radiomics_candidates": len(radiomics_cols),
    "n_embedding_dims": len(emb_cols),
    "pca_components": cfg.PCA_N_COMPONENTS,
    "bootstrap_replicates": cfg.BOOTSTRAP_N,
    "fold_scores": {label: {m: list(map(float, runs[label]["results"][m])) for m in MODELS}
                    for label in CONFIGS},
    "radiomics_per_fold": {label: [len(s) for s in runs[label]["selection"]]
                           for label in CONFIGS if runs[label]["selection"]},
    "summary": summary,
}
out_json = f"{cfg.RESULTS_DIR}/common_cohort_imaging.json"
with open(out_json, "w") as f:
    json.dump(payload, f, indent=2)
print(f"\nSaved results to {out_json}")

# Out-of-fold predictions, so any further comparison can be computed without refitting.
oof = pd.DataFrame({cfg.ID_COL: common[cfg.ID_COL], "outer_fold": fold_id,
                    "efs_time_days": time, "efs_event": event.astype(int)})
for label in CONFIGS:
    for model in MODELS:
        oof[f"{label}|{model}"] = runs[label]["oof"][model]
oof.to_csv(f"{cfg.OUTPUT_DIR}/common_cohort_oof_predictions.csv", index=False)
print(f"Saved out-of-fold predictions to {cfg.OUTPUT_DIR}/common_cohort_oof_predictions.csv")

print("\nNext: 10_make_figures.py")
