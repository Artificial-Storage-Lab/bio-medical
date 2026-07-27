"""
10 -- Figures.

Reads the saved JSON result files and produces the summary figures. Depends on
nothing but those files, so it can be re-run after any change without refitting
models.

    python 10_make_figures.py
"""

import matplotlib.pyplot as plt
import numpy as np

import config as cfg
import pipeline as pl

cfg.ensure_dirs()

MODELS = ["CoxPH", "RSF", "GradientBoosting", "DeepSurv"]
COLOURS = {"CoxPH": "#2C6FBB", "RSF": "#E8743B",
           "GradientBoosting": "#19A979", "DeepSurv": "#945ECF"}

TIERS = [
    ("A\nClinical", "tier_A_clinical"),
    ("B\n+Molecular", "tier_B_molecular"),
    ("C\n+Treatment", "tier_C_treatment"),
    ("D\n+Radiomics", "tier_D_radiomics"),
    ("E\n+Embeddings", "tier_E_embeddings"),
    ("F\nAll combined", "tier_F_combined"),
]

labels, saved = [], []
for label, name in TIERS:
    result = pl.load_results(name)
    if result:
        labels.append(label)
        saved.append(result)
    else:
        print(f"(skipping {name} -- not found)")

if not saved:
    raise SystemExit("No results found. Run scripts 02-05 first.")

means = {m: [r["summary"][m]["mean"] for r in saved] for m in MODELS}

# ------------------------------------------------------------
# Tier progression
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6.5))
x = np.arange(len(labels))
for model in MODELS:
    ax.plot(x, means[model], marker="o", lw=2.2, ms=8, label=model, color=COLOURS[model])
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("C-index (nested cross-validation)", fontsize=12)
ax.set_title("Survival Prediction Performance Across Feature Tiers\n"
             "Duke Breast Cancer MRI, event-free survival",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11, loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{cfg.FIGURE_DIR}/tier_progression_line.png", dpi=150, bbox_inches="tight")
plt.close()

# ------------------------------------------------------------
# Grouped bars
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6.5))
width = 0.2
for i, model in enumerate(MODELS):
    offset = (i - (len(MODELS) - 1) / 2) * width
    ax.bar(x + offset, means[model], width, label=model,
           color=COLOURS[model], edgecolor="white", lw=0.5)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("C-index (nested cross-validation)", fontsize=12)
ax.set_title("Model Performance by Feature Tier", fontsize=13, fontweight="bold")
lowest = min(min(means[m]) for m in MODELS)
highest = max(max(means[m]) for m in MODELS)
ax.set_ylim(lowest - 0.02, highest + 0.015)
ax.legend(fontsize=10, loc="upper left", ncol=4)
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(f"{cfg.FIGURE_DIR}/tier_progression_bars.png", dpi=150, bbox_inches="tight")
plt.close()

# ------------------------------------------------------------
# Incremental change with confidence intervals
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6))
transitions = [f"{labels[i].split(chr(10))[0]}->{labels[i + 1].split(chr(10))[0]}\n"
               f"{labels[i + 1].split(chr(10))[1]}" for i in range(len(labels) - 1)]
tx = np.arange(len(transitions))
for i, model in enumerate(MODELS):
    deltas = [means[model][j + 1] - means[model][j] for j in range(len(labels) - 1)]
    offset = (i - (len(MODELS) - 1) / 2) * width
    ax.bar(tx + offset, deltas, width, label=model,
           color=COLOURS[model], edgecolor="white", lw=0.5)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(tx)
ax.set_xticklabels(transitions, fontsize=9)
ax.set_ylabel("Change in C-index", fontsize=12)
ax.set_title("Incremental Contribution of Each Modality", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(f"{cfg.FIGURE_DIR}/tier_incremental_change.png", dpi=150, bbox_inches="tight")
plt.close()

# ------------------------------------------------------------
# Confidence intervals at the final tier
# ------------------------------------------------------------
final = saved[-1]
models_with_ensemble = MODELS + (["Ensemble"] if "Ensemble" in final["summary"] else [])

fig, ax = plt.subplots(figsize=(9, 5.5))
positions = np.arange(len(models_with_ensemble))
for i, model in enumerate(models_with_ensemble):
    s = final["summary"][model]
    ax.errorbar(i, s["mean"],
                yerr=[[s["mean"] - s["ci_low"]], [s["ci_high"] - s["mean"]]],
                fmt="o", ms=9, capsize=6, lw=2,
                color=COLOURS.get(model, "#555555"))
ax.set_xticks(positions)
ax.set_xticklabels(models_with_ensemble, fontsize=10)
ax.set_ylabel("C-index", fontsize=12)
ax.set_title(f"Performance with 95% Bootstrap Confidence Intervals\n"
             f"{labels[-1].replace(chr(10), ' ')}", fontsize=13, fontweight="bold")
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(f"{cfg.FIGURE_DIR}/final_tier_confidence_intervals.png", dpi=150, bbox_inches="tight")
plt.close()

# ------------------------------------------------------------
# Feature count sweep
# ------------------------------------------------------------
import json
import os

sweep_path = f"{cfg.RESULTS_DIR}/feature_count_experiment.json"
if os.path.exists(sweep_path):
    with open(sweep_path) as f:
        sweep = json.load(f)
    n_values = sorted(int(k) for k in sweep["sweep"])
    fig, ax = plt.subplots(figsize=(9, 6))
    for model in MODELS:
        values = [np.mean(sweep["sweep"][str(n)][model]) for n in n_values]
        ax.plot(n_values, values, marker="o", lw=2, ms=7, label=model, color=COLOURS[model])
    ax.set_xlabel("Number of radiomic features retained", fontsize=12)
    ax.set_ylabel("C-index", fontsize=12)
    ax.set_title("Effect of Radiomic Feature Count", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{cfg.FIGURE_DIR}/feature_count_sweep.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Wrote feature_count_sweep.png")

# ------------------------------------------------------------
# Reference configuration comparison
# ------------------------------------------------------------
reference = pl.load_results("modified_deepsurv_reference")
if reference and "comparison" in reference:
    rows = reference["comparison"]
    if rows:
        fig, ax = plt.subplots(figsize=(11, 6))
        names = [r["tier"] for r in rows]
        rx = np.arange(len(names))
        w = 0.36
        ax.bar(rx - w / 2, [r["reference"] for r in rows], w,
               label="Published configuration", color="#8C8C8C")
        ax.bar(rx + w / 2, [r["tuned"] for r in rows], w,
               label="Cross-validated configuration", color="#945ECF")
        ax.set_xticks(rx)
        ax.set_xticklabels([n.replace(": ", "\n") for n in names], fontsize=10)
        ax.set_ylabel("C-index", fontsize=12)
        ax.set_title("Reference Configuration vs Cross-Validated Configuration",
                     fontsize=13, fontweight="bold")
        ax.axhline(0.5, color="black", ls="--", lw=0.8, alpha=0.5)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(f"{cfg.FIGURE_DIR}/reference_configuration_comparison.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        print("Wrote reference_configuration_comparison.png")

# ------------------------------------------------------------
# Printed table, for transcribing into the write-up
# ------------------------------------------------------------
print(f"\n{'=' * 80}\nMEAN C-INDEX BY TIER\n{'=' * 80}")
print(f"{'Tier':<18}" + "".join(f"{m:<14}" for m in MODELS))
for i, label in enumerate(labels):
    clean = label.replace("\n", " ")
    print(f"{clean:<18}" + "".join(f"{means[m][i]:<14.3f}" for m in MODELS))

print(f"\nFigures written to {cfg.FIGURE_DIR}/")
