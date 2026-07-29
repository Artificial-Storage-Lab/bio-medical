# MultiModal-Surv

Code for a staged evaluation of multimodal cancer survival prediction using clinical, molecular, treatment, radiomic, and MRI-derived imaging features. This repository contains the preprocessing pipelines, feature engineering, model training, and evaluation code used throughout the project.

**Author:** Hannah Chin (REU, Summer 2026)
**Advisor:** Prof. Xuechen, Prof. Ji
**Institution:** WSU /Penn State / UC Berkeley

---

## Overview

This project investigates how different data modalities contribute to breast cancer survival prediction. A staged multimodal pipeline is used to progressively integrate clinical, molecular, treatment, radiomic, and MRI-derived imaging features, which evaluates the contribution of each modality independently. Multiple survival modeling approaches such as Cox proportional hazards, random survival forests, gradient boosting, and DeepSurv are benchmarked using nested cross-validation to compare predictive performance and assess whether deep learning provides advantages over traditional methods for datasets of this size. 

## Data

- **Dataset:** Duke-Breast-Cancer-MRI (n=922), accessed via NCI Imaging Data
  Commons (`idc-index`, collection `duke_breast_cancer_mri`)
- **Supplementary files** (from the TCIA collection page, DOI 10.7937/TCIA.e3sv-re93):
  `Annotation_Boxes.xlsx`, `Imaging_Features.xlsx`
- **Endpoint:** Event-free survival (EFS), defined as the time from diagnosis to either death or recurrence, whichever occured first. A composite endpoint was used because overall survival (62 events) and recurrence alone (90 events) were too infrequent for reliable modeling. After excluding patients with missing follow-up information, the final cohort consisted of 689 patients with 118 observed events (12.8%). 
- **License:** imaging CC BY-NC 4.0; AIMI annotations CC BY 4.0

> **No data files are included in this repository.** All data must be obtained
> directly from IDC/TCIA. See Setup below.

## Method

Features were added in tiers, ordered by how readily available each data type is in practice: clinical variables, molecular biomarkers, treatment information, radiomic features, and MRI-derived image embeddings. Each tier includes all features from the previous tiers, allowing the incremental predictive value of each additional modality to be measured. 

Model performance is evaluated using nested cross-validation with five outer folds for unbiased performance estimation and three inner folds for hyperparameter tuning. To prevent data leakage, all preprocessing and feature selection are performed independently within each training fold. Radiomic features are selected using correlation filtering, univariate Cox screening, and LASSO-Cox, while image embeddings are reduced using PCA fit only on the training data. This ensures that performance gains reflect the added information from each modality rather than information leaking from the validation set. 

| Tier | Features added |
|------|----------------|
| A | Clinical (age, T/N/M stage, grade, histology, menopause, race) |
| B | + Molecular (ER, PR, HER2, subtype, Oncotype) |
| C | + Treatment (chemo, endocrine, anti-HER2, radiation, surgery) |
| D | + Radiomics (529 pre-extracted features, Saha et al. 2018) |
| E | + MedImageInsight embeddings (1024-dim, PCA-reduced per fold) |
| F | All modalities combined |

**Models compared:** Cox proportional hazards, random survival forest,
gradient boosting survival, DeepSurv, and the Modified-DeepSurv configuration
from Lei et al. (2023) as a reference baseline.

## Results

- **Molecular data drives the biggest gains.** Clinical variables alone yielded a C-index of 0.708–0.735. Adding molecular features (ER, PR, HER2, subtype, Oncotype) improved every model (0.752–0.764), making it the only modality that consistently helped across the board. Treatment features added a further marginal gain (0.769–0.787).

- **Rigorous, leakage-controlled benchmarking of imaging.** Both hand-crafted radiomics and MedImageInsight foundation-model embeddings were evaluated head-to-head under identical nested cross-validation, on the same patients and endpoint. With per-fold feature selection and bootstrap confidence intervals, the framework quantifies exactly what each imaging representation contributes rather than assuming it helps.

- **Classical models match or beat deep learning.** No model consistently outperformed the others. Random Survival Forest was best or tied at most tiers, and Cox Proportional Hazards remained highly competitive throughout. At this sample size (689 patients, 118 events), DeepSurv offered no advantage — and the published Modified-DeepSurv configuration, applied without tuning, sat flat at ~0.66 across tiers, underscoring how sensitive deep models are to configuration and data scale.

- **Strict feature selection is critical.** A sweep of retained radiomic features showed a consistent optimum at 10–20 features across all models. Performance degraded monotonically beyond that, and using all 529 unselected features performed the worst.

- **Strong risk stratification.** Combining all modalities reached a C-index of 0.755–0.788, with a rank-averaged ensemble peaking at 0.793. Out-of-fold risk scores separated patients into low-, medium-, and high-risk groups with clearly distinct event-free survival (log-rank $p = 3.5 \times 10^{-18}$; event rates 7.8%, 9.6%, and 33.9%).

## Repository layout

```
config.py                     All paths and analysis settings
pipeline.py                   Shared components: loading, tiers, models, nested CV

01_build_patient_table.py     Fetch clinical data, build endpoints, write patient table
02_tiers_clinical_molecular_treatment.py   Tiers A, B, C
03_tier_d_radiomics.py        Tier D
04_tier_e_embeddings.py       Tier E
05_tier_f_combined.py         Tier F, plus the full progression
06_feature_count_experiment.py    How many radiomic features to retain
07_modified_deepsurv_baseline.py  Reference configuration comparison
08_risk_stratification.py     Kaplan-Meier by predicted risk group
10_make_figures.py            Figures from saved results

imaging/
  step1_identify_series.py    Find the first post-contrast series per patient
  step2_download.py           Download those series (~56 GB)
  step3_crop.py               Crop to the annotated tumour region
  step5_extract_embeddings.py Extract embeddings (runs in the MedImageInsight repo)
```

## Setup

```bash
pip install -r requirements.txt
```

Download from the TCIA collection page (DOI 10.7937/TCIA.e3sv-re93):
`Imaging_Features.xlsx` and `Annotation_Boxes.xlsx`.

Either place them in a `data/` folder beside the code, or point at wherever
they live:

```bash
export DUKE_DATA_DIR="/path/to/your/data"
```

Two further variables matter only if you re-run the imaging extraction. The
DICOM download is roughly 56 GB, so it often belongs on external storage:

```bash
export DUKE_DICOM_DIR="/Volumes/DRIVE/dicom_series"
export DUKE_CROP_DIR="/path/to/crops"
```

Copy `.env.example` to `.env`, fill in your paths, and `source .env` before
running. `.env` is gitignored, so your paths stay out of the repository.

Scripts fail immediately with a clear message if an expected input is missing,
rather than partway through an analysis.

MedImageInsight is a separate repository with its own environment:

```bash
git clone https://huggingface.co/lion-ai/MedImageInsights
cd MedImageInsights
uv sync
uv run example.py     # confirm the model loads
```

`imaging/step5_extract_embeddings.py` must be copied into that repository and
run with `uv run python`, since the model package is only importable there.

## Running

```bash
python 01_build_patient_table.py    # requires network access

# Imaging pipeline -- only needed for tiers E and F
python imaging/step1_identify_series.py
python imaging/step2_download.py --download    # ~56 GB, several hours
python imaging/step3_crop.py
# then step5 inside the MedImageInsight repository

python 02_tiers_clinical_molecular_treatment.py
python 03_tier_d_radiomics.py
python 04_tier_e_embeddings.py
python 05_tier_f_combined.py
python 06_feature_count_experiment.py
python 07_modified_deepsurv_baseline.py
python 08_risk_stratification.py
python 10_make_figures.py
```

Scripts 02 onward read from disk and write per-fold scores to
`outputs/results/*.json`, so results survive a restarted session and figures
can be regenerated without refitting anything.

Script 06 is slow: it fits a univariate Cox model per candidate feature per
fold, across four feature-count settings.

## Notes and limitations

- **Single institution, no external validation.** All results come from one cohort (Duke). The models have not been tested on data from another institution, so it is unknown how well they generalize across different scanners, protocols, and patient populations.

- **Modest event count.** The composite endpoint yields 118 events among 689 patients. Survival models learn primarily from events, so this limits statistical power.

- **Heterogeneous annotations.** The tumor bounding boxes were drawn by eight radiologists across two imaging protocol phases, introducing variability in how tumors were localized that the imaging features inherit.

- **General-domain imaging model.** MedImageInsight is a foundation model trained across medical imaging broadly, not fine-tuned for breast MRI. Its embeddings may therefore underrepresent breast-specific features, which could partly explain why they did not outperform hand-crafted radiomics.

- **One underperforming fold.** One cross-validation fold consistently scored lower than the others across every tier. Its clinical and molecular composition was balanced and similar to the other folds, so this reflects small-sample variance rather than a systematic subgroup effect. 

## References

**Data citation:**

Saha, A., Harowicz, M. R., Grimm, L. J., Weng, J., Cain, E. H., Kim, C. E.,
Ghate, S. V., Walsh, R., & Mazurowski, M. A. (2021). *Dynamic contrast-enhanced
magnetic resonance images of breast cancer patients with tumor locations*
[Data set]. The Cancer Imaging Archive.
https://doi.org/10.7937/TCIA.e3sv-re93

**TCIA citation:**

Clark, K., Vendt, B., Smith, K., Freymann, J., Kirby, J., Koppel, P.,
Moore, S., Phillips, S., Maffitt, D., Pringle, M., Tarbox, L., & Prior, F.
(2013). The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public
Information Repository. *Journal of Digital Imaging*, 26(6), 1045–1057.
https://doi.org/10.1007/s10278-013-9622-5

**Publication citation:**

Saha, A., Harowicz, M. R., Grimm, L. J., Kim, C. E., Ghate, S. V., Walsh, R.,
& Mazurowski, M. A. (2018). A machine learning approach to radiogenomics of
breast cancer: a study of 922 subjects and 529 DCE-MRI features.
*British Journal of Cancer*, 119(4), 508–516.
