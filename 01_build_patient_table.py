"""
01 -- Build the patient-level table.

Pulls the clinical table from NCI Imaging Data Commons, constructs the survival
endpoints, summarises imaging availability, and writes a single patient-level
CSV that every later script reads.

Requires network access on first run (downloads the clinical table via
idc-index). Subsequent scripts read the saved CSV and need no network.

    python 01_build_patient_table.py
"""

import numpy as np
import pandas as pd

import config as cfg
import pipeline as pl

cfg.ensure_dirs()

# ------------------------------------------------------------
# Clinical table
# ------------------------------------------------------------
print("Connecting to NCI Imaging Data Commons...")
from idc_index import index

client = index.IDCClient()

print("Fetching clinical table...")
clinical = client.get_clinical_table("duke_breast_cancer_mri_clinical")
print(f"  {clinical.shape[0]} patients, {clinical.shape[1]} columns")

for col in clinical.columns:
    if clinical[col].dtype == object:
        clinical[col] = pl.clean_placeholders(clinical[col])

clinical = pl.build_endpoints(clinical)

dob_col = [c for c in clinical.columns if "date_of_birth_days" in c.lower()]
if dob_col:
    clinical["age_at_diagnosis_years"] = (
        pd.to_numeric(clinical[dob_col[0]], errors="coerce").abs() / 365.25
    )

# ------------------------------------------------------------
# Imaging availability
# ------------------------------------------------------------
print("\nQuerying imaging metadata...")
series_metadata = client.sql_query("""
    SELECT * FROM index
    WHERE collection_id = 'duke_breast_cancer_mri'
""")
print(f"  {len(series_metadata)} series across {series_metadata['PatientID'].nunique()} patients")

imaging_summary = (
    series_metadata.groupby("PatientID")
    .agg(n_series=("SeriesInstanceUID", "count"),
         modalities=("Modality", lambda x: ",".join(sorted(set(x)))),
         has_segmentation=("Modality", lambda x: int("SEG" in set(x))))
    .reset_index()
    .rename(columns={"PatientID": cfg.ID_COL})
)

patient_table = clinical.merge(imaging_summary, on=cfg.ID_COL, how="left")
patient_table["n_series"] = patient_table["n_series"].fillna(0)
patient_table["has_segmentation"] = patient_table["has_segmentation"].fillna(0).astype(int)

patient_table.to_csv(cfg.PATIENT_TABLE_CSV, index=False)
print(f"\nWrote {cfg.PATIENT_TABLE_CSV}  ({patient_table.shape[0]} x {patient_table.shape[1]})")

# ------------------------------------------------------------
# Cohort summary
# ------------------------------------------------------------
print("\n" + "=" * 62)
print("COHORT SUMMARY")
print("=" * 62)

for label, event_col, time_col in [
    ("Overall survival", "os_event", "os_time_days"),
    ("Event-free survival", "efs_event", "efs_time_days"),
]:
    n_events = int(patient_table[event_col].sum())
    n_valid = int(patient_table[time_col].notna().sum())
    print(f"{label:<22} {n_events} events / {len(patient_table)} patients "
          f"({n_events / len(patient_table):.1%}), {n_valid} with usable follow-up")

evaluable = patient_table.dropna(subset=["efs_time_days", "efs_event"])
excluded = len(patient_table) - len(evaluable)
n_ev = int(evaluable["efs_event"].sum())
print(f"\nEvaluable for modelling: {len(evaluable)} "
      f"({n_ev} EFS events, {n_ev / len(evaluable):.1%} of the evaluable cohort)")
print(f"Excluded (no recorded follow-up): {excluded}")

# Confirm the exclusions are genuine rather than a coercion artefact.
if excluded:
    date_cols = [
        [c for c in patient_table.columns if "days_to_death" in c.lower()][0],
        [c for c in patient_table.columns if "age_at_last_contact" in c.lower()][0],
        [c for c in patient_table.columns if "days_to_local_recurrence" in c.lower()][0],
        [c for c in patient_table.columns if "days_to_distant_recurrence" in c.lower()][0],
    ]
    missing = patient_table[patient_table["efs_time_days"].isna()]
    all_absent = int(missing[date_cols].isna().all(axis=1).sum())
    print(f"  of which {all_absent}/{excluded} have none of the four date fields")
    if all_absent < excluded:
        print("  WARNING: some excluded patients have a date present -- investigate")

print(f"\nMedian follow-up: {evaluable['efs_time_days'].median():.0f} days")
print(f"Patients with a tumour segmentation available: {int(patient_table['has_segmentation'].sum())}")
print("\nNext: 02_tiers_clinical_molecular_treatment.py")
