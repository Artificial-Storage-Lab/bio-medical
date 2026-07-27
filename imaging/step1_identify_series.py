"""
Imaging step 1 -- identify the first post-contrast series for each patient.

The dataset spans 2000-2014 and several scanners, and series descriptions vary
accordingly. Matching is tiered:

  Tier 1  explicit phase-1 or 1st-pass tagging   (high confidence)
  Tier 2  a generic dynamic series with no phase number  (lower confidence)

Naming variants encountered include "ax dyn 1st pass", "ax 3d dyn 1st pass",
"Ph1/ax dynamic", "Ph1/ax 3d dyn" and GE's "Ph1/Ax Vibrant MultiPhase".
Pre-contrast, segmentation and single post-contrast T1 series are excluded.

Writes the patient -> series mapping used by step 2.

    python imaging/step1_identify_series.py
"""

import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg

cfg.ensure_dirs()

TIER1_PATTERNS = [
    r"ax\s*(3d\s*)?dyn(amic)?\s*1st\s*pass",
    r"ph1[\s/]*ax\s*(3d\s*)?dyn(amic)?\b",
    r"ph1[\s/]*ax\s*vibrant\s*multiphase",
]
TIER2_PATTERNS = [
    r"^ax\s*(3d\s*)?dyn(amic)?$",
    r"^ax\s*vibrant\s*multiphase$",
]
EXCLUDE_PATTERNS = [
    r"^ax\s*(3d\s*)?t1\s*(non\s*fs)?$",
    r"segmentation",
    r"^ax\s*t1\s*\+c$",
]

tier1_re = re.compile("|".join(TIER1_PATTERNS), re.IGNORECASE)
tier2_re = re.compile("|".join(TIER2_PATTERNS), re.IGNORECASE)
exclude_re = re.compile("|".join(EXCLUDE_PATTERNS), re.IGNORECASE)


def find_series(patient_series):
    """Return (SeriesInstanceUID, tier) or (None, None)."""
    descriptions = patient_series["SeriesDescription"].fillna("")
    candidates = patient_series[~descriptions.str.contains(exclude_re, regex=True, na=False)]

    for pattern, tier in ((tier1_re, 1), (tier2_re, 2)):
        matches = candidates[candidates["SeriesDescription"].str.contains(
            pattern, na=False, regex=True)]
        if len(matches):
            if "SeriesNumber" in matches.columns:
                matches = matches.sort_values("SeriesNumber")
            return matches.iloc[0]["SeriesInstanceUID"], tier

    return None, None


print("Querying imaging metadata...")
from idc_index import index

client = index.IDCClient()
series_metadata = client.sql_query("""
    SELECT * FROM index
    WHERE collection_id = 'duke_breast_cancer_mri'
""")
print(f"  {len(series_metadata)} series, {series_metadata['PatientID'].nunique()} patients")

matched, tiers, unmatched = {}, {}, []
for patient_id, group in series_metadata.groupby("PatientID"):
    uid, tier = find_series(group)
    if uid is not None:
        matched[patient_id] = uid
        tiers[patient_id] = tier
    else:
        unmatched.append(patient_id)

n_tier1 = sum(1 for t in tiers.values() if t == 1)
n_tier2 = sum(1 for t in tiers.values() if t == 2)

print(f"\nTier 1 (explicit phase tagging): {n_tier1}")
print(f"Tier 2 (generic dynamic series): {n_tier2}")
print(f"Matched: {len(matched)} / {series_metadata['PatientID'].nunique()}")
print(f"Unmatched: {len(unmatched)}")

if unmatched:
    print(f"\nSeries descriptions for the first few unmatched patients:")
    for pid in unmatched[:5]:
        descriptions = series_metadata.loc[
            series_metadata["PatientID"] == pid, "SeriesDescription"].unique()
        print(f"  {pid}: {list(descriptions)}")

if os.path.exists(cfg.PATIENT_TABLE_CSV):
    cohort = pd.read_csv(cfg.PATIENT_TABLE_CSV, low_memory=False)
    cohort = cohort.dropna(subset=["efs_time_days"])
    overlap = set(cohort[cfg.ID_COL].astype(str)) & set(matched)
    print(f"\nOverlap with the modelling cohort (N={len(cohort)}): {len(overlap)}")

output = os.path.join(cfg.OUTPUT_DIR, "patient_series_mapping.json")
with open(output, "w") as f:
    json.dump({"series": matched, "tiers": tiers, "unmatched": unmatched}, f, indent=2)
print(f"\nWrote {output}")
print("Next: imaging/step2_download.py")
