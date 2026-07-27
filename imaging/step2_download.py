"""
Imaging step 2 -- download the identified series.

One series per patient rather than all ~5.6, and only for patients in the
modelling cohort. Roughly 56 GB.

Runs a size estimate first. Pass --download to actually fetch.

    python imaging/step2_download.py             # estimate only
    python imaging/step2_download.py --download  # download

The download takes hours. On macOS, run `caffeinate -i` in another terminal to
stop the machine sleeping partway through.
"""

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg

mapping_path = os.path.join(cfg.OUTPUT_DIR, "patient_series_mapping.json")
if not os.path.exists(mapping_path):
    raise SystemExit("Run imaging/step1_identify_series.py first.")

with open(mapping_path) as f:
    mapping = json.load(f)["series"]

cohort = pd.read_csv(cfg.PATIENT_TABLE_CSV, low_memory=False)
cohort = cohort.dropna(subset=["efs_time_days", "efs_event"])
cohort_ids = set(cohort[cfg.ID_COL].astype(str))

series_uids = [uid for pid, uid in mapping.items() if pid in cohort_ids]
patient_ids = [pid for pid in mapping if pid in cohort_ids]
print(f"Patients to download: {len(series_uids)}")

from idc_index import index

client = index.IDCClient()

print("\nEstimating download size...")
client.download_dicom_series(
    seriesInstanceUID=series_uids,
    downloadDir=cfg.DICOM_DOWNLOAD_DIR,
    dry_run=True,
)

if "--download" not in sys.argv:
    print("\nEstimate only. Re-run with --download to fetch the files.")
    raise SystemExit(0)

print(f"\nDownloading to {cfg.DICOM_DOWNLOAD_DIR}...")
client.download_dicom_series(
    seriesInstanceUID=series_uids,
    downloadDir=cfg.DICOM_DOWNLOAD_DIR,
    dry_run=False,
    show_progress_bar=True,
    dirTemplate="%PatientID/%SeriesInstanceUID",
)

downloaded = [d for d in os.listdir(cfg.DICOM_DOWNLOAD_DIR)
              if os.path.isdir(os.path.join(cfg.DICOM_DOWNLOAD_DIR, d))]
print(f"\nDownloaded: {len(downloaded)} / {len(patient_ids)} patients")
missing = set(patient_ids) - set(downloaded)
if missing:
    print(f"Missing {len(missing)}: {sorted(missing)[:10]}")

print("\nNext: imaging/step3_crop.py")
