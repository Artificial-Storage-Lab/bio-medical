"""
Imaging step 3 -- crop each series to the annotated tumour region.

Takes the middle slice of the annotated slice range and crops it to the
radiologist-drawn bounding box (Saha et al. 2018, 922 patients).

Slice ordering comes from the DICOM InstanceNumber tag rather than filenames:
IDC renames files on download, so the original numbering is not recoverable
from the filename.

Reads headers only when determining slice order and decodes pixel data for the
single target slice. Reading pixels for every slice would mean roughly 88,000
unnecessary decodes.

Writes one .npy per patient plus a contact sheet for visual checking.

    python imaging/step3_crop.py
"""

import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg

cfg.check_inputs(("ANNOTATION_BOXES_XLSX",))
os.makedirs(cfg.CROP_DIR, exist_ok=True)

boxes = pd.read_excel(cfg.ANNOTATION_BOXES_XLSX).set_index("Patient ID")
print(f"Loaded {len(boxes)} annotation boxes")

patients = [d for d in os.listdir(cfg.DICOM_DOWNLOAD_DIR)
            if os.path.isdir(os.path.join(cfg.DICOM_DOWNLOAD_DIR, d))]
print(f"Found {len(patients)} downloaded patients")

log = []

for patient_id in patients:
    if patient_id not in boxes.index:
        log.append({"patient_id": patient_id, "status": "no_annotation_box"})
        continue

    box = boxes.loc[patient_id]
    start_row, end_row = int(box["Start Row"]), int(box["End Row"])
    start_col, end_col = int(box["Start Column"]), int(box["End Column"])
    start_slice, end_slice = int(box["Start Slice"]), int(box["End Slice"])

    files = glob.glob(os.path.join(cfg.DICOM_DOWNLOAD_DIR, patient_id, "**", "*.dcm"),
                      recursive=True)
    if not files:
        log.append({"patient_id": patient_id, "status": "no_dicom_files"})
        continue

    # headers only -- fast
    slices = []
    for path in files:
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            slices.append((int(getattr(ds, "InstanceNumber", -1)), path))
        except Exception:
            continue

    if not slices:
        log.append({"patient_id": patient_id, "status": "unreadable"})
        continue

    slices.sort(key=lambda s: s[0])
    instance_numbers = [s[0] for s in slices]

    if start_slice < min(instance_numbers) or end_slice > max(instance_numbers):
        log.append({"patient_id": patient_id, "status": "slice_range_mismatch",
                    "annotated": [start_slice, end_slice],
                    "available": [min(instance_numbers), max(instance_numbers)]})
        continue

    target = (start_slice + end_slice) // 2
    _, chosen_path = min(slices, key=lambda s: abs(s[0] - target))

    ds = pydicom.dcmread(chosen_path)   # full read, one slice
    pixels = ds.pixel_array

    if end_row > pixels.shape[0] or end_col > pixels.shape[1]:
        log.append({"patient_id": patient_id, "status": "crop_out_of_bounds",
                    "box": [start_row, end_row, start_col, end_col],
                    "image_shape": list(pixels.shape)})
        continue

    crop = pixels[start_row:end_row, start_col:end_col]
    np.save(os.path.join(cfg.CROP_DIR, f"{patient_id}.npy"), crop)
    log.append({"patient_id": patient_id, "status": "success",
                "crop_shape": list(crop.shape)})

log_df = pd.DataFrame(log)
print(f"\n{'=' * 62}\nSUMMARY\n{'=' * 62}")
print(log_df["status"].value_counts())

successes = log_df[log_df["status"] == "success"]
print(f"\nCropped: {len(successes)} / {len(patients)}")

problems = log_df[log_df["status"].isin(["slice_range_mismatch", "crop_out_of_bounds"])]
if len(problems):
    print(f"\nAlignment problems ({len(problems)}):")
    print(problems.head(10).to_string(index=False))

with open(os.path.join(cfg.OUTPUT_DIR, "crop_log.json"), "w") as f:
    json.dump(log, f, indent=2)

# ------------------------------------------------------------
# Contact sheet -- inspect this before trusting the crops
# ------------------------------------------------------------
sample = successes["patient_id"].head(8).tolist()
if sample:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, pid in zip(axes.flatten(), sample):
        ax.imshow(np.load(os.path.join(cfg.CROP_DIR, f"{pid}.npy")), cmap="gray")
        ax.set_title(pid, fontsize=9)
        ax.axis("off")
    for ax in axes.flatten()[len(sample):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{cfg.FIGURE_DIR}/crop_examples.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nWrote {cfg.FIGURE_DIR}/crop_examples.png")
    print("Check it before continuing -- crops should show tissue with a")
    print("plausible enhancing lesion, not background or the wrong region.")

print("\nNext: imaging/step5_extract_embeddings.py (run inside the MedImageInsight repo)")
