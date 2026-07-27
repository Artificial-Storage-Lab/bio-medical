"""
Imaging step 5 -- extract MedImageInsight embeddings.

Converts each cropped tumour region to a normalised 512x512 PNG and runs it
through MedImageInsight's image encoder, producing a 1024-dimensional vector
per patient. The model is frozen; nothing is trained here.

MedImageInsight is a vision transformer (DaViT), not a CNN, trained
contrastively against medical text. Only the image half is used.

This script must run inside the cloned MedImageInsight repository, using its
environment:

    git clone https://huggingface.co/lion-ai/MedImageInsights
    cd MedImageInsights
    uv sync
    uv run example.py                    # confirm the model loads
    cp /path/to/this/file .
    uv run python step5_extract_embeddings.py

Set CROP_DIR and OUTPUT_CSV below to absolute paths, since the working
directory is the model repository rather than this project.
"""

import base64
import glob
import io
import os

import numpy as np
import pandas as pd
from PIL import Image

# This runs from the MedImageInsight repository, so it cannot import the
# project's config module. It reads the same environment variables instead --
# source your .env before running, or set the paths directly below.
CROP_DIR = os.environ.get("DUKE_CROP_DIR") or os.path.join(
    os.environ.get("DUKE_DATA_DIR", "./data"), "crops")
OUTPUT_CSV = os.path.join(
    os.environ.get("DUKE_DATA_DIR", "./data"), "medimageinsight_embeddings.csv")
BATCH_SIZE = 32

if not os.path.isdir(CROP_DIR):
    raise SystemExit(
        f"Crop directory not found: {CROP_DIR}\n"
        "Set DUKE_CROP_DIR (or DUKE_DATA_DIR) to point at the output of "
        "imaging/step3_crop.py."
    )

from medimageinsightmodel import MedImageInsight


def crop_to_png_bytes(path):
    """Load a crop, normalise to 8-bit, resize to 512x512, return PNG bytes.

    MRI intensities have no fixed scale, so each crop is min-max normalised
    individually.
    """
    arr = np.load(path).astype(np.float32)
    lo, hi = arr.min(), arr.max()
    arr = (arr - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(arr)

    img = Image.fromarray(arr.astype(np.uint8), mode="L").resize((512, 512))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


print("Loading MedImageInsight...")
classifier = MedImageInsight(
    model_dir="2024.09.27",
    vision_model_name="medimageinsigt-v1.0.0.pt",
    language_model_name="language_model.pth",
)
classifier.load_model()
print("Model loaded.")

crop_files = sorted(glob.glob(os.path.join(CROP_DIR, "*.npy")))
print(f"Found {len(crop_files)} crops")

embeddings, failures = {}, []

for start in range(0, len(crop_files), BATCH_SIZE):
    batch_files = crop_files[start:start + BATCH_SIZE]
    encoded, patient_ids = [], []

    for path in batch_files:
        patient_id = os.path.splitext(os.path.basename(path))[0]
        try:
            encoded.append(base64.encodebytes(crop_to_png_bytes(path)).decode("utf-8"))
            patient_ids.append(patient_id)
        except Exception as exc:
            failures.append((patient_id, f"png conversion: {exc}"))

    if not encoded:
        continue

    try:
        result = classifier.encode(images=encoded)
        for pid, vec in zip(patient_ids, result["image_embeddings"]):
            embeddings[pid] = np.asarray(vec, dtype=np.float32)
    except Exception:
        # fall back to one at a time so a single bad crop cannot lose the batch
        for pid, image in zip(patient_ids, encoded):
            try:
                single = classifier.encode(images=[image])
                embeddings[pid] = np.asarray(single["image_embeddings"][0], dtype=np.float32)
            except Exception as exc:
                failures.append((pid, f"encoding: {exc}"))

    print(f"  {min(start + BATCH_SIZE, len(crop_files))} / {len(crop_files)}")

print(f"\nEmbedded: {len(embeddings)} / {len(crop_files)}")
if failures:
    print(f"Failed: {len(failures)}")
    for pid, message in failures[:5]:
        print(f"  {pid}: {message}")

if embeddings:
    dim = len(next(iter(embeddings.values())))
    print(f"Embedding dimension: {dim}")

    rows = []
    for pid, vec in embeddings.items():
        row = {"dicom_patient_id": pid}
        row.update({f"medimg_emb_{i}": float(vec[i]) for i in range(dim)})
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {OUTPUT_CSV}  ({df.shape[0]} x {df.shape[1]})")
