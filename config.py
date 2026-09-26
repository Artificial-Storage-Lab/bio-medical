"""
Configuration -- all paths and analysis settings in one place.

Paths default to a `data/` folder alongside the code. If your files live
elsewhere, set the environment variables below instead of editing this file:

    export DUKE_DATA_DIR="/path/to/your/data"        # small files
    export DUKE_DICOM_DIR="/Volumes/DRIVE/dicom"     # ~56 GB download
    export DUKE_CROP_DIR="/path/to/crops"            # extracted crops

Only DUKE_DATA_DIR is needed for the tier analyses. The other two matter only
if you re-run the imaging extraction.
"""

import os

# Input data (not included in the repository -- see README)
DATA_DIR = os.environ.get("DUKE_DATA_DIR", "./data")

# Downloaded from the TCIA collection page, DOI 10.7937/TCIA.e3sv-re93
IMAGING_FEATURES_XLSX = os.path.join(DATA_DIR, "Imaging_Features.xlsx")
ANNOTATION_BOXES_XLSX = os.path.join(DATA_DIR, "Annotation_Boxes.xlsx")

# Produced by imaging/step5_extract_embeddings.py
EMBEDDINGS_CSV = os.path.join(DATA_DIR, "medimageinsight_embeddings.csv")

# Imaging working directories. The DICOM download is roughly 56 GB and often
# lives on external storage, so it is worth pointing elsewhere.
DICOM_DOWNLOAD_DIR = os.environ.get("DUKE_DICOM_DIR", os.path.join(DATA_DIR, "dicom_series"))
CROP_DIR = os.environ.get("DUKE_CROP_DIR", os.path.join(DATA_DIR, "crops"))

# Output
OUTPUT_DIR = os.environ.get("DUKE_OUTPUT_DIR", "./outputs")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")     # per-fold scores as JSON
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")

# Produced by 01_build_patient_table.py
PATIENT_TABLE_CSV = os.path.join(OUTPUT_DIR, "patient_level_table.csv")

# Analysis settings
RANDOM_STATE = 42
OUTER_FOLDS = 5
INNER_FOLDS = 3

RADIOMICS_COVERAGE_MIN = 0.85     # keep radiomics columns with at least 85% non-missing values
CORRELATION_THRESHOLD = 0.9       # drop one of any pair above this
UNIVARIATE_TOP_K = 100            # candidates surviving to the LASSO stage
LASSO_TARGET_RANGE = (20, 50)     # target sparsity for LASSO-Cox
PCA_N_COMPONENTS = 20             # embedding dimensions retained per fold
BOOTSTRAP_N = 2000                # patient-level bootstrap replicates

# Column names in the IDC clinical table
ID_COL = "dicom_patient_id"
N_STAGE_COL = "tumor_characteristics_stagingnodesnx_replaced_by__1n"
MRI_NODE_READ_COL = "mri_findings_lymphadenopathy_or_suspicious_nodes"

OUTCOME_COLS = ("efs_time_days", "efs_event", "os_time_days", "os_event")

# Strings that mean "missing" in this dataset. Several columns encode
# missingness as text rather than blanks, which silently breaks numeric
# operations if not cleaned first.
PLACEHOLDER_VALUES = {"", " ", "NA", "N/A", "NC", "NP", "NR",
                      "Unknown", "unknown", "nan", "NaN"}


def ensure_dirs():
    for d in (OUTPUT_DIR, RESULTS_DIR, FIGURE_DIR):
        os.makedirs(d, exist_ok=True)


def check_inputs(require=()):
    """Report which expected input files are present.

    Pass names to require, e.g. check_inputs(("IMAGING_FEATURES_XLSX",)), to
    fail early with a clear message rather than partway through an analysis.
    """
    paths = {
        "PATIENT_TABLE_CSV": PATIENT_TABLE_CSV,
        "IMAGING_FEATURES_XLSX": IMAGING_FEATURES_XLSX,
        "ANNOTATION_BOXES_XLSX": ANNOTATION_BOXES_XLSX,
        "EMBEDDINGS_CSV": EMBEDDINGS_CSV,
    }
    missing = [name for name in require if not os.path.exists(paths[name])]
    if missing:
        lines = [f"  {name}: {paths[name]}" for name in missing]
        raise FileNotFoundError(
            "Required input files not found:\n" + "\n".join(lines)
            + f"\n\nDATA_DIR is currently '{DATA_DIR}'. Either place the files "
              "there, or set DUKE_DATA_DIR to where they live:\n"
              "    export DUKE_DATA_DIR=/path/to/your/data"
        )