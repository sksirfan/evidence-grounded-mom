"""Central project paths for the MoM application.

Project data and generated outputs live in Google Drive when the configured
Drive project folder exists. Code remains in the GitHub clone.

Override the automatic selection by setting MOM_PROJECT_ROOT to an absolute
path before launching Streamlit or importing the pipeline.
"""

import os
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent

# Explicit override takes precedence, which is useful on either laptop.
_explicit_root = os.environ.get("MOM_PROJECT_ROOT", "").strip()

# Use the shared Google Drive folder only when that exact project directory
# exists. Merely having Drive mounted is not enough.
_drive_candidates = [
    Path("/content/drive/MyDrive/MTechIndProj/MoM_Project"),
    Path(r"G:\My Drive\MTechIndProj\MoM_Project"),
]

if _explicit_root:
    PROJECT_DIR = Path(_explicit_root).expanduser().resolve()
else:
    _drive_root = next(
        (candidate for candidate in _drive_candidates if candidate.is_dir()),
        None,
    )
    PROJECT_DIR = (_drive_root or REPOSITORY_ROOT).resolve()

# Data directories
DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
AUDIO_DIR = DATA_DIR / "audio"
VAD_DIR = DATA_DIR / "vad"
TRANSCRIPT_DIR = DATA_DIR / "transcripts"
DIALOGUE_ACT_DIR = DATA_DIR / "dialogue_acts"
REFERENCE_DIR = DATA_DIR / "references"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EVIDENCE_DIR = DATA_DIR / "evidence"
EMBEDDING_DIR = DATA_DIR / "embeddings"
FAISS_DIR = DATA_DIR / "faiss"

# Model directories
MODEL_DIR = PROJECT_DIR / "models"
ASR_MODEL_DIR = MODEL_DIR / "asr"
DIARIZATION_MODEL_DIR = MODEL_DIR / "diarization"
DIALOGUE_MODEL_DIR = MODEL_DIR / "dialogue_act"
LLM_MODEL_DIR = MODEL_DIR / "llm"
BGE_MODEL_DIR = MODEL_DIR / "bge"

# Output directories
OUTPUT_DIR = PROJECT_DIR / "outputs"
MOM_OUTPUT_DIR = OUTPUT_DIR / "mom"
EVIDENCE_OUTPUT_DIR = OUTPUT_DIR / "evidence"
VERIFIED_OUTPUT_DIR = OUTPUT_DIR / "verified"
PDF_OUTPUT_DIR = OUTPUT_DIR / "pdf"
LIVE_OUTPUT_DIR = OUTPUT_DIR / "live"
TRANSCRIPT_OUTPUT_DIR = OUTPUT_DIR / "transcripts"
PHASE2_DAG_OUTPUT_DIR = OUTPUT_DIR / "phase2_dag"

# Evaluation
EVALUATION_DIR = PROJECT_DIR / "evaluation_results"

# Hardware detection is optional; importing path configuration should not
# fail merely because PyTorch is unavailable in a lightweight environment.
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"
