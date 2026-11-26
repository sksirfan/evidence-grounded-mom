"""Central project paths for the MoM application.

Code stays in the GitHub clone. Meeting data and generated outputs are read
from the shared Google Drive project folder when it is available.
Set MOM_PROJECT_ROOT to override automatic discovery.
"""

import os
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent
_explicit_root = os.environ.get("MOM_PROJECT_ROOT", "").strip()

_mac_drive_candidates = sorted(
    (Path.home() / "Library/CloudStorage").glob(
        "GoogleDrive-*/My Drive/MTechIndProj/MoM_Project"
    )
)
_legacy_mac_drive = Path.home() / "Google Drive/My Drive/MTechIndProj/MoM_Project"
_drive_candidates = [
    Path("/content/drive/MyDrive/MTechIndProj/MoM_Project"),
    Path(r"G:\My Drive\MTechIndProj\MoM_Project"),
    *_mac_drive_candidates,
    _legacy_mac_drive,
]

if _explicit_root:
    PROJECT_DIR = Path(_explicit_root).expanduser().resolve()
else:
    _drive_root = next((p for p in _drive_candidates if p.is_dir()), None)
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

EVALUATION_DIR = PROJECT_DIR / "evaluation_results"

# Importing config must work in the lightweight Intel Mac environment.
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"
