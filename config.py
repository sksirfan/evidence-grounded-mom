import os
import torch

# Main project directory
PROJECT_DIR = "/content/drive/MyDrive/MTechIndProj/MoM_Project"

# Data directories
RAW_DATA_DIR = os.path.join(PROJECT_DIR, "data", "raw")
AUDIO_DIR = os.path.join(PROJECT_DIR, "data", "audio")
VAD_DIR = os.path.join(PROJECT_DIR, "data", "vad")
TRANSCRIPT_DIR = os.path.join(PROJECT_DIR, "data", "transcripts")
DIALOGUE_ACT_DIR = os.path.join(PROJECT_DIR, "data", "dialogue_acts")
REFERENCE_DIR = os.path.join(PROJECT_DIR, "data", "references")
PROCESSED_DATA_DIR = os.path.join(PROJECT_DIR, "data", "processed")

# Model directories
ASR_MODEL_DIR = os.path.join(PROJECT_DIR, "models", "asr")
DIARIZATION_MODEL_DIR = os.path.join(PROJECT_DIR, "models", "diarization")
DIALOGUE_MODEL_DIR = os.path.join(PROJECT_DIR, "models", "dialogue_act")
LLM_MODEL_DIR = os.path.join(PROJECT_DIR, "models", "llm")

# Output directories
MOM_OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "mom")
EVIDENCE_OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "evidence")
VERIFIED_OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "verified")
PDF_OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "pdf")

# Evaluation
EVALUATION_DIR = os.path.join(PROJECT_DIR, "evaluation_results")

# Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
