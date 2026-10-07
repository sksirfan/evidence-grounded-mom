"""
Verifiable Meeting Minutes
Demo Environment Health Check

Run this BEFORE starting Streamlit.

The check verifies:
- Python
- NumPy / SciPy compatibility
- PyTorch / CUDA / GPU
- WhisperX
- Pyannote
- Hugging Face authentication
- BGE / Sentence Transformers
- FAISS
- Project pipeline import
"""

import os
import sys
import importlib
import importlib.metadata as md
from pathlib import Path


EXPECTED = {
    "numpy": "2.2.6",
    "scipy": "1.15.3",
    "whisperx": "3.8.6",
    "pyannote.audio": "4.0.7",
    "transformers": "4.57.6",
    "sentence-transformers": "5.7.0",
    "faiss-cpu": "1.15.1",
    "huggingface-hub": "0.36.2",
    "tokenizers": "0.22.2",
    "accelerate": "1.15.0",
    "pandas": "2.2.3",
    "scikit-learn": "1.6.1",
    "librosa": "0.11.0",
    "numba": "0.61.2",
    "soundfile": "0.14.0",
    "pillow": "11.3.0",
    "tqdm": "4.67.3",
    "streamlit": "1.65.0",
}


passed = True


def check(label, condition, detail=""):
    global passed

    if condition:
        print(f"✅ {label}" + (f" — {detail}" if detail else ""))
    else:
        print(f"❌ {label}" + (f" — {detail}" if detail else ""))
        passed = False


print()
print("=" * 60)
print("       VERIFIABLE MEETING MINUTES")
print("       DEMO ENVIRONMENT HEALTH CHECK")
print("=" * 60)
print()


# ---------------------------------------------------------
# Python
# ---------------------------------------------------------

print("SYSTEM")
print("-" * 60)

python_version = sys.version.split()[0]
print("Python:", python_version)

check(
    "Python version",
    sys.version_info >= (3, 10),
    python_version
)


# ---------------------------------------------------------
# NumPy / SciPy
# ---------------------------------------------------------

print()
print("NUMPY / SCIPY")
print("-" * 60)

try:
    import numpy
    import scipy

    print("NumPy:", numpy.__version__)
    print("SciPy:", scipy.__version__)

    check(
        "NumPy",
        numpy.__version__ == EXPECTED["numpy"],
        numpy.__version__
    )

    check(
        "SciPy",
        scipy.__version__ == EXPECTED["scipy"],
        scipy.__version__
    )

except Exception as e:
    check("NumPy / SciPy import", False, str(e))


# ---------------------------------------------------------
# PyTorch / CUDA
# ---------------------------------------------------------

print()
print("PYTORCH / GPU")
print("-" * 60)

try:
    import torch

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    check(
        "PyTorch",
        torch.__version__.startswith("2.8.0"),
        torch.__version__
    )

    check(
        "CUDA",
        torch.cuda.is_available(),
        "CUDA available" if torch.cuda.is_available() else "CUDA unavailable"
    )

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        check(
            "Tesla T4 GPU",
            "Tesla T4" in torch.cuda.get_device_name(0),
            torch.cuda.get_device_name(0)
        )

except Exception as e:
    check("PyTorch", False, str(e))


# ---------------------------------------------------------
# Python package versions
# ---------------------------------------------------------

print()
print("PROJECT PACKAGES")
print("-" * 60)

for package, expected_version in EXPECTED.items():
    try:
        actual = md.version(package)

        check(
            package,
            actual == expected_version,
            f"{actual} (expected {expected_version})"
        )

    except Exception as e:
        check(package, False, "not installed")


# ---------------------------------------------------------
# Pyannote import
# ---------------------------------------------------------

print()
print("PYANNOTE")
print("-" * 60)

try:
    from pyannote.audio import Pipeline
    check("pyannote.audio import", True)
except Exception as e:
    check("pyannote.audio import", False, str(e))


# ---------------------------------------------------------
# Hugging Face authentication
# ---------------------------------------------------------

print()
print("HUGGING FACE")
print("-" * 60)

try:
    from huggingface_hub import whoami

    token = os.environ.get("HF_TOKEN")

    check(
        "HF_TOKEN available",
        bool(token),
        "Environment variable loaded"
    )

    if token:
        try:
            info = whoami(token=token)

            check(
                "Hugging Face authentication",
                True,
                info.get("name", "authenticated")
            )

        except Exception as e:
            check(
                "Hugging Face authentication",
                False,
                str(e)
            )

except Exception as e:
    check("Hugging Face authentication", False, str(e))


# ---------------------------------------------------------
# BGE / Sentence Transformers
# ---------------------------------------------------------

print()
print("EVIDENCE RETRIEVAL")
print("-" * 60)

try:
    from sentence_transformers import SentenceTransformer
    check("Sentence Transformers import", True)

    model_name = "BAAI/bge-small-en-v1.5"
    print("BGE model:", model_name)

    check("BGE configuration", model_name == "BAAI/bge-small-en-v1.5")

except Exception as e:
    check("Sentence Transformers / BGE", False, str(e))


# ---------------------------------------------------------
# FAISS
# ---------------------------------------------------------

try:
    import faiss
    check(
        "FAISS import",
        True,
        f"FAISS {faiss.__version__}"
    )
except Exception as e:
    check("FAISS import", False, str(e))


# ---------------------------------------------------------
# Project pipeline
# ---------------------------------------------------------

print()
print("PROJECT PIPELINE")
print("-" * 60)

try:
    import meeting_pipeline

    required_functions = [
        "run_silero_vad",
        "run_whisperx_pipeline",
        "run_pyannote_diarization",
        "run_speaker_word_alignment",
        "filter_mom_candidates_v3",
        "generate_structured_mom",
        "build_evidence_retrieval_resources",
        "verify_multi_attribute_claim",
        "format_verified_claim_result",
    ]

    missing = [
        name
        for name in required_functions
        if not hasattr(meeting_pipeline, name)
    ]

    check(
        "meeting_pipeline import",
        len(missing) == 0,
        "All required functions available"
        if not missing
        else f"Missing: {missing}"
    )

except Exception as e:
    check("meeting_pipeline import", False, str(e))


# ---------------------------------------------------------
# Final result
# ---------------------------------------------------------

print()
print("=" * 60)

if passed:
    print("              ✅ DEMO ENVIRONMENT READY")
    print("=" * 60)
    print()
    print("Safe to start the Streamlit dashboard.")
else:
    print("              ❌ DEMO ENVIRONMENT NOT READY")
    print("=" * 60)
    print()
    print("Fix the failed checks before starting Streamlit.")

print()
