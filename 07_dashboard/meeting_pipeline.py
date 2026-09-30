# ============================================================
# meeting_pipeline.py
# Multi-Attribute Evidence Consistency for Verifiable MoM
# ============================================================

from pathlib import Path
import json
import os


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(
    "/content/drive/MyDrive/MTechIndProj/MoM_Project"
)

DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"

RAW_DIR = DATA_DIR / "raw"
VAD_DIR = DATA_DIR / "vad"
TRANSCRIPT_DIR = DATA_DIR / "transcripts"
EVIDENCE_DIR = DATA_DIR / "evidence"
EMBEDDING_DIR = DATA_DIR / "embeddings"
FAISS_DIR = DATA_DIR / "faiss"

ASR_MODEL_DIR = MODEL_DIR / "asr"
DIARIZATION_MODEL_DIR = MODEL_DIR / "diarization"
BGE_MODEL_DIR = MODEL_DIR / "bge"
LLM_MODEL_DIR = MODEL_DIR / "llm"


# ============================================================
# MODEL / PIPELINE CONFIGURATION
# ============================================================

VAD_THRESHOLD = 0.5
GAP_THRESHOLD_SECONDS = 1.0

WHISPER_MODEL_NAME = "small"
WHISPER_LANGUAGE = "en"

BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"

NLI_MODEL_NAME = (
    "cross-encoder/nli-deberta-v3-small"
)

DEFAULT_TOP_K = 10
DEFAULT_CONTEXT_WINDOW = 2


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def ensure_directories():

    directories = [
        DATA_DIR,
        MODEL_DIR,
        OUTPUT_DIR,
        RAW_DIR,
        VAD_DIR,
        TRANSCRIPT_DIR,
        EVIDENCE_DIR,
        EMBEDDING_DIR,
        FAISS_DIR,
        ASR_MODEL_DIR,
        DIARIZATION_MODEL_DIR,
        BGE_MODEL_DIR,
        LLM_MODEL_DIR,
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )


def save_json(data, path):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )


def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


ensure_directories()


# ============================================================
# SILERO VAD
# ============================================================

import torch


def load_silero_vad():

    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False
    )

    return model, utils


def run_silero_vad(
    audio_path,
    output_path=None,
    threshold=VAD_THRESHOLD
):

    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    model, utils = load_silero_vad()

    (
        get_speech_timestamps,
        save_audio,
        read_audio,
        VADIterator,
        collect_chunks
    ) = utils

    wav = read_audio(
        str(audio_path),
        sampling_rate=16000
    )

    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        threshold=threshold,
        sampling_rate=16000
    )

    segments = []

    for segment in speech_timestamps:

        segments.append({
            "start": round(
                segment["start"] / 16000,
                3
            ),
            "end": round(
                segment["end"] / 16000,
                3
            )
        })

    result = {
        "audio_path": str(audio_path),
        "sampling_rate": 16000,
        "threshold": threshold,
        "num_segments": len(segments),
        "segments": segments
    }

    if output_path is not None:
        save_json(
            result,
            output_path
        )

    return result



# ============================================================
# WHISPERX ASR + WORD-LEVEL ALIGNMENT
# ============================================================

def load_whisperx_model(model_name=WHISPER_MODEL_NAME, device=None):
    import torch
    import whisperx

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    compute_type = "float16" if device == "cuda" else "int8"

    model = whisperx.load_model(
        model_name,
        device=device,
        compute_type=compute_type,
        language="en"
    )

    print(f"WhisperX model loaded: {model_name}")
    print(f"Device: {device}")
    print(f"Compute type: {compute_type}")

    return model


def transcribe_with_whisperx(audio_path, model=None):
    import whisperx

    if model is None:
        model = load_whisperx_model()

    audio = whisperx.load_audio(str(audio_path))

    result = model.transcribe(
        audio,
        batch_size=8
    )

    print("WhisperX transcription completed.")
    print("Language:", result.get("language"))
    print("Transcript segments:", len(result.get("segments", [])))

    return result, audio


def align_whisperx_transcription(transcription_result, audio, device=None):
    import torch
    import whisperx

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    language = transcription_result.get("language", "en")

    align_model, metadata = whisperx.load_align_model(
        language_code=language,
        device=device
    )

    aligned_result = whisperx.align(
        transcription_result["segments"],
        align_model,
        metadata,
        audio,
        device,
        return_char_alignments=False
    )

    print("WhisperX word-level alignment completed.")
    print(
        "Aligned segments:",
        len(aligned_result.get("segments", []))
    )

    return aligned_result


def run_whisperx_pipeline(audio_path, output_path=None):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_whisperx_model(
        model_name=WHISPER_MODEL_NAME,
        device=device
    )

    transcription_result, audio = transcribe_with_whisperx(
        audio_path,
        model=model
    )

    aligned_result = align_whisperx_transcription(
        transcription_result,
        audio,
        device=device
    )

    if output_path is not None:
        save_json(aligned_result, output_path)
        print(f"WhisperX output saved to: {output_path}")

    return aligned_result



# ============================================================
# Pyannote Speaker Diarization
# ============================================================

def load_pyannote_pipeline(hf_token=None):
    """Load Pyannote speaker diarization pipeline."""
    import os
    import torch
    from pyannote.audio import Pipeline

    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    pipeline.to(device)

    return pipeline


def run_pyannote_diarization(
    audio_path,
    output_path=None,
    hf_token=None
):
    """
    Run Pyannote speaker diarization using
    an in-memory waveform.

    Returns:
        Dictionary containing diarization segments
        and detected speakers.
    """

    import json
    import torch
    import soundfile as sf

    pipeline = load_pyannote_pipeline(hf_token)

    waveform, sample_rate = sf.read(str(audio_path))

    # Convert audio to [channels, time].
    if waveform.ndim == 1:
        waveform = waveform[None, :]
    else:
        waveform = waveform.T

    waveform = torch.tensor(
        waveform,
        dtype=torch.float32
    )

    diarization_output = pipeline({
        "waveform": waveform,
        "sample_rate": sample_rate
    })

    # Pyannote Audio 4.x returns DiarizeOutput.
    # Exclusive diarization provides one speaker
    # for each time region.
    annotation = (
        diarization_output.exclusive_speaker_diarization
    )

    segments = []

    for turn, _, speaker in annotation.itertracks(
        yield_label=True
    ):
        segments.append({
            "start": round(float(turn.start), 3),
            "end": round(float(turn.end), 3),
            "speaker": speaker
        })

    result = {
        "audio_path": str(audio_path),
        "sample_rate": sample_rate,
        "num_segments": len(segments),
        "speakers": sorted(
            list(set(
                item["speaker"] for item in segments
            ))
        ),
        "segments": segments
    }

    if output_path is not None:
        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                result,
                f,
                indent=2,
                ensure_ascii=False
            )

        print(
            f"Pyannote output saved to: {output_path}"
        )

    return result



# ============================================================
# Speaker–Word Alignment
# ============================================================

def calculate_temporal_overlap(
    word_start,
    word_end,
    speaker_start,
    speaker_end
):
    """Calculate temporal overlap between a word and speaker segment."""

    overlap_start = max(word_start, speaker_start)
    overlap_end = min(word_end, speaker_end)

    return max(0.0, overlap_end - overlap_start)


def assign_speaker_to_word(
    word_start,
    word_end,
    diarization_segments
):
    """Assign the speaker with the largest temporal overlap."""

    best_speaker = None
    best_overlap = 0.0

    for segment in diarization_segments:

        overlap = calculate_temporal_overlap(
            word_start,
            word_end,
            segment["start"],
            segment["end"]
        )

        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = segment["speaker"]

    return best_speaker, round(best_overlap, 3)


def align_words_with_speakers(
    whisper_data,
    pyannote_data
):
    """
    Align WhisperX word-level timestamps with Pyannote
    speaker diarization.
    """

    words = whisper_data.get("word_segments", [])

    if not words:
        for segment in whisper_data.get("segments", []):
            words.extend(segment.get("words", []))

    diarization_segments = pyannote_data.get(
        "segments",
        []
    )

    aligned_words = []

    for word in words:

        if "start" not in word or "end" not in word:
            continue

        speaker, overlap = assign_speaker_to_word(
            float(word["start"]),
            float(word["end"]),
            diarization_segments
        )

        aligned_words.append({
            "word": word["word"],
            "start": round(float(word["start"]), 3),
            "end": round(float(word["end"]), 3),
            "score": word.get("score"),
            "speaker": speaker,
            "speaker_overlap": overlap
        })

    return aligned_words


def group_words_into_speaker_utterances(
    aligned_words
):
    """Group consecutive speaker-attributed words into utterances."""

    if not aligned_words:
        return []

    utterances = []

    current_speaker = aligned_words[0]["speaker"]
    current_words = [aligned_words[0]]

    for word in aligned_words[1:]:

        if word["speaker"] == current_speaker:
            current_words.append(word)

        else:
            utterances.append({
                "speaker": current_speaker,
                "start": current_words[0]["start"],
                "end": current_words[-1]["end"],
                "text": " ".join(
                    item["word"]
                    for item in current_words
                ),
                "words": current_words
            })

            current_speaker = word["speaker"]
            current_words = [word]

    utterances.append({
        "speaker": current_speaker,
        "start": current_words[0]["start"],
        "end": current_words[-1]["end"],
        "text": " ".join(
            item["word"]
            for item in current_words
        ),
        "words": current_words
    })

    return utterances


def run_speaker_word_alignment(
    whisper_data,
    pyannote_data,
    output_path=None
):
    """
    Run complete speaker-word alignment.

    Returns:
        Dictionary containing aligned words and
        speaker-attributed utterances.
    """

    import json

    aligned_words = align_words_with_speakers(
        whisper_data,
        pyannote_data
    )

    utterances = group_words_into_speaker_utterances(
        aligned_words
    )

    result = {
        "num_words": len(aligned_words),
        "num_utterances": len(utterances),
        "speakers": sorted(
            list(set(
                word["speaker"]
                for word in aligned_words
                if word["speaker"] is not None
            ))
        ),
        "words": aligned_words,
        "utterances": utterances
    }

    if output_path is not None:

        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                result,
                f,
                indent=2,
                ensure_ascii=False
            )

        print(
            f"Speaker-word alignment saved to: {output_path}"
        )

    return result
