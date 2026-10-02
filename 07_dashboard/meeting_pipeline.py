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

PROJECT_DIR = Path(__file__).resolve().parent.parent

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




# ============================================================
# MEETING UNDERSTANDING — MoM CANDIDATE FILTERING V3
#
# Purpose:
# Select meeting-relevant utterances before structured
# MoM generation.
#
# Important:
# - This is candidate filtering, NOT final verification.
# - Explicit activity/icebreaker content is rejected.
# - Generic meeting introduction is rejected.
# - Project discussion, proposals, actions and decisions
#   are retained.
# ============================================================

import re


NON_MOM_ACTIVITY_TERMS = [
    "draw",
    "drawing",
    "whiteboard",
    "pen",
    "favourite animal",
    "favorite animal",
    "tiger",
    "dog",
    "cat",
    "eagle",
    "bat",
    "seagull",
    "guess",
    "drawing skill",
    "sum up the characteristics of that animal",
]


MEETING_INTRO_TERMS = [
    "first meeting",
    "hello everybody",
    "our agenda",
    "get to know each other",
    "feel more comfortable",
    "tool training",
    "we've got 25 minutes",
    "we're going to practice",
    "space for everyone else",
]


MOM_DECISION_PATTERNS = [
    r"\bwe decided\b",
    r"\bwe agreed\b",
    r"\bagreed to\b",
    r"\blet's use\b",
    r"\blet's go with\b",
    r"\bwe will use\b",
    r"\bwe'll use\b",
]


MOM_ACTION_PATTERNS = [
    r"\bwe need to\b",
    r"\bwe have to\b",
    r"\bwe should\b",
    r"\bwe must\b",
    r"\bneed to be\b",
    r"\bhas to be\b",
    r"\bhave to be\b",
    r"\bassigned to\b",
    r"\bresponsible for\b",
    r"\bdeadline\b",
    r"\bdue\b",
]


MOM_FORMAL_MEETING_PATTERNS = [
    r"\bbe it resolved\b",
    r"\bresolved that\b",
    r"\bit is resolved\b",
    r"\bapproved\b",
    r"\bapproval\b",
    r"\bunanimous\b",
    r"\bso carried\b",
    r"\bthose in support\b",
    r"\bmover\b",
    r"\bseconder\b",
    r"\bdelegated to\b",
    r"\bappointed\b",
    r"\badopted\b",
    r"\bshall be\b",
    r"\bis hereby approved\b",
]


MOM_PROPOSAL_PATTERNS = [
    r"\bi suggest\b",
    r"\bi propose\b",
    r"\bwe could\b",
    r"\bwe can\b",
    r"\bhow about\b",
    r"\bwhat if\b",
    r"\bmight be useful\b",
    r"\bcould have\b",
    r"\bcould be\b",
]


MOM_PROJECT_TERMS = [
    "remote control",
    "product",
    "design",
    "functional",
    "conceptual",
    "feature",
    "button",
    "screen",
    "lcd",
    "menu",
    "display",
    "plastic",
    "metal",
    "material",
    "price",
    "cost",
    "production",
    "market",
    "customer",
    "user",
    "user-friendly",
    "accessible",
    "usable",
    "international",
    "target",
    "recording",
    "volume",
    "channel",
    "programming",
]


def _mom_has_term(text, terms):
    text_lower = str(text).lower()

    return any(
        term.lower() in text_lower
        for term in terms
    )


def _mom_matches_pattern(text, patterns):
    return any(
        re.search(
            pattern,
            str(text),
            flags=re.IGNORECASE
        )
        for pattern in patterns
    )


def _mom_count_project_terms(text):
    text_lower = str(text).lower()

    return sum(
        1
        for term in MOM_PROJECT_TERMS
        if term.lower() in text_lower
    )


def classify_mom_candidate_v3(utterance):

    text = re.sub(
        r"\s+",
        " ",
        str(utterance.get("text", ""))
    ).strip()

    speaker = (
        utterance.get("speaker")
        or "UNKNOWN"
    )

    result = {
        **utterance,
        "speaker": speaker,
        "text": text,
        "mom_candidate": False,
        "candidate_type": None,
        "candidate_score": 0,
        "candidate_reasons": [],
    }

    if len(text.split()) < 6:

        result["candidate_reasons"] = [
            "too_short"
        ]

        return result

    activity_present = _mom_has_term(
        text,
        NON_MOM_ACTIVITY_TERMS
    )

    project_count = _mom_count_project_terms(
        text
    )

    if activity_present and project_count == 0:

        result["candidate_reasons"] = [
            "non_mom_activity"
        ]

        return result

    intro_present = _mom_has_term(
        text,
        MEETING_INTRO_TERMS
    )

    decision = _mom_matches_pattern(
        text,
        MOM_DECISION_PATTERNS
    )

    action = _mom_matches_pattern(
        text,
        MOM_ACTION_PATTERNS
    )

    proposal = _mom_matches_pattern(
        text,
        MOM_PROPOSAL_PATTERNS
    )

    formal_meeting = _mom_matches_pattern(
        text,
        MOM_FORMAL_MEETING_PATTERNS
    )

    if intro_present and not (
        decision
        or action
        or proposal
        or formal_meeting
    ):

        result["candidate_reasons"] = [
            "meeting_introduction"
        ]

        return result

    candidate_type = None
    score = 0
    reasons = []

    if decision:

        candidate_type = "DECISION"
        score = 5
        reasons.append("decision_signal")

    elif formal_meeting:

        candidate_type = "DECISION"
        score = 5
        reasons.append(
            "formal_meeting_decision_signal"
        )

    elif action:

        candidate_type = "ACTION"
        score = 4
        reasons.append("action_signal")

    elif proposal:

        candidate_type = "PROPOSAL"
        score = 4
        reasons.append("proposal_signal")

    elif project_count >= 2:

        candidate_type = "DISCUSSION_POINT"
        score = 3
        reasons.append(
            "substantive_project_content"
        )

    elif project_count == 1:

        candidate_type = "PROJECT_CONTEXT"
        score = 2
        reasons.append(
            "project_context"
        )

    else:

        result["candidate_reasons"] = [
            "no_mom_signal"
        ]

        return result

    if len(text.split()) >= 25:

        score += 1
        reasons.append(
            "substantive_length"
        )

    if score >= 3:

        result["mom_candidate"] = True
        result["candidate_type"] = candidate_type
        result["candidate_score"] = score
        result["candidate_reasons"] = reasons

    else:

        result["candidate_reasons"] = [
            "insufficient_mom_signal"
        ]

    return result


def filter_mom_candidates_v3(
    utterances,
    return_all=False
):
    """
    Apply V3 meeting-relevance filtering.

    Parameters
    ----------
    utterances : list
        Speaker-attributed meeting utterances.

    return_all : bool
        If True, return both filtered and rejected items.

    Returns
    -------
    list or dict
        Final MoM candidates, optionally with all results.
    """

    classified = [
        classify_mom_candidate_v3(
            utterance
        )
        for utterance in utterances
    ]

    candidates = [
        item
        for item in classified
        if item["mom_candidate"]
    ]

    if return_all:

        return {
            "candidates": candidates,
            "all_items": classified,
        }

    return candidates


# ============================================================
# STRUCTURED MoM GENERATION INTERFACE
#
# BART = baseline
# Qwen = final generator
# ============================================================

GENERATOR_CONFIG = {
    "bart": {
        "model_name": "facebook/bart-large-cnn",
        "role": "baseline",
    },

    "qwen": {
        "model_name": "Qwen/Qwen2.5-3B-Instruct",
        "role": "final",
    },
}


def create_structured_claim(
    text,
    speaker=None,
    start=None,
    end=None,
    event_type=None,
    source_utterances=None,
):
    """
    Standard claim representation used by downstream
    evidence retrieval and verification.
    """

    return {
        "claim": str(text).strip(),
        "speaker": speaker,
        "start": start,
        "end": end,
        "event_type": event_type,
        "source_utterances": (
            source_utterances
            if source_utterances is not None
            else []
        ),
        "generation_model": None,
        "verification": None,
    }


def prepare_candidate_context(candidates):
    """
    Convert candidate utterances into a compact context
    for a generation model.
    """

    context_blocks = []

    for idx, candidate in enumerate(
        candidates,
        start=1
    ):

        text = str(
            candidate.get("text", "")
        ).strip()

        if not text:
            continue

        speaker = (
            candidate.get("speaker")
            or "UNKNOWN"
        )

        start = candidate.get(
            "start",
            None
        )

        end = candidate.get(
            "end",
            None
        )

        candidate_type = (
            candidate.get(
                "candidate_type"
            )
            or "DISCUSSION_POINT"
        )

        context_blocks.append(
            f"[Candidate {idx}]\n"
            f"Speaker: {speaker}\n"
            f"Time: {start} - {end}\n"
            f"Type: {candidate_type}\n"
            f"Text: {text}"
        )

    return "\n\n".join(
        context_blocks
    )


def generate_mom_bart(
    candidates,
    model=None,
    tokenizer=None,
):
    """
    BART baseline generation.

    The model is loaded only when explicitly supplied.
    """

    context = prepare_candidate_context(
        candidates
    )

    if not context:
        return []

    if model is None or tokenizer is None:

        return {
            "generator": "bart",
            "model_name": GENERATOR_CONFIG[
                "bart"
            ]["model_name"],
            "status": "context_ready",
            "context": context,
        }

    inputs = tokenizer(
        context,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )

    # Keep tensors on the same device as the model.
    device = next(
        model.parameters()
    ).device

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    output_ids = model.generate(
        **inputs,
        max_length=256,
        min_length=40,
        num_beams=4,
        early_stopping=True,
    )

    generated_text = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
    )

    return {
        "generator": "bart",
        "model_name": GENERATOR_CONFIG[
            "bart"
        ]["model_name"],
        "status": "generated",
        "output": generated_text,
        "text": generated_text,
        "context": context,
    }


def generate_mom_qwen(
    candidates,
    model=None,
    tokenizer=None,
):
    """
    Candidate-level structured MoM generation.

    Each candidate is processed independently so that
    long conversational context from one candidate does
    not interfere with extraction from another.

    Qwen generates only the claim text.

    Speaker, event_type, start, and end are inherited
    from the original candidate to preserve provenance.
    """

    if not candidates:
        return {
            "generator": "qwen",
            "model_name": GENERATOR_CONFIG[
                "qwen"
            ]["model_name"],
            "status": "empty_context",
            "claims": [],
            "context": "",
        }

    if model is None or tokenizer is None:
        return {
            "generator": "qwen",
            "model_name": GENERATOR_CONFIG[
                "qwen"
            ]["model_name"],
            "status": "context_ready",
            "claims": [],
            "context": prepare_candidate_context(
                candidates
            ),
        }

    import json
    import re
    import torch

    generated_claims = []
    raw_outputs = []

    for idx, candidate in enumerate(
        candidates,
        start=1
    ):

        candidate_text = str(
            candidate.get("text", "")
        ).strip()

        if not candidate_text:
            continue

        candidate_type = (
            candidate.get(
                "candidate_type"
            )
            or "DISCUSSION_POINT"
        )

        candidate_speaker = (
            candidate.get("speaker")
            or "UNKNOWN"
        )

        candidate_start = candidate.get(
            "start"
        )

        candidate_end = candidate.get(
            "end"
        )

        # ----------------------------------------------------
        # Candidate-specific extraction prompt
        # ----------------------------------------------------

        prompt = f"""
Extract the single most important meeting-minutes
claim from the meeting utterance below.

SOURCE UTTERANCE:
{candidate_text}

SOURCE TYPE:
{candidate_type}

Rules:
1. Use ONLY information explicitly stated in the source.
2. Do NOT add information.
3. Do NOT copy the entire utterance.
4. Remove greetings, acknowledgements, filler,
   procedural language, agenda transitions, and
   unrelated conversation.
5. Preserve important factual details such as names,
   dates, amounts, locations, responsibilities,
   decisions, and proposals.
6. Produce ONE concise claim.
7. If the source contains a proposal that was not
   explicitly accepted, describe it as a proposal.
8. Do not convert a proposal into a decision.
9. Do not invent an outcome.
10. Do not include commentary or explanation.

Return ONLY valid JSON in exactly this format:

{{
  "claim": "concise factual meeting claim"
}}
"""

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise meeting-minutes "
                    "claim extraction system. "
                    "Extract facts from the supplied "
                    "utterance and return JSON only."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        formatted_prompt = (
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )

        device = next(
            model.parameters()
        ).device

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            output_ids = model.generate(
                **inputs,
                max_new_tokens=160,
                do_sample=False,
                pad_token_id=(
                    tokenizer.eos_token_id
                ),
            )

        input_length = inputs[
            "input_ids"
        ].shape[1]

        generated_ids = output_ids[
            0,
            input_length:
        ]

        raw_output = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

        raw_outputs.append({
            "candidate_index": idx,
            "raw_output": raw_output,
        })

        clean_output = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            raw_output,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()

        # ----------------------------------------------------
        # Parse JSON
        # ----------------------------------------------------

        try:
            parsed = json.loads(
                clean_output
            )

        except json.JSONDecodeError:
            continue

        if not isinstance(
            parsed,
            dict
        ):
            continue

        claim_text = str(
            parsed.get(
                "claim",
                ""
            )
        ).strip()

        if not claim_text:
            continue

        # ----------------------------------------------------
        # Preserve source provenance
        # ----------------------------------------------------

        generated_claims.append({
            "claim": claim_text,
            "speaker": candidate_speaker,
            "event_type": candidate_type,
            "start": candidate_start,
            "end": candidate_end,
            "source_candidate": {
                "speaker": candidate_speaker,
                "start": candidate_start,
                "end": candidate_end,
                "candidate_type": candidate_type,
                "candidate_score": candidate.get(
                    "candidate_score"
                ),
                "candidate_reasons": candidate.get(
                    "candidate_reasons",
                    []
                ),
                "text": candidate_text,
            },
            "source_overlap": 1.0,
        })

    return {
        "generator": "qwen",
        "model_name": GENERATOR_CONFIG[
            "qwen"
        ]["model_name"],
        "status": (
            "generated"
            if generated_claims
            else "no_claims_generated"
        ),
        "claims": generated_claims,
        "raw_outputs": raw_outputs,
        "context": prepare_candidate_context(
            candidates
        ),
    }

def generate_structured_mom(
    candidates,
    generator="bart",
    model=None,
    tokenizer=None,
):
    """
    Unified MoM generation entry point.

    generator:
        bart -> baseline
        qwen -> final structured generator
    """

    generator = (
        str(generator)
        .lower()
        .strip()
    )

    if generator == "bart":

        return generate_mom_bart(
            candidates,
            model=model,
            tokenizer=tokenizer,
        )

    if generator == "qwen":

        return generate_mom_qwen(
            candidates,
            model=model,
            tokenizer=tokenizer,
        )

    raise ValueError(
        f"Unsupported generator: {generator}. "
        f"Use 'bart' or 'qwen'."
    )

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# =============================================================================
# BGE + FAISS EVIDENCE RETRIEVAL
# =============================================================================
# Generalized retrieval layer.
#
# IMPORTANT:
# - Evidence is created from the CURRENT meeting's speaker utterances.
# - No meeting name or transcript file is hard-coded.
# - BGE similarity is used for retrieval only.
# - Evidence verification is performed separately by the NLI stage.
# =============================================================================

BGE_MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_bge_model(device=None):
    """
    Load the BGE embedding model used for evidence retrieval.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return SentenceTransformer(
        BGE_MODEL_NAME,
        device=device
    )


def build_evidence_documents(speaker_utterances):
    """
    Convert the current meeting's speaker utterances into
    evidence documents suitable for BGE + FAISS retrieval.

    Parameters
    ----------
    speaker_utterances : list
        Speaker-level utterances produced by the current meeting pipeline.

    Returns
    -------
    list
        Evidence documents containing speaker, timestamp and text metadata.
    """

    evidence_documents = []

    for i, utterance in enumerate(speaker_utterances, start=1):

        text = str(utterance.get("text", "")).strip()

        if not text:
            continue

        start = float(utterance.get("start", 0.0))
        end = float(utterance.get("end", start))

        evidence_documents.append({
            "evidence_id": i,
            "speaker": utterance.get("speaker"),
            "start": start,
            "end": end,
            "duration": end - start,
            "text": text
        })

    return evidence_documents


def build_faiss_evidence_index(
    evidence_documents,
    bge_model
):
    """
    Build a FAISS cosine-similarity index from the current meeting's
    evidence documents.

    BGE embeddings are normalized, therefore IndexFlatIP provides
    cosine-similarity search.
    """

    if not evidence_documents:
        raise ValueError(
            "Cannot build evidence index: no evidence documents available."
        )

    evidence_texts = [
        document["text"]
        for document in evidence_documents
    ]

    evidence_embeddings = bge_model.encode(
        evidence_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    ).astype(np.float32)

    embedding_dimension = evidence_embeddings.shape[1]

    faiss_index = faiss.IndexFlatIP(
        embedding_dimension
    )

    faiss_index.add(evidence_embeddings)

    return faiss_index, evidence_embeddings


def retrieve_evidence(
    claim_text,
    bge_model,
    faiss_index,
    evidence_documents,
    top_k=10
):
    """
    Retrieve the top-K semantically relevant evidence documents
    for a generated MoM claim.

    Retrieval similarity identifies candidate evidence.
    It does NOT establish that the evidence supports the claim.
    """

    if not claim_text or not str(claim_text).strip():
        return []

    if faiss_index is None or not evidence_documents:
        return []

    query_embedding = bge_model.encode(
        [str(claim_text)],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype(np.float32)

    # Prevent requesting more results than available documents.
    actual_top_k = min(
        int(top_k),
        len(evidence_documents)
    )

    similarities, indices = faiss_index.search(
        query_embedding,
        actual_top_k
    )

    results = []

    for rank, (index, similarity) in enumerate(
        zip(indices[0], similarities[0]),
        start=1
    ):

        if index < 0:
            continue

        evidence = evidence_documents[index].copy()

        evidence["retrieval_rank"] = rank
        evidence["retrieval_similarity"] = float(similarity)

        results.append(evidence)

    return results


def build_evidence_retrieval_resources(
    speaker_utterances,
    device=None
):
    """
    Build all BGE + FAISS resources for the CURRENT meeting.

    This is the main generalized entry point for the retrieval stage.
    """

    evidence_documents = build_evidence_documents(
        speaker_utterances
    )

    bge_model = load_bge_model(
        device=device
    )

    faiss_index, evidence_embeddings = build_faiss_evidence_index(
        evidence_documents,
        bge_model
    )

    return {
        "bge_model": bge_model,
        "faiss_index": faiss_index,
        "evidence_documents": evidence_documents,
        "evidence_embeddings": evidence_embeddings
    }


def retrieve_evidence_for_generated_claims(
    generation_result,
    speaker_utterances,
    top_k=10,
    device=None,
):
    """
    Attach BGE + FAISS evidence to generated MoM claims.

    Generation and retrieval remain separate stages.

    Parameters
    ----------
    generation_result : dict
        Output from generate_structured_mom(),
        including the "claims" list.

    speaker_utterances : list
        Speaker-level meeting utterances.

    top_k : int
        Number of evidence items retrieved per claim.

    device : str or None
        Device used by the BGE model.

    Returns
    -------
    dict
        Copy of generation_result with retrieved
        evidence attached to each generated claim.
    """

    if not isinstance(
        generation_result,
        dict
    ):
        raise TypeError(
            "generation_result must be a dictionary."
        )

    claims = generation_result.get(
        "claims",
        []
    )

    if not isinstance(
        claims,
        list
    ):
        claims = []

    retrieval_resources = (
        build_evidence_retrieval_resources(
            speaker_utterances,
            device=device,
        )
    )

    bge_model = retrieval_resources[
        "bge_model"
    ]

    faiss_index = retrieval_resources[
        "faiss_index"
    ]

    evidence_documents = retrieval_resources[
        "evidence_documents"
    ]

    retrieved_claims = []

    for claim in claims:

        if not isinstance(
            claim,
            dict
        ):
            continue

        claim_copy = claim.copy()

        claim_text = str(
            claim.get(
                "claim",
                ""
            )
        ).strip()

        evidence = retrieve_evidence(
            claim_text=claim_text,
            bge_model=bge_model,
            faiss_index=faiss_index,
            evidence_documents=evidence_documents,
            top_k=top_k,
        )

        claim_copy[
            "retrieved_evidence"
        ] = evidence

        retrieved_claims.append(
            claim_copy
        )

    result = generation_result.copy()

    result[
        "claims"
    ] = retrieved_claims

    result[
        "retrieval"
    ] = {
        "model": BGE_MODEL_NAME,
        "top_k": int(top_k),
        "evidence_count": len(
            evidence_documents
        ),
        "status": "completed",
    }

    return result


# ============================================================
# M10 — MULTI-ATTRIBUTE EVIDENCE CONSISTENCY
# Atomic evidence classification
# ============================================================

def classify_atomic_candidate(
    candidate,
    entailment_threshold=0.80,
    contradiction_threshold=0.80,
    relevance_threshold=0.20,
):
    """
    Classify one retrieved evidence candidate.

    Rules:
      1. Strong entailment -> SUPPORTED
      2. Strong contradiction requires relevance
      3. Irrelevant contradiction -> UNCERTAIN
      4. Otherwise -> UNCERTAIN

    Retrieval similarity is not treated as verification.
    """

    nli_label = str(
        candidate.get("nli_label", "")
    ).lower()

    entailment = float(
        candidate.get("nli_entailment", 0.0)
    )

    contradiction = float(
        candidate.get("nli_contradiction", 0.0)
    )

    lexical_overlap = float(
        candidate.get("lexical_overlap", 0.0)
    )

    # --------------------------------------------------------
    # Strong semantic entailment
    # --------------------------------------------------------

    if (
        nli_label == "entailment"
        and entailment >= entailment_threshold
    ):
        return "SUPPORTED"

    # --------------------------------------------------------
    # Relevant contradiction
    # --------------------------------------------------------

    if (
        nli_label == "contradiction"
        and contradiction >= contradiction_threshold
        and lexical_overlap >= relevance_threshold
    ):
        return "CONTRADICTED"

    # --------------------------------------------------------
    # Everything else
    # --------------------------------------------------------

    return "UNCERTAIN"


# ============================================================
# M10 — EVIDENCE ANALYSIS
# BGE retrieval + DeBERTa NLI + lexical support
# ============================================================

def analyze_evidence_candidate(
    proposition,
    evidence,
    claim_speaker=None,
):
    """
    Analyze one retrieved evidence candidate.

    Combines:
      - retrieval metadata
      - DeBERTa NLI
      - lexical support
      - speaker consistency

    This function does NOT make the final
    SUPPORTED / CONTRADICTED / UNCERTAIN decision.
    """

    evidence_text = str(
        evidence.get("text", "")
    ).strip()

    # --------------------------------------------------------
    # NLI
    # --------------------------------------------------------

    # Reuse a cached DeBERTa NLI model.
    global _NLI_RESOURCES

    if "_NLI_RESOURCES" not in globals():
        _NLI_RESOURCES = load_nli_model()

    nli_tokenizer, nli_model, nli_device = _NLI_RESOURCES

    nli_result = run_nli(
        evidence_text,
        proposition,
        nli_tokenizer,
        nli_model,
        nli_device,
    )

    # --------------------------------------------------------
    # Lexical support
    # --------------------------------------------------------

    lexical_result = lexical_support(
        proposition,
        evidence_text
    )

    # Support both known lexical-result schemas.
    lexical_overlap = float(
        lexical_result.get(
            "lexical_overlap",
            lexical_result.get(
                "overlap_ratio",
                0.0
            )
        )
    )

    # --------------------------------------------------------
    # Speaker consistency
    # --------------------------------------------------------

    evidence_speaker = evidence.get(
        "speaker"
    )

    speaker_match = None

    if claim_speaker is not None:
        speaker_match = (
            evidence_speaker == claim_speaker
            or claim_speaker == "MULTIPLE"
        )

    # --------------------------------------------------------
    # Standardized candidate record
    # --------------------------------------------------------

    return {
        "evidence_id":
            evidence.get("evidence_id"),

        "speaker":
            evidence_speaker,

        "start":
            evidence.get("start"),

        "end":
            evidence.get("end"),

        "text":
            evidence_text,

        "retrieval_similarity":
            float(
                evidence.get(
                    "retrieval_similarity",
                    0.0
                )
            ),

        "nli_label":
            str(
                nli_result.get(
                    "nli_label",
                    nli_result.get(
                        "predicted_label",
                        ""
                    )
                )
            ).lower(),

        "nli_entailment":
            float(
                nli_result.get(
                    "entailment",
                    nli_result.get(
                        "entailment_probability",
                        0.0
                    )
                )
            ),

        "nli_contradiction":
            float(
                nli_result.get(
                    "contradiction",
                    nli_result.get(
                        "contradiction_probability",
                        0.0
                    )
                )
            ),

        "nli_neutral":
            float(
                nli_result.get(
                    "neutral",
                    nli_result.get(
                        "neutral_probability",
                        0.0
                    )
                )
            ),

        "lexical_overlap":
            lexical_overlap,

        "matched_terms":
            lexical_result.get(
                "matched_terms",
                []
            ),

        "missing_terms":
            lexical_result.get(
                "missing_terms",
                []
            ),

        "speaker_match":
            speaker_match,
    }


# ============================================================
# M10 — DEBERTA NLI
# ============================================================

NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"


def load_nli_model(device=None):
    """
    Load the DeBERTa-v3 NLI model used for
    evidence consistency verification.
    """

    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
    )

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        NLI_MODEL_NAME
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        NLI_MODEL_NAME
    )

    model = model.to(device)
    model.eval()

    return tokenizer, model, device


def run_nli(
    evidence_text,
    proposition,
    nli_tokenizer,
    nli_model,
    device,
):
    """
    Run DeBERTa-v3 NLI.

    Ordering follows the validated project implementation:
        premise    = evidence
        hypothesis = proposition

    Returns contradiction, entailment and neutral
    probabilities.
    """

    inputs = nli_tokenizer(
        evidence_text,
        proposition,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = nli_model(**inputs)

    probabilities = (
        torch.softmax(
            outputs.logits,
            dim=-1
        )[0]
        .cpu()
        .tolist()
    )

    label_map = {
        0: "contradiction",
        1: "entailment",
        2: "neutral",
    }

    predicted_id = int(
        torch.argmax(
            outputs.logits,
            dim=-1
        )[0]
        .cpu()
    )

    return {
        "nli_label": label_map[predicted_id],
        "contradiction": float(probabilities[0]),
        "entailment": float(probabilities[1]),
        "neutral": float(probabilities[2]),
    }


# ============================================================
# M10 — LEXICAL SUPPORT
# ============================================================

def lexical_support(
    proposition,
    evidence_text,
):
    """
    Measure content-word overlap between a proposition
    and an evidence passage.

    This is a supporting relevance signal.
    It is NOT a standalone verification decision.
    """

    proposition_text = str(
        proposition or ""
    ).lower()

    evidence_text = str(
        evidence_text or ""
    ).lower()

    proposition_terms = set(
        re.findall(
            r"\\b[a-zA-Z0-9]+\\b",
            proposition_text,
        )
    )

    evidence_terms = set(
        re.findall(
            r"\\b[a-zA-Z0-9]+\\b",
            evidence_text,
        )
    )

    # Remove very common function words.
    stopwords = {
        "the", "a", "an", "is", "are",
        "was", "were", "be", "been",
        "being", "to", "of", "and",
        "or", "in", "on", "at",
        "for", "with", "by", "from",
        "this", "that", "these", "those",
        "it", "its", "as",
    }

    proposition_terms = {
        term
        for term in proposition_terms
        if term not in stopwords
    }

    evidence_terms = {
        term
        for term in evidence_terms
        if term not in stopwords
    }

    if not proposition_terms:
        return {
            "lexical_overlap": 0.0,
            "matched_terms": [],
            "missing_terms": [],
        }

    matched_terms = sorted(
        proposition_terms.intersection(
            evidence_terms
        )
    )

    missing_terms = sorted(
        proposition_terms.difference(
            evidence_terms
        )
    )

    overlap = (
        len(matched_terms)
        / len(proposition_terms)
    )

    return {
        "lexical_overlap": float(overlap),
        "matched_terms": matched_terms,
        "missing_terms": missing_terms,
    }


# ============================================================
# M10 — MULTI-EVIDENCE ATOMIC VERIFICATION
# ============================================================

def verify_atomic_proposition(
    proposition,
    bge_model,
    faiss_index,
    evidence_documents,
    claim_speaker=None,
    top_k=10,
):
    """
    Verify one atomic proposition against multiple
    retrieved evidence candidates.

    Decision priority:
        SUPPORTED > CONTRADICTED > UNCERTAIN

    The validated M10 thresholds are preserved:
        entailment       >= 0.80
        contradiction    >= 0.80
        lexical relevance >= 0.20
    """

    proposition = str(
        proposition or ""
    ).strip()

    if not proposition:
        return {
            "proposition": proposition,
            "status": "UNCERTAIN",
            "selected_evidence": None,
            "candidates": [],
        }

    # --------------------------------------------------------
    # Retrieve Top-K evidence
    # --------------------------------------------------------

    retrieved = retrieve_evidence(
        proposition,
        bge_model=bge_model,
        faiss_index=faiss_index,
        evidence_documents=evidence_documents,
        top_k=top_k,
    )

    analyzed_candidates = []

    # --------------------------------------------------------
    # Analyze every retrieved candidate
    # --------------------------------------------------------

    for evidence in retrieved:

        analyzed = analyze_evidence_candidate(
            proposition=proposition,
            evidence=evidence,
            claim_speaker=claim_speaker,
        )

        status = classify_atomic_candidate(
            analyzed
        )

        analyzed["status"] = status

        analyzed_candidates.append(
            analyzed
        )

    # --------------------------------------------------------
    # Select strongest candidate
    #
    # Priority:
    #   1. SUPPORTED
    #   2. CONTRADICTED
    #   3. UNCERTAIN
    # --------------------------------------------------------

    supported = [
        candidate
        for candidate in analyzed_candidates
        if candidate["status"] == "SUPPORTED"
    ]

    contradicted = [
        candidate
        for candidate in analyzed_candidates
        if candidate["status"] == "CONTRADICTED"
    ]

    uncertain = [
        candidate
        for candidate in analyzed_candidates
        if candidate["status"] == "UNCERTAIN"
    ]

    selected = None
    overall_status = "UNCERTAIN"

    if supported:

        selected = max(
            supported,
            key=lambda candidate: (
                candidate["nli_entailment"],
                candidate["retrieval_similarity"],
            ),
        )

        overall_status = "SUPPORTED"

    elif contradicted:

        selected = max(
            contradicted,
            key=lambda candidate: (
                candidate["nli_contradiction"],
                candidate["lexical_overlap"],
                candidate["retrieval_similarity"],
            ),
        )

        overall_status = "CONTRADICTED"

    elif uncertain:

        selected = max(
            uncertain,
            key=lambda candidate: (
                candidate["retrieval_similarity"],
                candidate["lexical_overlap"],
            ),
        )

    # --------------------------------------------------------
    # Return complete verification record
    # --------------------------------------------------------

    return {
        "proposition": proposition,
        "status": overall_status,
        "selected_evidence": selected,
        "candidates": analyzed_candidates,
        "retrieved_count": len(retrieved),
        "supported_count": len(supported),
        "contradicted_count": len(contradicted),
        "uncertain_count": len(uncertain),
    }


# ============================================================
# MULTI-ATTRIBUTE CONSISTENCY — SPEAKER
# ============================================================

def check_speaker_consistency_pipeline(
    claim_speaker,
    evidence_speaker,
):
    """
    Check whether the speaker attributed to a claim
    matches the speaker of the selected evidence.

    MULTIPLE is treated as a valid multi-speaker claim.
    """

    if claim_speaker is None:
        return {
            "status": "NOT_EVALUATED",
            "claim_speaker": None,
            "evidence_speaker": evidence_speaker,
        }

    if evidence_speaker is None:
        return {
            "status": "NOT_EVALUATED",
            "claim_speaker": claim_speaker,
            "evidence_speaker": None,
        }

    if claim_speaker == "MULTIPLE":
        status = "PASS"

    elif claim_speaker == evidence_speaker:
        status = "PASS"

    else:
        status = "FAIL"

    return {
        "status": status,
        "claim_speaker": claim_speaker,
        "evidence_speaker": evidence_speaker,
    }


# ============================================================
# MULTI-ATTRIBUTE CONSISTENCY — EVENT TYPE
# ============================================================

def check_event_type_consistency_pipeline(
    claim_event_type,
    evidence_event_type,
):
    """
    Check whether the event type attributed to a claim
    is consistent with the event type of the evidence.

    Supported current schemas:
        - event_type
        - candidate_type

    Event types are compared as normalized labels.
    """

    if claim_event_type is None:
        return {
            "status": "NOT_EVALUATED",
            "claim_event_type": None,
            "evidence_event_type": evidence_event_type,
        }

    if evidence_event_type is None:
        return {
            "status": "NOT_EVALUATED",
            "claim_event_type": claim_event_type,
            "evidence_event_type": None,
        }

    claim_type = str(
        claim_event_type
    ).strip().upper()

    evidence_type = str(
        evidence_event_type
    ).strip().upper()

    if claim_type == evidence_type:
        status = "PASS"
    else:
        status = "FAIL"

    return {
        "status": status,
        "claim_event_type": claim_type,
        "evidence_event_type": evidence_type,
    }


def get_claim_event_type(claim):
    """
    Resolve the event type from the current claim schemas.

    Priority:
        event_type → candidate_type → type

    The legacy `type` field is retained only as a
    compatibility fallback.
    """

    if not isinstance(claim, dict):
        return None

    if claim.get("event_type") is not None:
        return claim.get("event_type")

    if claim.get("candidate_type") is not None:
        return claim.get("candidate_type")

    if claim.get("type") is not None:
        return claim.get("type")

    return None


def get_evidence_event_type(evidence):
    """
    Resolve event type from an evidence record.

    Current evidence records may carry `event_type`
    or `candidate_type`.
    """

    if not isinstance(evidence, dict):
        return None

    if evidence.get("event_type") is not None:
        return evidence.get("event_type")

    if evidence.get("candidate_type") is not None:
        return evidence.get("candidate_type")

    return None


# ============================================================
# MULTI-ATTRIBUTE CONSISTENCY — TEMPORAL
# ============================================================

def check_timestamp_consistency_pipeline(
    claim_start,
    claim_end,
    evidence_start,
    evidence_end,
    tolerance_seconds=5.0,
):
    """
    Check whether a claim timestamp is consistent with
    the timestamp of its selected evidence.

    The evidence is considered temporally consistent when
    the claim interval and evidence interval overlap, or
    when their boundaries are within the configured tolerance.

    If usable timestamps are unavailable, return
    NOT_EVALUATED rather than forcing a decision.
    """

    # --------------------------------------------------------
    # Validate timestamps
    # --------------------------------------------------------

    if (
        claim_start is None
        or claim_end is None
        or evidence_start is None
        or evidence_end is None
    ):
        return {
            "status": "NOT_EVALUATED",
            "claim_start": claim_start,
            "claim_end": claim_end,
            "evidence_start": evidence_start,
            "evidence_end": evidence_end,
            "temporal_distance": None,
        }

    try:
        claim_start = float(claim_start)
        claim_end = float(claim_end)
        evidence_start = float(evidence_start)
        evidence_end = float(evidence_end)
        tolerance_seconds = float(tolerance_seconds)
    except (TypeError, ValueError):
        return {
            "status": "NOT_EVALUATED",
            "claim_start": claim_start,
            "claim_end": claim_end,
            "evidence_start": evidence_start,
            "evidence_end": evidence_end,
            "temporal_distance": None,
        }

    # --------------------------------------------------------
    # Normalize reversed intervals
    # --------------------------------------------------------

    if claim_end < claim_start:
        claim_start, claim_end = claim_end, claim_start

    if evidence_end < evidence_start:
        evidence_start, evidence_end = (
            evidence_end,
            evidence_start,
        )

    # --------------------------------------------------------
    # Calculate interval relationship
    # --------------------------------------------------------

    overlap = max(
        0.0,
        min(claim_end, evidence_end)
        - max(claim_start, evidence_start),
    )

    if overlap > 0.0:
        return {
            "status": "PASS",
            "claim_start": claim_start,
            "claim_end": claim_end,
            "evidence_start": evidence_start,
            "evidence_end": evidence_end,
            "temporal_distance": 0.0,
            "overlap_seconds": overlap,
        }

    # No overlap — calculate the gap between intervals.
    if claim_end < evidence_start:
        temporal_distance = evidence_start - claim_end
    else:
        temporal_distance = claim_start - evidence_end

    if temporal_distance <= tolerance_seconds:
        status = "PASS"
    else:
        status = "FAIL"

    return {
        "status": status,
        "claim_start": claim_start,
        "claim_end": claim_end,
        "evidence_start": evidence_start,
        "evidence_end": evidence_end,
        "temporal_distance": temporal_distance,
        "overlap_seconds": 0.0,
    }


# ============================================================
# MULTI-ATTRIBUTE EVIDENCE CONSISTENCY
# ============================================================

def verify_multi_attribute_claim(
    claim,
    bge_model,
    faiss_index,
    evidence_documents,
    source_candidates=None,
    top_k=10,
    temporal_tolerance_seconds=5.0,
):
    """
    Perform multi-attribute evidence consistency verification.

    Attributes:
        1. Content
        2. Speaker
        3. Event type
        4. Temporal consistency

    Final decision:
        VERIFIED  -> all applicable attributes pass
        FLAGGED   -> one or more applicable attributes fail

    NOT_EVALUATED attributes are not treated as failures.
    """

    if not isinstance(claim, dict):
        raise TypeError(
            "claim must be a dictionary"
        )

    claim_text = str(
        claim.get(
            "claim",
            claim.get("claim_text", "")
        )
    ).strip()

    claim_speaker = claim.get(
        "speaker"
    )

    claim_event_type = get_claim_event_type(
        claim
    )

    claim_start = claim.get(
        "start"
    )

    claim_end = claim.get(
        "end"
    )

    # --------------------------------------------------------
    # 1. CONTENT / EVIDENCE VERIFICATION
    # --------------------------------------------------------

    content_result = verify_atomic_proposition(
        proposition=claim_text,
        bge_model=bge_model,
        faiss_index=faiss_index,
        evidence_documents=evidence_documents,
        claim_speaker=claim_speaker,
        top_k=top_k,
    )

    selected_evidence = content_result.get(
        "selected_evidence"
    )

    content_status = content_result.get(
        "status",
        "UNCERTAIN"
    )

    # --------------------------------------------------------
    # 2. SPEAKER CONSISTENCY
    # --------------------------------------------------------

    if selected_evidence is not None:

        speaker_result = (
            check_speaker_consistency_pipeline(
                claim_speaker=claim_speaker,
                evidence_speaker=selected_evidence.get(
                    "speaker"
                ),
            )
        )

    else:

        speaker_result = {
            "status": "NOT_EVALUATED",
            "claim_speaker": claim_speaker,
            "evidence_speaker": None,
        }

    # --------------------------------------------------------
    # 3. EVENT-TYPE CONSISTENCY
    # --------------------------------------------------------

    if selected_evidence is not None:

        # Resolve event type from the V3 candidate that
        # temporally overlaps the selected evidence.
        resolved_event = (
            resolve_evidence_event_type_pipeline(
                evidence=selected_evidence,
                source_candidates=source_candidates,
            )
        )

        if resolved_event is not None:
            evidence_event_type = (
                resolved_event.get("event_type")
            )
        else:
            evidence_event_type = (
                get_evidence_event_type(
                    selected_evidence
                )
            )

        event_result = (
            check_event_type_consistency_pipeline(
                claim_event_type=claim_event_type,
                evidence_event_type=evidence_event_type,
            )
        )

    else:

        event_result = {
            "status": "NOT_EVALUATED",
            "claim_event_type": claim_event_type,
            "evidence_event_type": None,
        }

    # --------------------------------------------------------
    # 4. TEMPORAL CONSISTENCY
    # --------------------------------------------------------

    if selected_evidence is not None:

        temporal_result = (
            check_timestamp_consistency_pipeline(
                claim_start=claim_start,
                claim_end=claim_end,
                evidence_start=selected_evidence.get(
                    "start"
                ),
                evidence_end=selected_evidence.get(
                    "end"
                ),
                tolerance_seconds=(
                    temporal_tolerance_seconds
                ),
            )
        )

    else:

        temporal_result = {
            "status": "NOT_EVALUATED",
            "claim_start": claim_start,
            "claim_end": claim_end,
            "evidence_start": None,
            "evidence_end": None,
            "temporal_distance": None,
        }

    # --------------------------------------------------------
    # ATTRIBUTE STATUS COLLECTION
    # --------------------------------------------------------

    attribute_results = {
        "content": content_status,
        "speaker": speaker_result["status"],
        "event_type": event_result["status"],
        "temporal": temporal_result["status"],
    }

    # --------------------------------------------------------
    # FINAL MULTI-ATTRIBUTE DECISION
    #
    # Content must be SUPPORTED.
    # Every applicable additional attribute must PASS.
    # NOT_EVALUATED attributes do not cause failure.
    # --------------------------------------------------------

    content_pass = (
        content_status == "SUPPORTED"
    )

    additional_failures = []

    for attribute_name in [
        "speaker",
        "event_type",
        "temporal",
    ]:

        attribute_status = attribute_results[
            attribute_name
        ]

        if attribute_status == "FAIL":
            additional_failures.append(
                attribute_name
            )

    if (
        content_pass
        and not additional_failures
    ):
        overall_status = "VERIFIED"
    else:
        overall_status = "FLAGGED"

    # --------------------------------------------------------
    # RETURN COMPLETE VERIFICATION RECORD
    # --------------------------------------------------------

    return {
        "claim": claim,
        "claim_text": claim_text,

        "overall_status": overall_status,

        "content": {
            "status": content_status,
            "verification": content_result,
        },

        "speaker": speaker_result,

        "event_type": event_result,

        "temporal": temporal_result,

        "attribute_results": attribute_results,

        "failed_attributes": additional_failures,

        "selected_evidence": selected_evidence,
    }


# ============================================================
# EVENT-TYPE RESOLUTION FOR EVIDENCE
# ============================================================

def resolve_evidence_event_type_pipeline(
    evidence,
    source_candidates=None,
):
    """
    Resolve the event type associated with a retrieved
    evidence utterance using temporal overlap with the
    V3 MoM candidate set.

    The retrieved evidence itself is still the source of
    content verification. V3 candidates provide the
    event-type metadata.
    """

    if not source_candidates:
        return None

    evidence_start = evidence.get("start")
    evidence_end = evidence.get("end")

    if (
        evidence_start is None
        or evidence_end is None
    ):
        return None

    try:
        evidence_start = float(evidence_start)
        evidence_end = float(evidence_end)
    except (TypeError, ValueError):
        return None

    best_candidate = None
    best_overlap = 0.0

    for candidate in source_candidates:

        candidate_start = candidate.get("start")
        candidate_end = candidate.get("end")

        if (
            candidate_start is None
            or candidate_end is None
        ):
            continue

        try:
            candidate_start = float(candidate_start)
            candidate_end = float(candidate_end)
        except (TypeError, ValueError):
            continue

        overlap = max(
            0.0,
            min(
                evidence_end,
                candidate_end
            )
            - max(
                evidence_start,
                candidate_start
            )
        )

        if overlap > best_overlap:
            best_overlap = overlap
            best_candidate = candidate

    if best_candidate is None:
        return None

    return {
        "event_type": (
            best_candidate.get("candidate_type")
            or best_candidate.get("event_type")
            or best_candidate.get("type")
        ),
        "candidate_overlap": best_overlap,
        "candidate": best_candidate,
    }


# ============================================================
# Dashboard Result Formatting
# ============================================================

def format_verified_claim_result(result):
    """
    Convert the internal verification result into a stable
    dashboard-ready schema.
    """

    if result is None:
        return {
            "claim": None,
            "speaker": None,
            "start": None,
            "end": None,
            "event_type": None,
            "status": "ERROR",
            "verification": {},
            "evidence": {},
        }

    claim_data = result.get("claim", {})

    if isinstance(claim_data, dict):
        claim_text = claim_data.get("claim", "")
        speaker = claim_data.get("speaker")
        start = claim_data.get("start")
        end = claim_data.get("end")
        event_type = claim_data.get("event_type")
    else:
        claim_text = claim_data
        speaker = result.get("speaker")
        start = result.get("start")
        end = result.get("end")
        event_type = result.get("event_type")

    selected_evidence = result.get("selected_evidence") or {}

    status = (
        result.get("overall_status")
        or result.get("status")
        or "UNKNOWN"
    )

    content_value = result.get("content")
    content_status = (
        content_value.get("status", "NOT_EVALUATED")
        if isinstance(content_value, dict)
        else content_value or "NOT_EVALUATED"
    )

    speaker_value = result.get("speaker")
    speaker_status = (
        speaker_value.get("status", "NOT_EVALUATED")
        if isinstance(speaker_value, dict)
        else "NOT_EVALUATED"
    )

    event_value = result.get("event_type")
    event_status = (
        event_value.get("status", "NOT_EVALUATED")
        if isinstance(event_value, dict)
        else "NOT_EVALUATED"
    )

    temporal_value = result.get("temporal")
    temporal_status = (
        temporal_value.get("status", "NOT_EVALUATED")
        if isinstance(temporal_value, dict)
        else temporal_value or "NOT_EVALUATED"
    )

    return {
        "claim": claim_text,
        "speaker": speaker,
        "start": start,
        "end": end,
        "event_type": event_type,
        "status": status,
        "verification": {
            "content": content_status,
            "speaker": speaker_status,
            "event_type": event_status,
            "temporal": temporal_status,
        },
        "evidence": {
            "evidence_id": selected_evidence.get("evidence_id"),
            "speaker": selected_evidence.get("speaker"),
            "start": selected_evidence.get("start"),
            "end": selected_evidence.get("end"),
            "text": selected_evidence.get("text"),
            "retrieval_similarity": selected_evidence.get(
                "retrieval_similarity"
            ),
            "nli_label": selected_evidence.get("nli_label"),
            "nli_entailment": selected_evidence.get(
                "nli_entailment"
            ),
            "nli_contradiction": selected_evidence.get(
                "nli_contradiction"
            ),
            "lexical_overlap": selected_evidence.get(
                "lexical_overlap"
            ),
        },
    }

