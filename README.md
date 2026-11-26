# Verifiable Meeting Minutes

## Multi-Attribute Evidence Consistency for Verifiable Meeting Minutes

An open-source AI system for converting meeting audio into structured Minutes of Meeting (MoM) and verifying generated claims against evidence from the original meeting audio.

**Core idea:** Meeting Audio -> MoM Claim -> Evidence -> Timestamp -> Original Audio

---

## 1. Project Overview

Traditional automatic meeting-minutes systems can generate fluent summaries, but generated claims may contain unsupported information, incorrect speaker attribution, incorrect event types, temporal inconsistencies, or hallucinated claims.

This project addresses these problems using:

- Silero VAD
- WhisperX ASR
- Pyannote speaker diarization
- Speaker-word alignment
- MoM candidate detection
- Structured claim generation
- Evidence retrieval
- Multi-attribute verification

The final system classifies generated claims as **VERIFIED** or **FLAGGED**.

---

## 2. Main Features

### Meeting Processing

- Silero VAD
- WhisperX ASR
- Word-level timestamps
- Pyannote speaker diarization
- Speaker-word alignment
- MoM candidate filtering
- Structured MoM generation

### Evidence Verification

- BGE sentence embeddings
- FAISS similarity search
- Evidence retrieval
- Content verification
- Speaker verification
- Event-type verification
- Temporal consistency verification
- NLI verification using DeBERTa

### Evidence Traceability

For each generated claim, the dashboard can show:

**Claim -> Evidence -> Speaker -> Exact Timestamp -> Original Audio**

The exact evidence segment from the original meeting can be played directly from the dashboard.

### Dashboard

- Meeting upload
- Meeting processing
- Generated MoM
- VERIFIED / FLAGGED claims
- Evidence display
- Evidence timestamp navigation
- Exact evidence audio playback
- Previously processed meetings
- PDF export

---

## 3. Architecture

### Part A - Meeting Understanding

```text
Meeting Audio
     |
     v
Silero VAD
     |
     v
WhisperX ASR
     |
     v
Timestamped Transcript
     |
     v
Pyannote Speaker Diarization
     |
     v
Speaker-Word Alignment
     |
     v
MoM Candidate Filtering
     |
     v
Structured MoM Claims
```

### Part B - Evidence Verification

```text
Generated Claim
      |
      v
Evidence Retrieval
      |
      v
BGE Embeddings
      |
      v
FAISS Similarity Search
      |
      v
Candidate Evidence
      |
      v
Multi-Attribute Verification
      |
      v
VERIFIED / FLAGGED
```

---

## 4. Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| Dashboard | Streamlit |
| VAD | Silero VAD |
| ASR | WhisperX |
| Speaker Diarization | Pyannote |
| MoM Generation | BART / Qwen |
| Embeddings | BGE-small-en-v1.5 |
| Vector Search | FAISS |
| NLI | DeBERTa-v3 |
| PDF Processing | PyMuPDF |
| PDF Generation | ReportLab |
| Audio Processing | FFmpeg |
| Dataset | AMI Meeting Corpus |

---

## 5. Models

### ASR

`WhisperX`

Used for speech recognition and timestamps.

### Speaker Diarization

`pyannote/speaker-diarization-3.1`

### Baseline MoM Generator

`facebook/bart-large-cnn`

### Final MoM Generator

`Qwen/Qwen2.5-3B-Instruct`

### Evidence Embeddings

`BAAI/bge-small-en-v1.5`

### NLI

`cross-encoder/nli-deberta-v3-small`

---

## 6. Repository Structure

```text
MoM_Project/
|
+-- 07_dashboard/
|   +-- app.py
|   +-- meeting_pipeline.py
|   +-- demo_environment_check.py
|   +-- requirements_demo.txt
|
+-- data/
|   +-- raw/
|   +-- transcripts/
|   +-- evidence/
|   +-- embeddings/
|   +-- faiss/
|
+-- outputs/
|   +-- live/
|   +-- verified/
|   +-- pdf/
|
+-- README.md
```

---

## 7. Requirements

Recommended:

- Python 3.10
- NVIDIA GPU for complete meeting processing
- FFmpeg

The dashboard can run on CPU when loading already processed meeting results.

## 08. Process a New Meeting

Select **Process New Meeting** in the Streamlit dashboard and upload a meeting audio file.

The pipeline performs:

```text
Audio
  |
  v
VAD
  |
  v
WhisperX
  |
  v
Speaker Diarization
  |
  v
Speaker Alignment
  |
  v
MoM Candidate Detection
  |
  v
MoM Generation
  |
  v
Evidence Retrieval
  |
  v
Claim Verification
```

---

## 09. VERIFIED vs FLAGGED

### VERIFIED

The available evidence sufficiently supports the generated claim according to the verification criteria.

### FLAGGED

One or more important verification attributes could not be sufficiently supported.

A FLAGGED claim does not automatically mean that the claim is false. It means the system could not establish sufficient evidence consistency.

---

## 10. Evidence Playback

A key feature of the project is evidence-level traceability.

**Claim -> Evidence -> Speaker -> Exact Timestamp -> Original Audio Segment**

The evidence timestamp can be used to play the corresponding section of the original meeting recording.

This provides an auditable connection between generated MoM claims and their source audio.

---

## 11. PDF Export

The dashboard supports PDF export of generated meeting minutes and verification results.

PDF generation uses **ReportLab**.

---

## 12. Dataset

Experiments use the **AMI Meeting Corpus**.

Meetings used during the project include:

- ES2004a
- ES2004b
- ES2004c
- ES2004d
- ES2005a
- ES2005b
- ES2005c
- ES2006a
- ES2006b
- ES2008a

Users should obtain the dataset from its official source and follow its licensing and usage conditions.

---

## 13. Experimental Results

### ES2004a

| Metric | Result |
|---|---:|
| Structured claims | 12 |
| Verified claims | 7 |
| Flagged claims | 5 |
| Verification rate | 58.33% |
| Recall@10 | 79.44% |
| Precision@10 | 26.67% |

Controlled validation achieved **5/5 correct (100%)**.

### Live Demonstration

| Metric | Result |
|---|---:|
| Transcript segments | 21 |
| Aligned segments | 132 |
| MoM candidates | 9 |
| Generated claims | 9 |
| Evidence documents | 21 |
| Verified claims | 6 |
| Flagged claims | 3 |
| Verification rate | 66.67% |

---

## 14. CPU vs GPU

The dashboard can run on CPU when viewing previously processed meetings.

GPU is strongly recommended for complete processing because the pipeline includes WhisperX, Pyannote, Transformer generation, embeddings, and NLI verification.

---

## 15. Speaker Names

Diarization initially produces labels such as:

`SPEAKER_00`, `SPEAKER_01`, `SPEAKER_02`

Where participant information is available, these labels can be mapped to actual participant names.

---

## 16. Research Contribution

The main contribution is not simply automatic meeting summarization.

The project focuses on:

**Verifiable Meeting Minutes through Multi-Attribute Evidence Consistency**

The verification framework considers:

- Content
- Speaker
- Event Type
- Temporal Consistency

The system therefore moves from automatic meeting summarization toward evidence-grounded and auditable meeting minutes.

---

## 17. Current Status

Implemented:

- Silero VAD
- WhisperX ASR
- Pyannote diarization
- Speaker-word alignment
- MoM candidate filtering
- Structured claim generation
- BGE + FAISS evidence retrieval
- Multi-attribute evidence verification
- VERIFIED / FLAGGED classification
- Streamlit dashboard
- Evidence timestamps
- Exact evidence audio playback
- PDF export
- Previously processed meeting loading

---

## 18. Important Notes

Meeting audio, model files, embeddings, FAISS indexes, and generated outputs can be large. Not every generated artifact should be committed to GitHub.

Some Hugging Face models may require authentication or acceptance of their terms.

Never commit API keys, Hugging Face tokens, passwords, or other credentials.

---

## 19. Acknowledgements

This project uses open-source technologies including:

- AMI Meeting Corpus
- Silero VAD
- WhisperX
- Pyannote
- Hugging Face Transformers
- BGE
- FAISS
- DeBERTa
- Streamlit
- ReportLab

Please refer to the respective projects and model cards for their licenses and citation requirements.

---

## Author
Sunaina Kouser S
M.Tech Data Science and Artificial Intelligence
PES University

---

## License

Add the intended project license before public redistribution.
