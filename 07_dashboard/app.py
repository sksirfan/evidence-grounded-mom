
import json
from pathlib import Path
import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

# Google Colab / Linux
if Path("/content/drive/MyDrive").exists():
    PROJECT_ROOT = Path(
        "/content/drive/MyDrive/MTechIndProj/MoM_Project"
    )

# Windows Google Drive for Desktop
elif Path(r"G:\My Drive\MTechIndProj\MoM_Project").exists():
    PROJECT_ROOT = Path(
        r"G:\My Drive\MTechIndProj\MoM_Project"
    )

else:
    PROJECT_ROOT = Path.cwd().parent


# ============================================================
# DATA PATHS
# ============================================================

CLAIMS_PATH = (
    PROJECT_ROOT
    / "data"
    / "transcripts"
    / "ES2004a_structured_mom_claims.json"
)

VERIFICATION_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "verified"
    / "ES2004a_M9_full_multi_attribute_verification.json"
)

PDF_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "pdf"
    / "ES2004a_Verifiable_Meeting_Minutes.pdf"
)


# ============================================================
# LOAD DATA
# ============================================================

with open(CLAIMS_PATH, "r", encoding="utf-8") as f:
    claims_data = json.load(f)

with open(VERIFICATION_PATH, "r", encoding="utf-8") as f:
    verification_data = json.load(f)


claims = claims_data.get("claims", [])
verification_results = verification_data.get("results", [])


# ============================================================
# BUILD VERIFICATION LOOKUP
# ============================================================

verification_by_claim = {}

for result in verification_results:

    claim_text = result.get("claim", "")

    if claim_text:
        verification_by_claim[claim_text] = result


# ============================================================
# VERIFIED EVIDENCE LOOKUP
# ============================================================

verified_evidence_by_id = {}

for result in verification_results:

    checks = result.get(
        "checks",
        {}
    )

    content_check = checks.get(
        "content_consistency",
        {}
    )

    details = content_check.get(
        "details",
        {}
    )

    supporting_candidates = details.get(
        "supporting_candidates",
        []
    )

    for candidate in supporting_candidates:

        evidence_id = candidate.get(
            "evidence_id"
        )

        if evidence_id is not None:

            verified_evidence_by_id[
                evidence_id
            ] = candidate



# ============================================================
# BUILD DASHBOARD ROWS
# ============================================================

dashboard_rows = []

for claim in claims:

    claim_text = claim.get(
        "claim_text",
        ""
    )

    verification_result = verification_by_claim.get(
        claim_text,
        {}
    )

    dashboard_rows.append({
        "claim_id": claim.get("claim_id"),
        "topic": claim.get("topic"),
        "claim_text": claim_text,
        "speaker": claim.get("speaker"),
        "start": claim.get("start"),
        "end": claim.get("end"),
        "event_type": claim.get("event_type"),

        "status": verification_result.get(
            "status",
            "UNKNOWN"
        ),

        "reason": verification_result.get(
            "reason",
            ""
        ),

        "checks": verification_result.get(
            "checks",
            {}
        ),

        "evidence_ids": verification_result.get(
            "evidence_ids",
            []
        ),

        "evidence_start": verification_result.get(
            "evidence_start"
        ),

        "evidence_end": verification_result.get(
            "evidence_end"
        ),
    })


# ============================================================
# PAGE CONFIG
# ============================================================


RAW_UPLOAD_DIR = PROJECT_ROOT / "data" / "raw"
RAW_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
}

def save_uploaded_meeting(uploaded_file):
    """Save an uploaded meeting recording to data/raw."""
    if uploaded_file is None:
        raise ValueError("No meeting file was supplied.")

    filename = Path(uploaded_file.name).name
    output_path = RAW_UPLOAD_DIR / filename

    with open(output_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return output_path


st.set_page_config(
    page_title="Verifiable Meeting Minutes",
    page_icon="📝",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("📝 Verifiable Meeting Minutes")

st.caption(
    "AI-generated meeting minutes with evidence retrieval "
    "and multi-level verification"
)



# ============================================================
# MEETING METADATA
# ============================================================

meeting_id = claims_data.get(
    "meeting_id",
    "ES2004a"
)

total_claims = len(
    dashboard_rows
)

verified_claims = sum(
    1
    for row in dashboard_rows
    if row["status"] == "VERIFIED"
)

flagged_claims = total_claims - verified_claims

verification_rate = (
    verified_claims / total_claims * 100
    if total_claims
    else 0
)

meeting_metadata = {
    "meeting_id": meeting_id,
    "status": "PROCESSED",
    "total_claims": total_claims,
    "verified_claims": verified_claims,
    "flagged_claims": flagged_claims,
    "verification_rate": verification_rate,
    "pdf_available": PDF_PATH.exists(),
}



# ============================================================
# IMPLEMENTED MEETINGS
# ============================================================

st.subheader("📂 Implemented Meetings")

meeting_col1, meeting_col2 = st.columns([2, 3])

with meeting_col1:

    selected_meeting = st.selectbox(
        "Select Meeting",
        [meeting_metadata["meeting_id"]],
        index=0
    )

with meeting_col2:

    st.markdown(
        f"### {selected_meeting}"
    )

    st.caption(
        "✓ Processed and verified"
    )


# ------------------------------------------------------------
# Meeting summary
# ------------------------------------------------------------

mc1, mc2, mc3, mc4 = st.columns(4)

with mc1:
    st.metric(
        "Claims",
        meeting_metadata["total_claims"]
    )

with mc2:
    st.metric(
        "Verified",
        meeting_metadata["verified_claims"]
    )

with mc3:
    st.metric(
        "Flagged",
        meeting_metadata["flagged_claims"]
    )

with mc4:
    st.metric(
        "Verification Rate",
        f"{meeting_metadata['verification_rate']:.2f}%"
    )



# ============================================================
# LIVE MEETING PROCESSING ORCHESTRATION
# ============================================================

import sys
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

DASHBOARD_DIR = PROJECT_ROOT / "07_dashboard"

if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(DASHBOARD_DIR)
    )

import meeting_pipeline as mp


@st.cache_resource(show_spinner=False)
def load_qwen_generator():
    """
    Load the final Qwen generator once per Streamlit runtime.
    """

    model_name = (
        "Qwen/Qwen2.5-3B-Instruct"
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            model_name
        )
    )

    if torch.cuda.is_available():

        model = (
            AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        )

    else:

        model = (
            AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32
            )
        )

    model.eval()

    return model, tokenizer


def process_uploaded_meeting(
    audio_path,
    generator="qwen",
    generator_model=None,
    generator_tokenizer=None,
    hf_token=None,
    top_k=10,
    temporal_tolerance_seconds=5.0,
):
    """
    Run the validated generic meeting-processing pipeline.

    Pipeline:
        VAD
        -> WhisperX
        -> Pyannote
        -> Speaker alignment
        -> MoM candidate detection
        -> Qwen claim generation
        -> BGE/FAISS evidence retrieval
        -> NLI/consistency verification
    """

    audio_path = Path(
        audio_path
    )

    if not audio_path.exists():

        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    # --------------------------------------------------------
    # 1. VAD
    # --------------------------------------------------------

    vad_result = (
        mp.run_silero_vad(
            audio_path
        )
    )

    # --------------------------------------------------------
    # 2. WhisperX ASR
    # --------------------------------------------------------

    whisper_result = (
        mp.run_whisperx_pipeline(
            audio_path
        )
    )

    # --------------------------------------------------------
    # 3. Pyannote diarization
    # --------------------------------------------------------

    diarization_result = (
        mp.run_pyannote_diarization(
            audio_path,
            hf_token=hf_token
        )
    )

    # --------------------------------------------------------
    # 4. Speaker alignment
    # --------------------------------------------------------

    alignment_result = (
        mp.run_speaker_word_alignment(
            whisper_result,
            diarization_result
        )
    )

    utterances = (
        alignment_result.get(
            "utterances",
            []
        )
    )

    # --------------------------------------------------------
    # 5. MoM candidate detection
    # --------------------------------------------------------

    candidate_result = (
        mp.filter_mom_candidates_v3(
            utterances,
            return_all=True
        )
    )

    candidates = (
        candidate_result.get(
            "candidates",
            []
        )
    )

    # --------------------------------------------------------
    # 6. Qwen structured claim generation
    # --------------------------------------------------------

    generation_result = (
        mp.generate_structured_mom(
            candidates,
            generator=generator,
            model=generator_model,
            tokenizer=generator_tokenizer
        )
    )

    generated_claims = (
        generation_result.get(
            "claims",
            []
        )
    )

    # --------------------------------------------------------
    # 7. Evidence retrieval resources
    # --------------------------------------------------------

    retrieval_resources = (
        mp.build_evidence_retrieval_resources(
            utterances
        )
    )

    bge_model = (
        retrieval_resources[
            "bge_model"
        ]
    )

    faiss_index = (
        retrieval_resources[
            "faiss_index"
        ]
    )

    evidence_documents = (
        retrieval_resources[
            "evidence_documents"
        ]
    )

    # --------------------------------------------------------
    # 8. Evidence verification
    # --------------------------------------------------------

    verification_results = []

    for claim in generated_claims:

        verification_result = (
            mp.verify_multi_attribute_claim(
                claim=claim,
                bge_model=bge_model,
                faiss_index=faiss_index,
                evidence_documents=(
                    evidence_documents
                ),
                source_candidates=candidates,
                top_k=top_k,
                temporal_tolerance_seconds=(
                    temporal_tolerance_seconds
                )
            )
        )

        verification_results.append(
            mp.format_verified_claim_result(
                verification_result
            )
        )

    # --------------------------------------------------------
    # 9. Verification summary
    # --------------------------------------------------------

    verified_count = sum(
        1
        for result in verification_results
        if result.get(
            "status"
        ) == "VERIFIED"
    )

    flagged_count = sum(
        1
        for result in verification_results
        if result.get(
            "status"
        ) == "FLAGGED"
    )

    total_claims = len(
        verification_results
    )

    verification_rate = (
        verified_count
        / total_claims
        * 100
        if total_claims
        else 0.0
    )

    # --------------------------------------------------------
    # 10. Final result
    # --------------------------------------------------------

    return {

        "status": "completed",

        "meeting_id": (
            audio_path.stem
        ),

        "generator": {

            "name": "Qwen",

            "model": (
                "Qwen/Qwen2.5-3B-Instruct"
            ),

            "strategy": (
                "candidate_level_claim_extraction"
            )
        },

        "retrieval": {

            "model": (
                mp.BGE_MODEL_NAME
            ),

            "top_k": int(
                top_k
            ),

            "evidence_count": len(
                evidence_documents
            )
        },

        "verification_summary": {

            "total_claims": (
                total_claims
            ),

            "verified_claims": (
                verified_count
            ),

            "flagged_claims": (
                flagged_count
            ),

            "verification_rate": (
                verification_rate
            )
        },

        "claims": (
            verification_results
        )
    }


# ============================================================
# UPLOAD NEW MEETING
# ============================================================

st.subheader("🎙️ Upload New Meeting")

st.write(
    "Upload an audio or video recording to generate "
    "verifiable meeting minutes."
)

uploaded_meeting = st.file_uploader(
    "Choose a meeting recording",
    type=[
        "wav",
        "mp3",
        "m4a",
        "mp4",
        "mov",
        "avi",
        "mkv"
    ]
)

if uploaded_meeting is not None:

    st.success(
        f"Selected file: {uploaded_meeting.name}"
    )

    st.write(
        f"File size: "
        f"{uploaded_meeting.size / (1024 * 1024):.2f} MB"
    )

    process_meeting = st.button(
        "▶️ Process Meeting",
        type="primary"
    )

    if process_meeting:

        try:

            # ------------------------------------------------
            # Save uploaded meeting
            # ------------------------------------------------

            saved_path = save_uploaded_meeting(
                uploaded_meeting
            )

            st.success(
                f"Meeting uploaded successfully: "
                f"{saved_path.name}"
            )

            st.write(
                f"Saved to: `{saved_path}`"
            )

            st.write(
                f"File size: "
                f"{saved_path.stat().st_size / (1024 * 1024):.2f} MB"
            )


            # ------------------------------------------------
            # Run validated meeting pipeline
            # ------------------------------------------------

            st.info(
                "Starting AI meeting processing..."
            )



            # ------------------------------------------------
            # Load cached Qwen generator
            # ------------------------------------------------

            st.info(
                "Loading Qwen generator..."
            )

            qwen_model, qwen_tokenizer = (
                load_qwen_generator()
            )

            st.info(
                "Running VAD → WhisperX → "
                "Pyannote → MoM → Evidence → "
                "Verification..."
            )

            pipeline_result = (
                process_uploaded_meeting(
                    audio_path=saved_path,
                    generator="qwen",
                    generator_model=qwen_model,
                    generator_tokenizer=qwen_tokenizer,
                    top_k=10,
                    temporal_tolerance_seconds=5.0
                )
            )


            # ------------------------------------------------
            # Save result
            # ------------------------------------------------

            live_dir = (
                PROJECT_ROOT
                / "outputs"
                / "live"
            )

            live_dir.mkdir(
                parents=True,
                exist_ok=True
            )


            result_path = (
                live_dir
                / (
                    saved_path.stem
                    + "_verified_mom.json"
                )
            )


            with open(
                result_path,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    pipeline_result,
                    f,
                    indent=2,
                    ensure_ascii=False
                )


            # ------------------------------------------------
            # Store result
            # ------------------------------------------------

            st.session_state[
                "uploaded_meeting_result"
            ] = pipeline_result

            st.session_state[
                "uploaded_meeting_result_path"
            ] = str(result_path)


            # ------------------------------------------------
            # Display verification summary
            # ------------------------------------------------

            summary = (
                pipeline_result.get(
                    "verification_summary",
                    {}
                )
            )


            st.success(
                "✓ Meeting processing completed."
            )


            st.subheader(
                "📊 Verification Summary"
            )


            c1, c2, c3, c4 = st.columns(4)


            with c1:

                st.metric(
                    "Total Claims",
                    summary.get(
                        "total_claims",
                        0
                    )
                )


            with c2:

                st.metric(
                    "Verified",
                    summary.get(
                        "verified_claims",
                        0
                    )
                )


            with c3:

                st.metric(
                    "Flagged",
                    summary.get(
                        "flagged_claims",
                        0
                    )
                )


            with c4:

                st.metric(
                    "Verification Rate",
                    f"{summary.get('verification_rate', 0.0):.2f}%"
                )


            # ------------------------------------------------
            # Display generated claims
            # ------------------------------------------------

            live_claims = (
                pipeline_result.get(
                    "claims",
                    []
                )
            )


            if live_claims:

                st.subheader(
                    "📝 Verified Meeting Minutes"
                )


                for idx, claim in enumerate(
                    live_claims,
                    start=1
                ):

                    status = claim.get(
                        "status",
                        "UNKNOWN"
                    )

                    claim_text = claim.get(
                        "claim",
                        ""
                    )

                    speaker = claim.get(
                        "speaker",
                        "Unknown"
                    )

                    event_type = claim.get(
                        "event_type",
                        "Unknown"
                    )


                    with st.expander(
                        f"Claim {idx} — {status}"
                    ):

                        st.markdown(
                            f"### {claim_text}"
                        )

                        st.write(
                            f"**Speaker:** {speaker}"
                        )

                        st.write(
                            f"**Event Type:** {event_type}"
                        )


                        verification = claim.get(
                            "verification",
                            {}
                        )


                        st.write(
                            f"**Content:** "
                            f"{verification.get('content', 'N/A')}"
                        )

                        st.write(
                            f"**Speaker:** "
                            f"{verification.get('speaker', 'N/A')}"
                        )

                        st.write(
                            f"**Event Type:** "
                            f"{verification.get('event_type', 'N/A')}"
                        )

                        st.write(
                            f"**Temporal:** "
                            f"{verification.get('temporal', 'N/A')}"
                        )


                        evidence = claim.get(
                            "evidence",
                            {}
                        )


                        if evidence:

                            st.markdown(
                                "#### 🔎 Supporting Evidence"
                            )

                            st.write(
                                f"**Evidence ID:** "
                                f"{evidence.get('evidence_id', 'N/A')}"
                            )

                            st.write(
                                f"**Evidence Speaker:** "
                                f"{evidence.get('speaker', 'N/A')}"
                            )

                            st.write(
                                f"**Evidence Time:** "
                                f"{format_timestamp(evidence.get('start'))}"
                                f" – "
                                f"{format_timestamp(evidence.get('end'))}"
                            )

                            st.write(
                                f"**Retrieval Similarity:** "
                                f"{evidence.get('retrieval_similarity', 'N/A')}"
                            )

                            st.write(
                                f"**NLI:** "
                                f"{evidence.get('nli_label', 'N/A')}"
                            )

                            st.write(
                                evidence.get(
                                    "text",
                                    ""
                                )
                            )


            # ------------------------------------------------
            # Download JSON
            # ------------------------------------------------

            st.download_button(
                "⬇️ Download Verified MoM JSON",
                data=json.dumps(
                    pipeline_result,
                    indent=2,
                    ensure_ascii=False
                ),
                file_name=(
                    saved_path.stem
                    + "_verified_mom.json"
                ),
                mime="application/json",
                use_container_width=True
            )


        except Exception as e:

            st.error(
                f"Meeting processing failed: {e}"
            )

            st.exception(e)



# ============================================================
# MEETING INFORMATION
# ============================================================

st.subheader("Meeting")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Meeting ID",
        claims_data.get("meeting_id", "Unknown")
    )

with col2:
    st.metric(
        "Total Claims",
        len(dashboard_rows)
    )


# ============================================================
# SUMMARY METRICS
# ============================================================

verified_count = sum(
    1
    for row in dashboard_rows
    if row["status"] == "VERIFIED"
)

flagged_count = sum(
    1
    for row in dashboard_rows
    if row["status"] != "VERIFIED"
)

verification_rate = (
    verified_count / len(dashboard_rows) * 100
    if dashboard_rows
    else 0
)


st.subheader("Verification Summary")

m1, m2, m3 = st.columns(3)

with m1:
    st.metric(
        "Total Claims",
        len(dashboard_rows)
    )

with m2:
    st.metric(
        "Verified",
        verified_count
    )

with m3:
    st.metric(
        "Verification Rate",
        f"{verification_rate:.2f}%"
    )


# ============================================================
# MEETING MINUTES
# ============================================================

st.subheader("Meeting Minutes")

for row in dashboard_rows:

    if row["status"] == "VERIFIED":
        status_label = "✅ VERIFIED"
    else:
        status_label = "⚠️ FLAGGED"

    with st.container(border=True):

        st.markdown(
            f"### {row['claim_id']}. {row['claim_text']}"
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.write(
                f"**Speaker:** {row['speaker']}"
            )

        with c2:
            st.write(
                f"**Event Type:** {row['event_type']}"
            )

        with c3:
            st.write(
                f"**Status:** {status_label}"
            )

        st.caption(
            f"Timestamp: "
            f"{row['start']:.2f}s – "
            f"{row['end']:.2f}s"
        )

        if row["reason"]:
            st.write(
                f"**Verification:** {row['reason']}"
            )

        # ----------------------------------------------------
        # Verification details
        # ----------------------------------------------------

        checks = row["checks"]

        if checks:

            with st.expander("🔍 View verification details"):

                content_check = checks.get(
                    "content_consistency",
                    {}
                )

                speaker_check = checks.get(
                    "speaker_consistency",
                    {}
                )

                timestamp_check = checks.get(
                    "timestamp_consistency",
                    {}
                )

                event_check = checks.get(
                    "event_type_consistency",
                    {}
                )

                st.markdown("#### Verification Checks")

                v1, v2, v3, v4 = st.columns(4)

                with v1:
                    if content_check.get("passed"):
                        st.success("Content\nPASS")
                    else:
                        st.error("Content\nFAIL")

                with v2:
                    if speaker_check.get("passed"):
                        st.success("Speaker\nPASS")
                    else:
                        st.error("Speaker\nFAIL")

                with v3:
                    if timestamp_check.get("passed"):
                        st.success("Timestamp\nPASS")
                    else:
                        st.error("Timestamp\nFAIL")

                with v4:
                    if event_check.get("passed"):
                        st.success("Event Type\nPASS")
                    else:
                        st.error("Event Type\nFAIL")

                st.markdown("#### Verification Reason")

                st.write(
                    row["reason"]
                    if row["reason"]
                    else "No verification reason available."
                )

                st.markdown("#### Evidence")

                if row["evidence_ids"]:

                    st.write(
                        "**Evidence IDs:**",
                        ", ".join(
                            str(x)
                            for x in row["evidence_ids"]
                        )
                    )

                    if row["evidence_start"] is not None:
                        st.write(
                            f"**Evidence time range:** "
                            f"{row['evidence_start']:.2f}s – "
                            f"{row['evidence_end']:.2f}s"
                        )

                else:
                    st.info(
                        "No supporting evidence was recorded."
                    )


                
                # ------------------------------------------------
                # Retrieved Evidence
                # ------------------------------------------------

                st.markdown("#### 📚 Retrieved Evidence")

                evidence_ids = row["evidence_ids"]

                if evidence_ids:

                    for evidence_id in evidence_ids:

                        evidence = verified_evidence_by_id.get(
                            evidence_id
                        )

                        if evidence is None:
                            continue

                        with st.container(border=True):

                            st.markdown(
                                f"**Evidence ID: {evidence_id}**"
                            )

                            st.write(
                                evidence.get(
                                    "text",
                                    "Evidence text unavailable."
                                )
                            )

                            e1, e2, e3 = st.columns(3)

                            with e1:
                                st.write(
                                    f"**Speaker:** "
                                    f"{evidence.get('speaker', 'Unknown')}"
                                )

                            with e2:
                                start = evidence.get("start")
                                end = evidence.get("end")

                                if (
                                    start is not None
                                    and end is not None
                                ):
                                    st.write(
                                        f"**Timestamp:** "
                                        f"{start:.2f}s – {end:.2f}s"
                                    )
                                else:
                                    st.write(
                                        "**Timestamp:** Not available"
                                    )

                            with e3:
                                similarity = evidence.get(
                                    "retrieval_similarity"
                                )

                                if similarity is not None:
                                    st.write(
                                        f"**Similarity:** "
                                        f"{similarity:.4f}"
                                    )
                                else:
                                    st.write(
                                        "**Similarity:** Not available"
                                    )

                            e4, e5 = st.columns(2)

                            with e4:
                                nli_label = evidence.get(
                                    "nli_label",
                                    "unknown"
                                )

                                st.write(
                                    f"**NLI:** {nli_label}"
                                )

                            with e5:
                                speaker_match = evidence.get(
                                    "speaker_match"
                                )

                                st.write(
                                    f"**Speaker Match:** "
                                    f"{speaker_match}"
                                )

                else:

                    st.info(
                        "No supporting evidence was recorded."
                    )



# ============================================================
# PDF
# ============================================================

st.subheader("Generated PDF")

if PDF_PATH.exists():

    with open(PDF_PATH, "rb") as pdf_file:

        st.download_button(
            label="📄 Download Verified Meeting Minutes PDF",
            data=pdf_file,
            file_name="ES2004a_Verifiable_Meeting_Minutes.pdf",
            mime="application/pdf"
        )

else:

    st.info(
        "Verified PDF is not available."
    )
