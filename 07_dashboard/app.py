
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
# D10-6 PROCESSED MEETING LOADER
# ============================================================

LIVE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "live"
)

RAW_MEETING_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
)


def format_timestamp(value):
    """
    Convert seconds into HH:MM:SS.mmm.
    Used for evidence and claim timestamps.
    """

    if value is None:
        return "N/A"

    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return str(value)

    if seconds < 0:
        seconds = 0

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:06.3f}"
    )


def discover_processed_meetings():
    """
    Find previously processed meeting results.

    Supported result formats:
        *_verified_mom.json
        *_D9V2_result.json
    """

    LIVE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    files = []

    for pattern in (
        "*_verified_mom.json",
        "*_D9V2_result.json",
    ):

        files.extend(
            LIVE_DIR.glob(pattern)
        )

    # Remove duplicates
    unique_files = {}

    for path in files:
        unique_files[str(path.resolve())] = path

    records = []

    for path in sorted(
        unique_files.values(),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    ):

        try:

            with open(
                path,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

            meeting_id = (
                data.get("meeting_id")
                or path.stem
            )

            if path.stem.endswith(
                "_verified_mom"
            ):

                result_type = (
                    "Verified MoM"
                )

            elif path.stem.endswith(
                "_D9V2_result"
            ):

                result_type = (
                    "D9 V2 Result"
                )

            else:

                result_type = (
                    "Processed Result"
                )

            records.append({
                "path": path,
                "meeting_id": meeting_id,
                "result_type": result_type,
                "modified": path.stat().st_mtime,
                "data": data,
            })

        except Exception:
            # Ignore malformed/unrelated JSON files.
            continue

    return records


def load_processed_meeting(
    result_path
):
    """
    Load a saved meeting result.

    IMPORTANT:
    This function does NOT run any AI model.
    """

    result_path = Path(
        result_path
    )

    if not result_path.exists():

        raise FileNotFoundError(
            f"Processed result not found:\n"
            f"{result_path}"
        )

    with open(
        result_path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def normalize_processed_meeting_result(
    result
):
    """
    Normalize current and D9 result formats
    into a dashboard-friendly structure.
    """

    # --------------------------------------------------------
    # Claims
    # --------------------------------------------------------

    claims = result.get(
        "claims",
        []
    )

    if not claims:

        claims = result.get(
            "results",
            []
        )

    if not isinstance(
        claims,
        list
    ):

        claims = []


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = result.get(
        "verification_summary",
        {}
    )

    if not isinstance(
        summary,
        dict
    ):

        summary = {}


    total = summary.get(
        "total_claims"
    )

    verified = summary.get(
        "verified_claims"
    )

    flagged = summary.get(
        "flagged_claims"
    )

    if total is None:
        total = len(claims)

    if verified is None:

        verified = sum(
            1
            for claim in claims
            if str(
                claim.get(
                    "status",
                    ""
                )
            ).upper()
            == "VERIFIED"
        )

    if flagged is None:
        flagged = total - verified

    if summary.get(
        "verification_rate"
    ) is not None:

        verification_rate = float(
            summary.get(
                "verification_rate"
            )
        )

    else:

        verification_rate = (
            verified / total * 100
            if total
            else 0.0
        )


    # --------------------------------------------------------
    # Normalize individual claims
    # --------------------------------------------------------

    normalized_claims = []

    for index, claim in enumerate(
        claims,
        start=1
    ):

        if not isinstance(
            claim,
            dict
        ):
            continue


        claim_text = (
            claim.get("claim")
            or claim.get("claim_text")
            or claim.get("text")
            or ""
        )


        speaker = (
            claim.get("speaker")
            or "Unknown"
        )


        start = claim.get(
            "start"
        )

        end = claim.get(
            "end"
        )


        event_type = (
            claim.get("event_type")
            or "Unknown"
        )


        status = (
            claim.get("status")
            or claim.get("overall_status")
            or "UNKNOWN"
        )


        verification = claim.get(
            "verification",
            {}
        )

        if not isinstance(
            verification,
            dict
        ):
            verification = {}


        # ----------------------------------------------------
        # Evidence
        # ----------------------------------------------------

        evidence = claim.get(
            "evidence"
        )

        if not evidence:

            evidence = claim.get(
                "selected_evidence"
            )

        if not evidence:

            evidence = {}

        if isinstance(
            evidence,
            list
        ):

            evidence = (
                evidence[0]
                if evidence
                else {}
            )

        if not isinstance(
            evidence,
            dict
        ):

            evidence = {}


        normalized_claims.append({

            "number": index,

            "claim": claim_text,

            "speaker": speaker,

            "start": start,

            "end": end,

            "event_type": event_type,

            "status": str(
                status
            ).upper(),

            "verification": verification,

            "evidence": evidence,

            "evidence_ids": (
                claim.get(
                    "evidence_ids",
                    []
                )
                or []
            ),

            "raw": claim,
        })


    return {

        "meeting_id": (
            result.get(
                "meeting_id",
                "Unknown"
            )
        ),

        "status": result.get(
            "status",
            "completed"
        ),

        "summary": {

            "total_claims": int(
                total
            ),

            "verified_claims": int(
                verified
            ),

            "flagged_claims": int(
                flagged
            ),

            "verification_rate": (
                verification_rate
            ),
        },

        "claims": normalized_claims,

        "generator": result.get(
            "generator",
            {}
        ),

        "retrieval": result.get(
            "retrieval",
            {}
        ),

        "raw": result,
    }


def find_processed_meeting_audio(
    meeting_id,
    result_path
):
    """
    Find the original audio/video associated
    with a saved processed result.
    """

    if not RAW_MEETING_DIR.exists():
        return None


    meeting_id = str(
        meeting_id
    )


    # Candidate base names
    base_names = [
        meeting_id,
        Path(result_path).stem,
    ]


    suffixes = [
        "_verified_mom",
        "_D9V2_result",
        "_result",
    ]


    expanded_names = []

    for name in base_names:

        expanded_names.append(
            name
        )

        for suffix in suffixes:

            if name.endswith(
                suffix
            ):

                expanded_names.append(
                    name[
                        :-len(suffix)
                    ]
                )


    # Exact stem matching
    for path in RAW_MEETING_DIR.iterdir():

        if not path.is_file():
            continue

        if path.suffix.lower() not in {
            ".wav",
            ".mp3",
            ".m4a",
            ".mp4",
            ".mov",
            ".avi",
            ".mkv",
        }:
            continue

        if path.stem in expanded_names:

            return path


    # Fallback: meeting ID contained in filename
    for path in RAW_MEETING_DIR.iterdir():

        if not path.is_file():
            continue

        if path.suffix.lower() not in {
            ".wav",
            ".mp3",
            ".m4a",
            ".mp4",
            ".mov",
            ".avi",
            ".mkv",
        }:
            continue

        if meeting_id in path.stem:

            return path


    return None


@st.cache_data(show_spinner=False)
def extract_evidence_audio_clip(audio_path_str, start_seconds, end_seconds):
    """Return only the requested evidence interval as MP3 bytes."""
    import os
    import subprocess
    import tempfile

    audio_path = Path(audio_path_str)
    start_seconds = max(float(start_seconds), 0.0)
    end_seconds = float(end_seconds)

    if not audio_path.exists() or end_seconds <= start_seconds:
        return None

    fd, output_path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)

    try:
        duration = end_seconds - start_seconds
        command = [
            "ffmpeg", "-y",
            "-ss", f"{start_seconds:.3f}",
            "-i", str(audio_path),
            "-t", f"{duration:.3f}",
            "-vn", "-ac", "1",
            "-c:a", "libmp3lame", "-q:a", "5",
            output_path,
        ]
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0 or not Path(output_path).exists():
            return None
        return Path(output_path).read_bytes()
    finally:
        try:
            Path(output_path).unlink(missing_ok=True)
        except Exception:
            pass


def set_active_evidence_player(key, is_active):
    st.session_state["active_evidence_player"] = (
        None if is_active else key
    )


def render_audio_evidence_player(
    audio_path,
    start,
    end,
    key,
    evidence_label="Evidence",
    speaker="N/A",
):
    """Render one horizontal evidence row with optional inline playback."""
    if audio_path is None:
        st.caption("Original meeting recording is not available.")
        return

    try:
        start_seconds = max(float(start), 0.0)
        end_seconds = float(end)
    except (TypeError, ValueError):
        st.caption("Evidence timestamp is not available.")
        return

    if end_seconds <= start_seconds:
        st.caption("Invalid evidence time range.")
        return

    audio_path = Path(audio_path)
    if not audio_path.exists():
        st.caption("Original meeting recording is not available.")
        return

    if "active_evidence_player" not in st.session_state:
        st.session_state["active_evidence_player"] = None

    is_active = (
        st.session_state["active_evidence_player"] == key
    )

    # Exact requested layout:
    # Evidence | Speaker | Timestamp | ▶ | Player
    c1, c2, c3, c4, c5 = st.columns(
        [0.85, 1.0, 1.55, 0.28, 2.4],
        vertical_alignment="center",
    )

    with c1:
        st.write(f"**{evidence_label}**")

    with c2:
        st.write(f"**{speaker}**")

    with c3:
        st.write(
            f"**{format_timestamp(start_seconds)} – "
            f"{format_timestamp(end_seconds)}**"
        )

    with c4:
        st.button(
            "⏸" if is_active else "▶",
            key=f"evidence_play_{key}",
            help=(
                "Hide evidence audio"
                if is_active
                else "Play this exact evidence segment"
            ),
            on_click=set_active_evidence_player,
            args=(key, is_active),
        )


    with c5:
        if is_active:
            try:
                clip_bytes = extract_evidence_audio_clip(
                    str(audio_path),
                    start_seconds,
                    end_seconds,
                )
            except Exception as exc:
                st.warning(
                    f"Could not prepare evidence audio: {exc}"
                )
                return

            if clip_bytes:
                st.audio(
                    clip_bytes,
                    format="audio/mp3",
                )
            else:
                st.warning(
                    "Could not prepare this evidence audio segment."
                )


def render_processed_meeting(
    result,
    result_path,
    view_mode="processed"
):
    """
    Render a previously processed meeting.
    No AI inference is performed here.
    """

    normalized = (
        normalize_processed_meeting_result(
            result
        )
    )

    meeting_id = normalized[
        "meeting_id"
    ]

    summary = normalized[
        "summary"
    ]

    claims = normalized[
        "claims"
    ]


    if view_mode == "live":

        st.subheader(
            "🎙️ Newly Processed Meeting"
        )

        st.success(
            "✓ Meeting processing completed successfully."
        )

    else:

        st.subheader(
            "📂 Processed Meeting"
        )

       
    st.caption(
        f"Result file: {Path(result_path).name}"
    )


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Meeting",
            meeting_id
        )

    with c2:
        st.metric(
            "Claims",
            summary["total_claims"]
        )

    with c3:
        st.metric(
            "Verified",
            summary["verified_claims"]
        )

    with c4:
        st.metric(
            "Verification Rate",
            f"{summary['verification_rate']:.2f}%"
        )


    st.write(
        f"**Flagged Claims:** "
        f"{summary['flagged_claims']}"
    )


    # --------------------------------------------------------
    # Generator / retrieval information
    # --------------------------------------------------------

    generator = normalized.get(
        "generator",
        {}
    )

    retrieval = normalized.get(
        "retrieval",
        {}
    )

    # Processing Information is shown only for a newly processed
    # live meeting. Saved/previously processed meetings should keep
    # the dashboard focused on the meeting results and evidence.
    if view_mode == "live" and (generator or retrieval):

        with st.expander(
            "⚙️ Processing Information"
        ):

            if generator:

                st.write(
                    "**Generator:**",
                    generator
                )

            if retrieval:

                st.write(
                    "**Retrieval:**",
                    retrieval
                )


    # --------------------------------------------------------
    # Original audio
    # --------------------------------------------------------

    audio_path = (
        find_processed_meeting_audio(
            meeting_id,
            result_path
        )
    )

    st.markdown(
        "### 🎧 Original Meeting Recording"
    )

    if audio_path is not None:

       
        st.audio(
            str(audio_path)
        )

    else:

        st.info(
            "Original audio/video was not found "
            "in data/raw."
        )


    # --------------------------------------------------------
    # Claims
    # --------------------------------------------------------

    st.markdown(
        "### 📝 Verified Meeting Minutes"
    )


    if not claims:

        st.warning(
            "No claims were found in this "
            "processed result."
        )

        return


    for claim in claims:

        status = claim[
            "status"
        ]

        if status == "VERIFIED":

            status_label = (
                "✅ VERIFIED"
            )

        elif status == "FLAGGED":

            status_label = (
                "⚠️ FLAGGED"
            )

        else:

            status_label = (
                f"ℹ️ {status}"
            )


        with st.container(
            border=True
        ):

            st.markdown(
                f"### Claim "
                f"{claim['number']} — "
                f"{status_label}"
            )

            st.write(
                f"**Claim:** "
                f"{claim['claim']}"
            )


            c1, c2, c3 = st.columns(3)


            with c1:

                st.write(
                    f"**Speaker:** "
                    f"{claim['speaker']}"
                )


            with c2:

                st.write(
                    f"**Event Type:** "
                    f"{claim['event_type']}"
                )


            with c3:

                st.write(
                    f"**Claim Time:** "
                    f"{format_timestamp(claim['start'])}"
                    f" – "
                    f"{format_timestamp(claim['end'])}"
                )


            # ------------------------------------------------
            # Verification
            # ------------------------------------------------

            verification = (
                claim["verification"]
            )

            if verification:

                st.markdown(
                    "#### 🔍 Verification"
                )

                verification_cols = st.columns(
                    min(
                        max(
                            len(verification),
                            1
                        ),
                        4
                    )
                )

                items = list(
                    verification.items()
                )

                for col, (
                    key,
                    value
                ) in zip(
                    verification_cols,
                    items
                ):

                    with col:

                        st.write(
                            f"**{key.replace('_', ' ').title()}:**"
                        )

                        st.write(
                            str(value)
                        )


            # ------------------------------------------------
            # Evidence
            # ------------------------------------------------

            evidence = (
                claim["evidence"]
            )

            evidence_ids = (
                claim["evidence_ids"]
            )


            if evidence or evidence_ids:

                st.markdown(
                    "#### 📚 Supporting Evidence"
                )


                if evidence_ids:

                    st.write(
                        "**Evidence IDs:**",
                        ", ".join(
                            str(x)
                            for x in evidence_ids
                        )
                    )


                if evidence:

                    evidence_text = (
                        evidence.get(
                            "text",
                            ""
                        )
                    )

                    if evidence_text:

                        st.info(
                            evidence_text
                        )


                    render_audio_evidence_player(
                        audio_path=audio_path,
                        start=evidence.get("start"),
                        end=evidence.get("end"),
                        speaker=evidence.get("speaker", "N/A"),
                        evidence_label=(
                            f"Evidence "
                            f"{evidence.get('evidence_id', 'N/A')}"
                        ),
                        key=(
                            f"processed_evidence_"
                            f"{meeting_id}_"
                            f"{claim['number']}_"
                            f"{evidence.get('evidence_id', 'na')}"
                        ),
                    )

                    similarity = evidence.get(
                        "retrieval_similarity"
                    )

                    if similarity is not None:
                        st.caption(
                            f"Retrieval Similarity: "
                            f"{float(similarity):.4f}"
                        )
                    else:
                        st.caption("Retrieval Similarity: N/A")

                    nli_label = evidence.get(
                        "nli_label"
                    )

                    if nli_label is not None:

                        st.write(
                            f"**NLI:** {nli_label}"
                        )


            else:

                st.info(
                    "No supporting evidence "
                    "was recorded."
                )


    # --------------------------------------------------------
    # Download saved result
    # --------------------------------------------------------





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



def render_legacy_es2004a_dashboard():
    """
    Render the existing ES2004a dashboard directly.

    This replaces the broken D10-7 string/exec approach. No AI
    inference is performed here; all information comes from the
    already-loaded ES2004a claims and verification artifacts.
    """

    legacy_audio_path = find_processed_meeting_audio(
        "ES2004a",
        "ES2004a"
    )

    st.markdown("### 📊 ES2004a Meeting Results")

    total = len(dashboard_rows)
    verified = sum(
        1 for row in dashboard_rows
        if str(row.get("status", "")).upper() == "VERIFIED"
    )
    flagged = total - verified
    rate = (verified / total * 100) if total else 0.0

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric("Total Claims", total)

    with m2:
        st.metric("Verified", verified)

    with m3:
        st.metric("Flagged", flagged)

    with m4:
        st.metric("Verification Rate", f"{rate:.2f}%")

    st.markdown("### 📝 Meeting Claims")

    for row in dashboard_rows:
        status = str(row.get("status", "UNKNOWN")).upper()

        if status == "VERIFIED":
            status_label = "✅ VERIFIED"
        elif status == "FLAGGED":
            status_label = "⚠️ FLAGGED"
        else:
            status_label = f"ℹ️ {status}"

        claim_number = row.get("claim_id") or row.get("number") or ""
        claim_text = row.get("claim_text", "")
        speaker = row.get("speaker", "Unknown")
        event_type = row.get("event_type", "Unknown")
        start = row.get("start")
        end = row.get("end")
        reason = row.get("reason", "")
        checks = row.get("checks") or {}
        evidence_ids = row.get("evidence_ids") or []

        with st.container(border=True):
            st.markdown(
                f"### Claim {claim_number} — {status_label}"
            )

            st.write(f"**Claim:** {claim_text}")

            c1, c2, c3 = st.columns(3)

            with c1:
                st.write(f"**Speaker:** {speaker}")

            with c2:
                st.write(f"**Event Type:** {event_type}")

            with c3:
                st.write(
                    f"**Claim Time:** "
                    f"{format_timestamp(start)} – "
                    f"{format_timestamp(end)}"
                )

            st.markdown("#### 🔍 Verification")

            if reason:
                st.write(f"**Reason:** {reason}")

            if checks:
                check_cols = st.columns(
                    min(max(len(checks), 1), 4)
                )

                for col, (check_name, check_value) in zip(
                    check_cols,
                    checks.items()
                ):
                    with col:
                        if isinstance(check_value, dict):
                            passed = check_value.get("passed")
                            if passed is True:
                                display_value = "PASSED"
                            elif passed is False:
                                display_value = "FAILED"
                            else:
                                display_value = str(check_value)
                        else:
                            display_value = str(check_value)

                        st.write(
                            f"**{str(check_name).replace('_', ' ').title()}:** "
                            f"{display_value}"
                        )

            st.markdown("#### 📚 Supporting Evidence")

            evidence_found = False

            # Prefer the verified evidence candidates already built
            # by the existing dashboard compatibility layer.
            for evidence_id in evidence_ids:
                evidence = verified_evidence_by_id.get(evidence_id)

                if not evidence:
                    continue

                evidence_found = True

                evidence_speaker = evidence.get(
                    "speaker", "N/A"
                )
                evidence_start = evidence.get("start")
                evidence_end = evidence.get("end")
                evidence_text = evidence.get("text", "")
                similarity = evidence.get(
                    "retrieval_similarity"
                )

                st.info(
                    f"**Evidence {evidence_id}** | "
                    f"{evidence_speaker} | "
                    f"{format_timestamp(evidence_start)} – "
                    f"{format_timestamp(evidence_end)}\n\n"
                    f"{evidence_text}"
                )

                render_audio_evidence_player(
                    audio_path=legacy_audio_path,
                    start=evidence_start,
                    end=evidence_end,
                    speaker=evidence_speaker,
                    evidence_label=f"Evidence {evidence_id}",
                    key=f"legacy_evidence_{claim_number}_{evidence_id}",
                )

                if similarity is not None:
                    st.write(
                        f"**Retrieval Similarity:** "
                        f"{float(similarity):.4f}"
                    )

            # Some legacy verification records contain supporting
            # candidates inside checks rather than evidence_ids.
            if not evidence_found:
                content_check = checks.get(
                    "content_consistency", {}
                )

                if isinstance(content_check, dict):
                    details = content_check.get(
                        "details", {}
                    )

                    supporting_candidates = (
                        details.get(
                            "supporting_candidates",
                            []
                        )
                        if isinstance(details, dict)
                        else []
                    )

                    for evidence in supporting_candidates:
                        evidence_found = True

                        evidence_id = evidence.get(
                            "evidence_id", "N/A"
                        )
                        evidence_speaker = evidence.get(
                            "speaker", "N/A"
                        )
                        evidence_start = evidence.get("start")
                        evidence_end = evidence.get("end")
                        evidence_text = evidence.get(
                            "text", ""
                        )
                        similarity = evidence.get(
                            "retrieval_similarity"
                        )

                        st.info(
                            f"**Evidence {evidence_id}** | "
                            f"{evidence_speaker} | "
                            f"{format_timestamp(evidence_start)} – "
                            f"{format_timestamp(evidence_end)}\n\n"
                            f"{evidence_text}"
                        )

                        render_audio_evidence_player(
                            audio_path=legacy_audio_path,
                            start=evidence_start,
                            end=evidence_end,
                            speaker=evidence_speaker,
                            evidence_label=f"Evidence {evidence_id}",
                            key=f"legacy_fallback_evidence_{claim_number}_{evidence_id}",
                        )

                        if similarity is not None:
                            st.caption(
                                f"Retrieval similarity: "
                                f"{float(similarity):.4f}"
                            )

            if not evidence_found:
                st.info(
                    "No supporting evidence was recorded "
                    "for this claim."
                )


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
# D10-7 GENERIC PDF EXPORT
# ============================================================

def create_context_pdf(
    result,
    output_path
):
    """
    Create a PDF for a processed/live meeting result.

    This function is UI/export only.
    It does NOT run any AI model.
    """

    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        PageBreak
    )
    from xml.sax.saxutils import escape

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    normalized = normalize_processed_meeting_result(
        result
    )

    meeting_id = str(
        normalized.get(
            "meeting_id",
            "Meeting"
        )
    )

    summary = normalized.get(
        "summary",
        {}
    )

    claims = normalized.get(
        "claims",
        []
    )

    styles = getSampleStyleSheet()

    title_style = styles["Title"]
    heading_style = styles["Heading2"]
    body_style = styles["BodyText"]

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm
    )

    story = []

    story.append(
        Paragraph(
            "Verifiable Meeting Minutes",
            title_style
        )
    )

    story.append(
        Spacer(
            1,
            8
        )
    )

    story.append(
        Paragraph(
            f"<b>Meeting:</b> "
            f"{escape(meeting_id)}",
            body_style
        )
    )

    story.append(
        Paragraph(
            f"<b>Total Claims:</b> "
            f"{summary.get('total_claims', 0)}",
            body_style
        )
    )

    story.append(
        Paragraph(
            f"<b>Verified:</b> "
            f"{summary.get('verified_claims', 0)}",
            body_style
        )
    )

    story.append(
        Paragraph(
            f"<b>Flagged:</b> "
            f"{summary.get('flagged_claims', 0)}",
            body_style
        )
    )

    story.append(
        Paragraph(
            f"<b>Verification Rate:</b> "
            f"{summary.get('verification_rate', 0):.2f}%",
            body_style
        )
    )

    story.append(
        Spacer(
            1,
            12
        )
    )

    story.append(
        Paragraph(
            "Verified Meeting Minutes",
            heading_style
        )
    )

    for claim in claims:

        claim_number = claim.get(
            "number",
            ""
        )

        claim_text = escape(
            str(
                claim.get(
                    "claim",
                    ""
                )
            )
        )

        speaker = escape(
            str(
                claim.get(
                    "speaker",
                    "Unknown"
                )
            )
        )

        event_type = escape(
            str(
                claim.get(
                    "event_type",
                    "Unknown"
                )
            )
        )

        status = escape(
            str(
                claim.get(
                    "status",
                    "UNKNOWN"
                )
            )
        )

        start = claim.get(
            "start"
        )

        end = claim.get(
            "end"
        )

        story.append(
            Paragraph(
                f"<b>Claim {claim_number} "
                f"— {status}</b>",
                heading_style
            )
        )

        story.append(
            Paragraph(
                f"<b>Claim:</b> "
                f"{claim_text}",
                body_style
            )
        )

        story.append(
            Paragraph(
                f"<b>Speaker:</b> "
                f"{speaker}",
                body_style
            )
        )

        story.append(
            Paragraph(
                f"<b>Event Type:</b> "
                f"{event_type}",
                body_style
            )
        )

        if start is not None:

            story.append(
                Paragraph(
                    f"<b>Claim Time:</b> "
                    f"{format_timestamp(start)}"
                    f" – "
                    f"{format_timestamp(end)}",
                    body_style
                )
            )

        evidence = claim.get(
            "evidence",
            {}
        )

        if isinstance(
            evidence,
            dict
        ) and evidence:

            evidence_text = escape(
                str(
                    evidence.get(
                        "text",
                        ""
                    )
                )
            )

            evidence_speaker = escape(
                str(
                    evidence.get(
                        "speaker",
                        "Unknown"
                    )
                )
            )

            evidence_start = evidence.get(
                "start"
            )

            evidence_end = evidence.get(
                "end"
            )

            story.append(
                Paragraph(
                    "<b>Supporting Evidence</b>",
                    body_style
                )
            )

            if evidence_text:

                story.append(
                    Paragraph(
                        evidence_text,
                        body_style
                    )
                )

            story.append(
                Paragraph(
                    f"<b>Evidence Speaker:</b> "
                    f"{evidence_speaker}",
                    body_style
                )
            )

            if evidence_start is not None:

                story.append(
                    Paragraph(
                        f"<b>Evidence Time:</b> "
                        f"{format_timestamp(evidence_start)}"
                        f" – "
                        f"{format_timestamp(evidence_end)}",
                        body_style
                    )
                )

        story.append(
            Spacer(
                1,
                10
            )
        )

    document.build(
        story
    )

    return output_path


def get_active_pdf_info():

    """
    Return:
        pdf_path,
        display_name,
        is_available
    """

    active_view = st.session_state.get(
        "active_view",
        "legacy"
    )

    # --------------------------------------------------------
    # Legacy ES2004a
    # --------------------------------------------------------

    if active_view == "legacy":

        if PDF_PATH.exists():

            return (
                PDF_PATH,
                "ES2004a_Verifiable_Meeting_Minutes.pdf",
                True
            )

        return (
            None,
            "ES2004a_Verifiable_Meeting_Minutes.pdf",
            False
        )

    # --------------------------------------------------------
    # Processed / live meeting
    # --------------------------------------------------------

    if active_view == "live":

        result = st.session_state.get(
            "active_live_result"
        )

        result_path = st.session_state.get(
            "active_live_result_path"
        )

    else:

        result = st.session_state.get(
            "active_processed_result"
        )

        result_path = st.session_state.get(
            "active_processed_meeting"
        )

    if not result:

        return (
            None,
            "Verifiable_Meeting_Minutes.pdf",
            False
        )

    meeting_id = str(
        result.get(
            "meeting_id",
            "Meeting"
        )
    )

    pdf_dir = (
        PROJECT_ROOT
        / "outputs"
        / "pdf"
    )

    pdf_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    pdf_path = (
        pdf_dir
        / (
            f"{meeting_id}_"
            "Verifiable_Meeting_Minutes.pdf"
        )
    )

    return (
        pdf_path,
        pdf_path.name,
        True
    )




# ============================================================
# D10-7 DASHBOARD STATE
# ============================================================

if "active_view" not in st.session_state:

    # Default view:
    # Existing ES2004a dashboard
    st.session_state[
        "active_view"
    ] = "legacy"


if "active_processed_meeting" not in st.session_state:
    st.session_state[
        "active_processed_meeting"
    ] = None


if "active_processed_result" not in st.session_state:
    st.session_state[
        "active_processed_result"
    ] = None


if "active_live_result" not in st.session_state:
    st.session_state[
        "active_live_result"
    ] = None


if "active_live_result_path" not in st.session_state:
    st.session_state[
        "active_live_result_path"
    ] = None


# ============================================================
# DYNAMIC PROCESSED MEETINGS
# ============================================================

processed_records = (
    discover_processed_meetings()
)


# ------------------------------------------------------------
# Deduplicate by meeting ID
#
# If both *_verified_mom.json and *_D9V2_result.json exist,
# prefer Verified MoM.
# ------------------------------------------------------------

unique_processed = {}

for record in processed_records:

    meeting_id_key = str(
        record.get(
            "meeting_id",
            record["path"].stem
        )
    )

    existing = unique_processed.get(
        meeting_id_key
    )

    if existing is None:

        unique_processed[
            meeting_id_key
        ] = record

    elif (
        record.get("result_type")
        == "Verified MoM"
    ):

        unique_processed[
            meeting_id_key
        ] = record


processed_records = list(
    unique_processed.values()
)


# ============================================================
# TOP OPERATING MODES
# ============================================================

implemented_col, process_col = st.columns(
    2,
    gap="large"
)


# ============================================================
# LEFT: IMPLEMENTED / PROCESSED MEETINGS
# ============================================================

with implemented_col:

    st.markdown(
        "## 📂 Implemented Meetings"
    )

    st.caption(
        "Select an existing meeting. "
        "Saved meetings load instantly without "
        "rerunning the AI pipeline."
    )


    # --------------------------------------------------------
    # Dropdown options
    # --------------------------------------------------------

    meeting_options = []

    # Always available
    meeting_options.append(
        {
            "kind": "legacy",
            "label": "ES2004a",
            "record": None,
        }
    )


    # Current live result gets a special entry
    live_result_path = (
        st.session_state.get(
            "active_live_result_path"
        )
    )

    live_result = (
        st.session_state.get(
            "active_live_result"
        )
    )

    if (
        st.session_state.get(
            "active_view"
        ) == "live"
        and live_result_path
        and live_result
    ):

        live_id = str(
            live_result.get(
                "meeting_id",
                Path(
                    live_result_path
                ).stem
            )
        )

        meeting_options.append(
            {
                "kind": "live",
                "label": (
                    f"🎙️ {live_id} "
                    "— Current Live Result"
                ),
                "record": None,
            }
        )


    # Saved meetings
    for record in processed_records:

        label = (
            f"{record['meeting_id']} "
            f"— {record['result_type']}"
        )

        meeting_options.append(
            {
                "kind": "processed",
                "label": label,
                "record": record,
            }
        )


    # --------------------------------------------------------
    # Determine selected/default item
    # --------------------------------------------------------

    active_view = st.session_state.get(
        "active_view",
        "legacy"
    )

    default_index = 0


    if active_view == "live":

        for index, option in enumerate(
            meeting_options
        ):

            if option["kind"] == "live":

                default_index = index
                break


    elif active_view == "processed":

        active_path = (
            st.session_state.get(
                "active_processed_meeting"
            )
        )

        if active_path:

            active_path = str(
                Path(active_path).resolve()
            )

            for index, option in enumerate(
                meeting_options
            ):

                record = option.get(
                    "record"
                )

                if not record:
                    continue

                record_path = str(
                    Path(
                        record["path"]
                    ).resolve()
                )

                if (
                    record_path
                    == active_path
                ):

                    default_index = index
                    break


    selected_index = st.selectbox(
        "Select Meeting",
        range(
            len(meeting_options)
        ),
        index=default_index,
        format_func=lambda i:
            meeting_options[i]["label"],
        key="implemented_meeting_selector"
    )


    selected_option = (
        meeting_options[
            selected_index
        ]
    )


    selected_kind = (
        selected_option["kind"]
    )


    # --------------------------------------------------------
    # Selection behavior
    # --------------------------------------------------------

    if selected_kind == "legacy":

        st.session_state[
            "active_view"
        ] = "legacy"

        active_view = "legacy"


    elif selected_kind == "live":

        st.session_state[
            "active_view"
        ] = "live"

        active_view = "live"


    elif selected_kind == "processed":

        record = (
            selected_option[
                "record"
            ]
        )

        selected_path = Path(
            record["path"]
        )

        try:

            # This ONLY reads the saved JSON.
            # No AI model is executed.
            selected_result = (
                load_processed_meeting(
                    selected_path
                )
            )

            st.session_state[
                "active_processed_meeting"
            ] = str(
                selected_path
            )

            st.session_state[
                "active_processed_result"
            ] = selected_result

            st.session_state[
                "active_view"
            ] = "processed"

            active_view = "processed"

        except Exception as error:

            st.error(
                f"Could not load saved meeting: "
                f"{error}"
            )

            active_view = "legacy"


    # --------------------------------------------------------
    # Selected meeting summary in left card
    # --------------------------------------------------------

    if active_view == "legacy":

        st.caption(
            "✓ Existing ES2004a verified meeting"
        )

        lm1, lm2 = st.columns(2)

        with lm1:

            st.metric(
                "Claims",
                meeting_metadata[
                    "total_claims"
                ]
            )

        with lm2:

            st.metric(
                "Verified",
                meeting_metadata[
                    "verified_claims"
                ]
            )


    elif active_view == "processed":

        normalized_selected = (
            normalize_processed_meeting_result(
                st.session_state[
                    "active_processed_result"
                ]
            )
        )

        selected_summary = (
            normalized_selected[
                "summary"
            ]
        )

        st.caption(
            "✓ Saved processed meeting"
        )

        pm1, pm2 = st.columns(2)

        with pm1:

            st.metric(
                "Claims",
                selected_summary[
                    "total_claims"
                ]
            )

        with pm2:

            st.metric(
                "Verified",
                selected_summary[
                    "verified_claims"
                ]
            )


    elif active_view == "live":

        live_normalized = (
            normalize_processed_meeting_result(
                st.session_state[
                    "active_live_result"
                ]
            )
        )

        live_summary = (
            live_normalized[
                "summary"
            ]
        )

        st.caption(
            "🟢 Newly processed meeting"
        )

        lm1, lm2 = st.columns(2)

        with lm1:

            st.metric(
                "Claims",
                live_summary[
                    "total_claims"
                ]
            )

        with lm2:

            st.metric(
                "Verified",
                live_summary[
                    "verified_claims"
                ]
            )


# ============================================================
# RIGHT: PROCESS NEW MEETING
# ============================================================

with process_col:

    st.markdown(
        "## 🎙️ Process New Meeting"
    )

    st.caption(
        "Upload a new meeting recording and run "
        "the complete verified MoM pipeline."
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
        ],
        key="new_meeting_uploader"
    )


    if uploaded_meeting is not None:

        st.success(
            f"Selected: "
            f"{uploaded_meeting.name}"
        )

        st.caption(
            f"Size: "
            f"{uploaded_meeting.size / (1024 * 1024):.2f} MB"
        )


        process_meeting = st.button(
            "▶️ Process Meeting",
            type="primary",
            use_container_width=True,
            key="process_new_meeting_button"
        )


        if process_meeting:

            try:

                # ------------------------------------------------
                # Save audio
                # ------------------------------------------------

                saved_path = (
                    save_uploaded_meeting(
                        uploaded_meeting
                    )
                )

                st.info(
                    "Starting AI meeting processing..."
                )


                # ------------------------------------------------
                # Load Qwen
                # ------------------------------------------------

                st.info(
                    "Loading Qwen generator..."
                )

                qwen_model, qwen_tokenizer = (
                    load_qwen_generator()
                )


                # ------------------------------------------------
                # Full validated pipeline
                # ------------------------------------------------

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
                # IMPORTANT:
                # Make this the active view.
                # ------------------------------------------------

                st.session_state[
                    "active_live_result"
                ] = pipeline_result

                st.session_state[
                    "active_live_result_path"
                ] = str(
                    result_path
                )

                st.session_state[
                    "active_view"
                ] = "live"


                # Keep compatibility with old state
                st.session_state[
                    "uploaded_meeting_result"
                ] = pipeline_result

                st.session_state[
                    "uploaded_meeting_result_path"
                ] = str(
                    result_path
                )


                st.success(
                    "✓ Meeting processing completed."
                )

                st.success(
                    "✓ Result saved and added to "
                    "Implemented Meetings."
                )


                # ------------------------------------------------
                # Rerun dashboard.
                #
                # This causes discover_processed_meetings()
                # to see the newly generated JSON.
                # ------------------------------------------------

                st.rerun()


            except Exception as error:

                st.error(
                    f"Meeting processing failed: "
                    f"{error}"
                )

                st.exception(error)


# ============================================================
# ACTIVE RESULT VIEW
# ============================================================

active_view = st.session_state.get(
    "active_view",
    "legacy"
)


# ------------------------------------------------------------
# LEGACY ES2004a
# ------------------------------------------------------------

if active_view == "legacy":

    st.divider()

    render_legacy_es2004a_dashboard()


# ------------------------------------------------------------
# SAVED PROCESSED MEETING
# ------------------------------------------------------------

elif active_view == "processed":

    st.divider()

    processed_result = (
        st.session_state.get(
            "active_processed_result"
        )
    )

    processed_path = (
        st.session_state.get(
            "active_processed_meeting"
        )
    )

    if (
        processed_result
        and processed_path
    ):

        render_processed_meeting(
            processed_result,
            Path(processed_path),
            view_mode="processed"
        )

    else:

        st.warning(
            "No processed meeting is currently selected."
        )


# ------------------------------------------------------------
# LIVE RESULT
# ------------------------------------------------------------

elif active_view == "live":

    st.divider()

    live_result = (
        st.session_state.get(
            "active_live_result"
        )
    )

    live_path = (
        st.session_state.get(
            "active_live_result_path"
        )
    )

    if (
        live_result
        and live_path
    ):

        render_processed_meeting(
            live_result,
            Path(live_path),
            view_mode="live"
        )

    else:

        st.warning(
            "No live meeting result is currently available."
        )


# ============================================================
# FIXED FOOTER EXPORT
# ============================================================

st.divider()

st.markdown(
    "## 📥 Export"
)

st.caption(
    "Download the verified meeting minutes for "
    "the meeting currently displayed above."
)


pdf_path, pdf_name, pdf_available = (
    get_active_pdf_info()
)


active_view = st.session_state.get(
    "active_view",
    "legacy"
)


if active_view == "legacy":

    if (
        pdf_available
        and pdf_path
        and pdf_path.exists()
    ):

        with open(
            pdf_path,
            "rb"
        ) as pdf_file:

            pdf_bytes = pdf_file.read()

        st.download_button(
            label=(
                "📄 Download Verified "
                "Meeting Minutes PDF"
            ),
            data=pdf_bytes,
            file_name=pdf_name,
            mime="application/pdf",
            use_container_width=True,
            key="footer_legacy_pdf"
        )

    else:

        st.info(
            "ES2004a verified PDF is not available."
        )


else:

    if (
        active_view == "live"
    ):

        current_result = (
            st.session_state.get(
                "active_live_result"
            )
        )

    else:

        current_result = (
            st.session_state.get(
                "active_processed_result"
            )
        )


    if current_result is not None:

        try:

            generated_pdf = (
                create_context_pdf(
                    current_result,
                    pdf_path
                )
            )

            with open(
                generated_pdf,
                "rb"
            ) as pdf_file:

                pdf_bytes = pdf_file.read()

            st.download_button(
                label=(
                    "📄 Download Verified "
                    "Meeting Minutes PDF"
                ),
                data=pdf_bytes,
                file_name=pdf_name,
                mime="application/pdf",
                use_container_width=True,
                key="footer_dynamic_pdf"
            )

        except Exception as error:

            st.error(
                f"Could not generate PDF: {error}"
            )

    else:

        st.info(
            "Select a processed meeting to "
            "enable PDF download."
        )


# ============================================================
# D10-7 FOOTER
# ============================================================

st.caption(
    "Verifiable Meeting Minutes • "
    "Evidence-grounded meeting intelligence"
)
