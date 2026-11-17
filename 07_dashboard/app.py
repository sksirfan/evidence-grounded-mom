
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


def _render_transcript_dependency_scenario(title, scenario):
    """Render transcript-explicit scenario edges separately from NLI-verified DAG edges."""
    if not isinstance(scenario, dict):
        return
    tasks = scenario.get("tasks", []) or []
    edges = scenario.get("dependencies", []) or []
    dag = scenario.get("dag", {}) or {}
    validation = dag.get("validation", {}) or {}
    st.markdown(f"##### {title}")
    st.caption(scenario.get("interpretation", ""))
    a, b, c = st.columns(3)
    a.metric("Tasks", len(tasks))
    b.metric("Explicit relationships", len(edges))
    bval = "Yes" if validation.get("cycle_detected") else "No"
    c.metric("Cycle detected", bval)
    if tasks:
        dot = ["digraph Scenario {", "rankdir=LR;", 'node [shape=box, style="rounded", fontsize=10];']
        def q(v):
            return '"' + str(v).replace('\\', '\\\\').replace('"', '\\"') + '"'
        for task in tasks:
            label = f"{task.get('task_id')}\\n{task.get('task', '')}"
            dot.append(f"{q(task.get('task_id'))} [label={q(label)}];")
        for edge in edges:
            dot.append(f"{q(edge.get('prerequisite_id'))} -> {q(edge.get('dependent_id'))};")
        dot.append("}")
        try:
            st.graphviz_chart("\n".join(dot), use_container_width=True)
        except Exception:
            st.code("\n".join(dot), language="dot")
        st.table([{"Task ID": t.get("task_id"), "Task": t.get("task"),
                   "Speaker": t.get("reported_by", "Not identified")} for t in tasks])
        with st.expander("Transcript quotes supporting these relationships", expanded=False):
            for edge in edges:
                st.markdown(f"**{edge.get('prerequisite_id')} → {edge.get('dependent_id')}**")
                st.write(edge.get("evidence_text", ""))
                st.caption(f"{edge.get('evidence_start')}s–{edge.get('evidence_end')}s | "
                           f"Speaker: {edge.get('evidence_speaker', 'Unknown')} | "
                           "Transcript quote match: exact normalized substring")
    if validation.get("cycle_detected"):
        st.warning("This transcript-explicit scenario contains a circular wait. It is a candidate view, not a Phase 1/NLI-verified DAG.")
    else:
        st.success("This transcript-explicit scenario is acyclic. It shows the proposed execution sequence, not an NLI-verified DAG.")


def render_phase2_dependency_analysis(phase2_result, meeting_id="meeting"):
    """Render Phase 2 task dependencies with compact graph and readable table."""
    import json as _json
    import textwrap as _textwrap

    st.markdown("### 🧩 Phase 2 — Task Dependency Analysis")

    if not isinstance(phase2_result, dict):
        st.info("Phase 2 analysis is not available in this saved result.")
        return

    status = str(phase2_result.get("status", "unknown"))
    st.caption(phase2_result.get("message", ""))

    scenarios = phase2_result.get("transcript_dependency_views", {}) or {}
    if scenarios:
        st.markdown("#### Transcript-grounded dependency scenarios")
        st.info("These scenario graphs are built from explicit statements and exact transcript quotes. They are shown separately from the NLI-verified DAG below; they do not change Phase 1 claim status or verified edges.")
        _render_transcript_dependency_scenario("1. Original dependency state — circular wait", scenarios.get("original_state", {}))
        _render_transcript_dependency_scenario("2. Resolved dependency state — revised execution plan", scenarios.get("resolved_state", {}))

    tasks = phase2_result.get("tasks", []) or []
    dependencies = phase2_result.get("dependencies", []) or []
    reviewed = phase2_result.get("candidate_dependencies", []) or []
    dag_data = phase2_result.get("dag", {}) or {}
    validation = dag_data.get("validation", {}) or {}
    ready = dag_data.get("ready_tasks", []) or []
    order = dag_data.get("topological_order", []) or []

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Verified action tasks", len(tasks))
    m2.metric("Verified dependencies", len(dependencies))
    m3.metric("Ready tasks", len(ready))
    m4.metric("Cycle detected", "Yes" if validation.get("cycle_detected") else "No")

    if status in ("skipped_no_model", "skipped_no_transcript", "extraction_error"):
        st.warning(
            f"Phase 2 status: {status.replace('_', ' ').title()}. "
            "Phase 1 verified minutes remain available."
        )
    elif status == "completed_no_verified_dependencies":
        st.info(
            "Phase 2 completed. No dependency passed both transcript-evidence "
            "validation and relation-specific NLI."
        )
    elif status == "completed":
        st.success("Phase 2 completed with verified task dependencies.")

    if tasks:
        st.markdown("#### Dependency Graph")

        if not dependencies:
            # For an edgeless DAG, show independent nodes side-by-side.
            # This is clearer than a Graphviz layout that stretches isolated nodes.
            columns_per_row = min(3, len(tasks))
            for row_start in range(0, len(tasks), columns_per_row):
                row_tasks = tasks[row_start:row_start + columns_per_row]
                cols = st.columns(columns_per_row)
                for col, task in zip(cols, row_tasks):
                    with col:
                        with st.container(border=True):
                            st.markdown(f"**{task.get('task_id', 'Task')}**")
                            st.write(task.get("task", "Task description unavailable"))
                            st.caption("Independent · Ready")
            st.caption("No arrows are shown because no prerequisite relationships passed verification.")
        else:
            # For connected DAGs, use Graphviz labels with real DOT line breaks.
            dot_lines = [
                "digraph TaskDependencies {",
                "rankdir=LR;",
                'graph [nodesep=0.35, ranksep=0.55, pad=0.15, bgcolor="transparent"];',
                'node [shape=box, style="rounded", fontsize=10, margin="0.12,0.08", width=2.5];',
                'edge [arrowsize=0.7];',
            ]

            def _dot_quote(value):
                return '"' + str(value).replace('\\', '\\\\').replace('"', '\\"') + '"'

            for task in tasks:
                task_id = str(task.get("task_id", ""))
                description = " ".join(str(task.get("task", "")).split())
                wrapped = _textwrap.wrap(description, width=28) or ["Task description unavailable"]
                if len(wrapped) > 5:
                    wrapped = wrapped[:5]
                    wrapped[-1] = wrapped[-1].rstrip(" .") + "…"
                # Graphviz interprets literal backslash-n as a line break.
                label_text = task_id + "\\n" + "\\n".join(wrapped)
                dot_lines.append(f"{_dot_quote(task_id)} [label={_dot_quote(label_text)}];")

            for edge in dependencies:
                dot_lines.append(
                    f"{_dot_quote(edge.get('prerequisite_id'))} -> "
                    f"{_dot_quote(edge.get('dependent_id'))};"
                )
            dot_lines.append("}")
            try:
                st.graphviz_chart("\n".join(dot_lines), use_container_width=True)
            except Exception as exc:
                st.warning(f"Graph rendering unavailable: {exc}")
                st.code("\n".join(dot_lines), language="dot")

        # Restore the requested table layout. Use a static Streamlit table and
        # wrapping CSS so long task text is not clipped in a narrow dataframe cell.
        st.markdown("#### Action Tasks")
        st.markdown(
            """<style>
            div[data-testid="stTable"] table {width:100%; table-layout:fixed;}
            div[data-testid="stTable"] th, div[data-testid="stTable"] td {
                white-space:normal !important;
                overflow-wrap:anywhere !important;
                word-break:normal !important;
                vertical-align:top !important;
            }
            div[data-testid="stTable"] th:nth-child(2),
            div[data-testid="stTable"] td:nth-child(2) {min-width:360px; width:48%;}
            </style>""",
            unsafe_allow_html=True,
        )
        task_rows = [
            {
                "Task ID": task.get("task_id"),
                "Task": task.get("task"),
                "Owner": task.get("owner") or "Unassigned",
                "Reported by": task.get("reported_by") or "Unknown",
                "Source claim": task.get("source_claim_number"),
                "Claim status": task.get("status"),
            }
            for task in tasks
        ]
        st.table(task_rows)

        with st.expander("Task source evidence", expanded=False):
            for task in tasks:
                evidence = task.get("source_evidence", {}) or {}
                st.markdown(f"**{task.get('task_id')} — {task.get('task')}**")
                st.caption(
                    f"Reported by: {task.get('reported_by') or 'Unknown'} | "
                    f"Claim time: {format_timestamp(task.get('source_start'))} – "
                    f"{format_timestamp(task.get('source_end'))}"
                )
                if evidence.get("text"):
                    st.write(evidence.get("text"))
                else:
                    st.caption("No source evidence text was stored for this task.")

    if dependencies:
        st.markdown("#### Verified Dependency Evidence")
        for edge in dependencies:
            with st.container(border=True):
                st.write(f"**{edge.get('prerequisite_id')} → {edge.get('dependent_id')}**")
                st.write(edge.get("evidence_text") or "No quote recorded.")
                st.caption(
                    f"Transcript time: {format_timestamp(edge.get('evidence_start'))} – "
                    f"{format_timestamp(edge.get('evidence_end'))} | "
                    f"Speaker: {edge.get('evidence_speaker') or 'Unknown'} | "
                    f"NLI: {edge.get('nli_label') or 'N/A'} "
                    f"(entailment={edge.get('nli_entailment') if edge.get('nli_entailment') is not None else 'N/A'})"
                )
    elif status == "completed_no_verified_dependencies":
        st.info("No verified prerequisite edges are present. Tasks are independent; no ordering is inferred.")

    if validation.get("is_dag"):
        st.markdown("#### Dependency-Respecting Execution Order")
        if dependencies and order:
            st.write(" → ".join(order))
        elif tasks:
            st.caption("No required execution sequence was established; the tasks are independent in the verified DAG.")
        else:
            st.caption("No tasks to order.")
        st.markdown("#### Ready Tasks")
        st.write(" · ".join(str(task_id) for task_id in ready) if ready else "None")
    else:
        st.error("The dependency graph contains a cycle. Execution order is withheld until the cycle is resolved.")

    if tasks:
        st.markdown("#### Downstream Change-Impact Analysis")
        task_options = {
            f"{task.get('task_id')} — {' '.join(str(task.get('task', '')).split())[:90]}": task.get("task_id")
            for task in tasks
        }
        selected_label = st.selectbox(
            "Select a task to inspect downstream impact",
            list(task_options.keys()),
            key=f"phase2_impact_{meeting_id}"
        )
        selected_id = task_options[selected_label]
        try:
            graph = phase2_dag.build_dependency_dag(tasks, dependencies)
            impact = phase2_dag.get_downstream_impact(graph, selected_id)
            impacted = impact.get("impacted_tasks", [])
            if impacted:
                st.write(f"Changing or delaying **{selected_id}** may affect:")
                for item in impacted:
                    st.write(f"- **{item.get('task_id')}** — {item.get('task')}")
            else:
                st.caption("No downstream tasks are linked to this task in the verified DAG.")
        except Exception as exc:
            st.warning(f"Impact analysis unavailable: {exc}")

    if reviewed:
        st.markdown("#### Unverified / Rejected Dependency Candidates")
        for index, candidate in enumerate(reviewed, start=1):
            source_id = candidate.get("prerequisite_id", "?")
            target_id = candidate.get("dependent_id", "?")
            review_status = candidate.get("verification_status", "NOT_EVALUATED")
            reason = candidate.get("verification_reason", "")
            with st.expander(
                f"{index}. {source_id} → {target_id} | {review_status} | {reason}",
                expanded=False
            ):
                evidence_text = candidate.get("evidence_text") or candidate.get("evidence_quote")
                if evidence_text:
                    st.write(evidence_text)
                else:
                    st.caption("No transcript evidence quote was retained.")
                if candidate.get("evidence_start") is not None:
                    st.caption(
                        f"Transcript time: {format_timestamp(candidate.get('evidence_start'))} – "
                        f"{format_timestamp(candidate.get('evidence_end'))}"
                    )
                if reason:
                    st.caption(f"Verification reason: {reason}")

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

    # --------------------------------------------------------
    # Phase 2 — Evidence-grounded task dependency DAG
    # Render only; this does not run inference.
    # --------------------------------------------------------
    raw_result = normalized.get("raw", {})
    phase2_result = (
        raw_result.get("phase2_dependency_analysis")
        if isinstance(raw_result, dict)
        else None
    )
    render_phase2_dependency_analysis(
        phase2_result,
        meeting_id=meeting_id
    )


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

# Colab compatibility: this application does not use Transformers vision
# features. Avoid importing an incompatible optional torchvision installation.
import transformers.utils.import_utils as _transformers_import_utils
_transformers_import_utils._torchvision_available = False

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
import dependency_extraction as phase2_dependency_extraction
import task_dependency_dag as phase2_dag


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


def verify_phase2_dependency_relation(evidence_text, proposition):
    """Reuse Phase 1's cached DeBERTa NLI model for dependency-edge verification."""
    resources = getattr(mp, "_NLI_RESOURCES", None)
    if resources is None:
        resources = mp.load_nli_model()
        mp._NLI_RESOURCES = resources
    tokenizer, model, device = resources
    return mp.run_nli(
        evidence_text,
        proposition,
        tokenizer,
        model,
        device
    )


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
    # Save the exact WhisperX + Pyannote speaker-aligned transcript
    # for later Phase 2 inspection/reuse. This is an intermediate
    # artifact; it does not modify the Phase 1 utterances.
    # --------------------------------------------------------

    transcript_output_path = None
    transcript_save_error = None

    try:
        transcript_dir = PROJECT_ROOT / "outputs" / "transcripts"
        transcript_dir.mkdir(parents=True, exist_ok=True)

        transcript_output_path = (
            transcript_dir
            / f"{audio_path.stem}_speaker_transcript.json"
        )

        transcript_payload = {
            "meeting_id": audio_path.stem,
            "source_audio": audio_path.name,
            "utterance_count": len(utterances),
            "utterances": utterances,
        }

        with open(transcript_output_path, "w", encoding="utf-8") as transcript_file:
            json.dump(
                transcript_payload,
                transcript_file,
                indent=2,
                ensure_ascii=False
            )

        transcript_output_path = str(transcript_output_path)

    except Exception as transcript_error:
        # Transcript archival must never interrupt the validated Phase 1 flow.
        transcript_save_error = str(transcript_error)
        transcript_output_path = None

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
    # 8B. Phase 2 — Task dependency extraction and verification
    # --------------------------------------------------------
    # Failure is isolated: Phase 1 verified claims are never discarded.
    try:
        phase2_result = phase2_dependency_extraction.extract_action_dependency_dag(
            claims=verification_results,
            utterances=utterances,
            model=generator_model,
            tokenizer=generator_tokenizer,
            nli_verifier=verify_phase2_dependency_relation,
            entailment_threshold=0.80
        )
    except Exception as phase2_error:
        phase2_result = {
            "status": "extraction_error",
            "tasks": [],
            "dependencies": [],
            "candidate_dependencies": [],
            "dag": {"nodes": [], "edges": [], "validation": {"is_dag": True, "cycle_detected": False, "cycles": []},
                    "topological_order": [], "ready_tasks": []},
            "message": f"Phase 2 failed safely; Phase 1 results are retained: {phase2_error}"
        }

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
        ),

        "phase2_dependency_analysis": (
            phase2_result
        ),

        "transcript_output_path": transcript_output_path,
        "transcript_save_error": transcript_save_error
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

    # --------------------------------------------------------
    # Phase 2 — Evidence-Grounded Task Dependency Analysis
    # --------------------------------------------------------
    phase2 = normalized.get("phase2_dependency_analysis") or result.get(
        "phase2_dependency_analysis", {}
    )
    if not isinstance(phase2, dict):
        phase2 = {}

    views = phase2.get("transcript_dependency_views", {})
    if not isinstance(views, dict):
        views = {}

    story.append(PageBreak())
    story.append(Paragraph(
        "Phase 2 – Evidence-Grounded Task Dependency Analysis",
        heading_style
    ))
    story.append(Paragraph(
        "This section records task dependencies explicitly supported by the meeting transcript. "
        "Transcript-explicit candidate edges are shown separately from edges that pass relation-specific "
        "NLI verification.", body_style
    ))
    story.append(Spacer(1, 6))

    phase2_tasks = phase2.get("tasks", []) or []
    phase2_edges = phase2.get("dependencies", []) or []
    phase2_candidates = phase2.get("candidate_dependencies", []) or []
    phase2_dag = phase2.get("dag", {}) or {}
    phase2_validation = phase2_dag.get("validation", {}) or {}

    story.append(Paragraph("Verified Phase 2 DAG result", heading_style))
    story.append(Paragraph(
        f"<b>Status:</b> {escape(str(phase2.get('status', 'not available')))}<br/>"
        f"<b>Tasks:</b> {len(phase2_tasks)} &nbsp;&nbsp; "
        f"<b>NLI-verified dependency edges:</b> {len(phase2_edges)} &nbsp;&nbsp; "
        f"<b>Candidate edges:</b> {len(phase2_candidates)}<br/>"
        f"<b>Cycle detected in verified DAG:</b> "
        f"{'Yes' if phase2_validation.get('cycle_detected') else 'No'}<br/>"
        f"<b>NLI entailment threshold:</b> {phase2.get('nli_entailment_threshold', 'not recorded')}",
        body_style
    ))

    if phase2_tasks:
        task_rows = [["Task ID", "Task", "Owner", "Status"]]
        for task in phase2_tasks:
            task_rows.append([
                str(task.get("task_id", "")),
                Paragraph(escape(str(task.get("task", ""))), body_style),
                str(task.get("owner") or "Not explicitly assigned"),
                str(task.get("status", ""))
            ])
        from reportlab.platypus import Table, TableStyle
        from reportlab.lib import colors
        task_table = Table(task_rows, colWidths=[48, 220, 100, 105], repeatRows=1)
        task_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EDF5")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#172B4D")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B8C2CC")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(task_table)
        story.append(Spacer(1, 8))

    if phase2_edges or phase2_candidates:
        story.append(Paragraph("Dependency verification details", heading_style))
        edge_rows = [["Type", "Dependency", "Transcript evidence", "Verification"]]
        for edge_type, edge_list in (("NLI verified", phase2_edges), ("Candidate", phase2_candidates)):
            for edge in edge_list:
                source = str(edge.get("prerequisite_id", "?"))
                target = str(edge.get("dependent_id", "?"))
                evidence = str(edge.get("evidence_text", ""))
                speaker = str(edge.get("evidence_speaker", "Unknown"))
                start = edge.get("evidence_start")
                end = edge.get("evidence_end")
                time_text = ""
                if start is not None:
                    time_text = f" ({format_timestamp(start)}–{format_timestamp(end)})"
                verification = str(edge.get("verification_status", edge_type))
                if edge.get("nli_label") is not None:
                    verification += f"; NLI={edge.get('nli_label')}; entailment={float(edge.get('nli_entailment') or 0):.3f}"
                edge_rows.append([
                    edge_type,
                    f"{source} → {target}",
                    Paragraph(escape(evidence) + f"<br/><font size='7'>Speaker: {escape(speaker)}{escape(time_text)}</font>", body_style),
                    Paragraph(escape(verification), body_style)
                ])
        edge_table = Table(edge_rows, colWidths=[65, 70, 230, 108], repeatRows=1)
        edge_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EDF5")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B8C2CC")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(edge_table)
        story.append(Spacer(1, 10))

    # A compact vector diagram: boxes and arrowheads are drawn directly in the PDF.
    from reportlab.platypus import Flowable
    from reportlab.lib import colors
    import math

    class DependencyDiagram(Flowable):
        def __init__(self, tasks, edges, cyclic=False, width=500):
            super().__init__()
            self.tasks = tasks
            self.edges = edges
            self.cyclic = cyclic
            self.width = width
            self.height = 170 if cyclic else 205

        def wrap(self, availWidth, availHeight):
            self.width = min(self.width, availWidth)
            return self.width, self.height

        def draw(self):
            c = self.canv
            by_id = {str(t.get("task_id")): t for t in self.tasks}
            ids = list(by_id)
            if self.cyclic and len(ids) >= 3:
                w, h = 138, 42
                centers = {
                    ids[0]: (self.width / 2, self.height - 34),
                    ids[1]: (82, 48),
                    ids[2]: (self.width - 82, 48),
                }
                for extra in ids[3:]:
                    centers[extra] = (self.width / 2, 48)
            else:
                w, h = min(112, (self.width - 30) / max(1, len(ids))), 48
                gap = (self.width - len(ids) * w) / (len(ids) + 1)
                centers = {tid: (gap + i * (w + gap) + w / 2, self.height / 2)
                           for i, tid in enumerate(ids)}

            c.setFont("Helvetica", 7)
            for tid, (cx, cy) in centers.items():
                task = by_id[tid]
                x, y = cx - w / 2, cy - h / 2
                c.setFillColor(colors.HexColor("#EEF3F8"))
                c.setStrokeColor(colors.HexColor("#52677D"))
                c.roundRect(x, y, w, h, 5, fill=1, stroke=1)
                c.setFillColor(colors.HexColor("#172B4D"))
                c.setFont("Helvetica-Bold", 8)
                c.drawCentredString(cx, y + h - 12, tid)
                import textwrap
                label = str(task.get("task", ""))
                wrap_width = max(14, int(w / 4.0))
                label_lines = textwrap.wrap(label, width=wrap_width)[:2]
                if label_lines and len(textwrap.wrap(label, width=wrap_width)) > 2:
                    label_lines[-1] = label_lines[-1][:max(8, wrap_width - 3)] + "..."
                c.setFont("Helvetica", 6.2)
                if len(label_lines) == 1:
                    c.drawCentredString(cx, y + 12, label_lines[0])
                else:
                    c.drawCentredString(cx, y + 15, label_lines[0])
                    c.drawCentredString(cx, y + 7, label_lines[1])

            c.setStrokeColor(colors.HexColor("#245B8F"))
            c.setFillColor(colors.HexColor("#245B8F"))
            for edge in self.edges:
                a = str(edge.get("prerequisite_id", ""))
                b = str(edge.get("dependent_id", ""))
                if a not in centers or b not in centers:
                    continue
                x1, y1 = centers[a]
                x2, y2 = centers[b]
                dx, dy = x2 - x1, y2 - y1
                dist = max(1, math.hypot(dx, dy))
                ux, uy = dx / dist, dy / dist
                sx, sy = x1 + ux * (w * .48), y1 + uy * (h * .48)
                ex, ey = x2 - ux * (w * .48), y2 - uy * (h * .48)
                c.setLineWidth(1.2)
                c.line(sx, sy, ex, ey)
                angle = math.atan2(ey - sy, ex - sx)
                size = 5
                p = c.beginPath()
                p.moveTo(ex, ey)
                p.lineTo(ex - size * math.cos(angle - .45), ey - size * math.sin(angle - .45))
                p.lineTo(ex - size * math.cos(angle + .45), ey - size * math.sin(angle + .45))
                p.close()
                c.drawPath(p, fill=1, stroke=0)

    def append_dependency_view(view_key, heading):
        view = views.get(view_key, {})
        if not isinstance(view, dict) or not view:
            return
        tasks = view.get("tasks", []) or []
        edges = view.get("dependencies", []) or []
        dag = view.get("dag", {}) or {}
        validation = dag.get("validation", {}) or {}
        cycle = bool(validation.get("cycle_detected"))
        story.append(Paragraph(heading, heading_style))
        story.append(Paragraph(escape(str(view.get("interpretation", ""))), body_style))
        story.append(Paragraph(
            f"<b>Tasks:</b> {len(tasks)} &nbsp;&nbsp; <b>Edges:</b> {len(edges)} &nbsp;&nbsp; "
            f"<b>Cycle detected:</b> {'Yes' if cycle else 'No'} &nbsp;&nbsp; "
            f"<b>Classification:</b> {escape(str(view.get('status', 'candidate')))}",
            body_style
        ))
        if tasks and edges:
            story.append(DependencyDiagram(tasks, edges, cyclic=cycle))
        if edges:
            for edge in edges:
                story.append(Paragraph(
                    f"<b>{escape(str(edge.get('prerequisite_id', '?')))} → "
                    f"{escape(str(edge.get('dependent_id', '?')))}</b>: "
                    f"{escape(str(edge.get('evidence_text', '')))}",
                    body_style
                ))
                start = edge.get("evidence_start")
                end = edge.get("evidence_end")
                if start is not None:
                    story.append(Paragraph(
                        f"<font size='8'>Speaker: {escape(str(edge.get('evidence_speaker', 'Unknown')))}; "
                        f"time: {format_timestamp(start)}–{format_timestamp(end)}; "
                        f"status: {escape(str(edge.get('verification_status', 'candidate')))}</font>",
                        body_style
                    ))
        story.append(Spacer(1, 8))

    append_dependency_view("original_state", "Original dependency state – circular wait")
    append_dependency_view("resolved_state", "Resolved dependency state – revised execution plan")

    if not views:
        story.append(Paragraph(
            "No transcript dependency views were saved for this meeting.", body_style
        ))

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


    # After a new meeting is processed, the keyed selectbox can retain
    # its previous index from Streamlit session state. Apply the newly
    # calculated live-result index BEFORE creating the widget.
    if st.session_state.pop("_force_latest_live_meeting_selection", False):
        st.session_state["implemented_meeting_selector"] = default_index

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

                # Save Phase 2 independently so its status/evidence can be
                # inspected without changing or replacing Phase 1 outputs.
                phase2_dir = PROJECT_ROOT / "outputs" / "phase2_dag"
                phase2_dir.mkdir(parents=True, exist_ok=True)
                phase2_result_path = (
                    phase2_dir / f"{saved_path.stem}_phase2_dependency_dag.json"
                )
                phase2_payload = pipeline_result.get(
                    "phase2_dependency_analysis", {}
                )
                with open(phase2_result_path, "w", encoding="utf-8") as phase2_file:
                    json.dump(
                        phase2_payload,
                        phase2_file,
                        indent=2,
                        ensure_ascii=False
                    )
                pipeline_result["phase2_output_path"] = str(phase2_result_path)

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

                # The meeting selector is keyed and may retain an older
                # selection. Force it to the live option on the next rerun.
                st.session_state[
                    "_force_latest_live_meeting_selection"
                ] = True


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
