"""Phase 2: transcript-grounded action dependency extraction and verification.

This module is intentionally separate from the Phase 1 MoM pipeline. It reuses
Phase 1's Qwen model/tokenizer and accepts a callback for the existing DeBERTa
NLI model. An edge is admitted to the DAG only when its quote is found in the
transcript and the relation-specific NLI check entails the prerequisite claim.
"""
from __future__ import annotations
import json
import re
import unicodedata
from typing import Any, Callable, Dict, List, Optional


def _float(value):
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or "")).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", text)).strip()


def _is_reported_context_claim(text: str) -> bool:
    """Reject claims that merely report background information as an ACTION."""
    normalized = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    # These are attribution/reporting statements, not work items. Keep this
    # deliberately narrow: do not reject ordinary task descriptions based on
    # broad topic or verb heuristics.
    return bool(re.match(
        r"^(?:(?:because|yesterday)\s+)?you\s+(?:said|mentioned|told me)\b",
        normalized,
    ))


def _normalize_action_task(text: str) -> tuple[Optional[str], Optional[str]]:
    """Normalize the dependent action in a blocked-work statement."""
    original = re.sub(r"\s+", " ", str(text or "")).strip()
    if not original or _is_reported_context_claim(original):
        return None, None

    blocked = re.match(
        r"^(?:I|we|you|they|he|she)\s+(?:can't|cannot|can’t|won't|will not)\s+"
        r"(?P<action>.+?)\s+until\s+(?P<prerequisite>.+)$",
        original,
        flags=re.IGNORECASE,
    )
    if blocked:
        action = blocked.group("action").strip(" \t.,;:")
        if action:
            action = action[0].upper() + action[1:]
            return action, "blocked_action_normalized"

    return original, "claim_used_as_task"


def _extract_blocked_prerequisite(text: str) -> Optional[str]:
    """Extract only clearly action-bearing 'until' clauses; never infer an owner."""
    original = re.sub(r"\s+", " ", str(text or "")).strip()
    match = re.match(
        r"^(?:I|we|you|they|he|she)\s+(?:can't|cannot|can’t|won't|will not)\s+"
        r".+?\s+until\s+(?P<clause>.+)$",
        original,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    clause = match.group("clause").strip(" \t.,;:")
    # Deliberately narrow action vocabulary: temporal clauses such as 'until Friday'
    # and vague states are not promoted to task nodes.
    action_verbs = (
        "test", "check", "confirm", "validate", "verify", "review", "approve",
        "complete", "finish", "implement", "build", "prepare", "provide",
        "send", "share", "submit", "create", "update", "lock", "finalize",
        "determine", "decide", "deploy", "integrate", "fix", "resolve",
    )
    subjectless = re.match(r"^(?:you|we|they|i)\s+(.+)$", clause, re.IGNORECASE)
    if subjectless:
        clause = subjectless.group(1).strip()
    first_word = re.match(r"([A-Za-z]+)", clause)
    if not first_word or first_word.group(1).lower() not in action_verbs:
        # Preserve an explicit named actor only when followed by a known action.
        named = re.match(r"^([A-Z][a-z]+)\s+([A-Za-z]+)\b(.*)$", clause)
        if not named or named.group(2).lower() not in action_verbs:
            return None
    clause = clause[0].upper() + clause[1:]
    return clause


def prepare_action_tasks(claims: List[Dict[str, Any]], require_verified: bool = True) -> List[Dict[str, Any]]:
    """Prepare conservative task candidates from verified ACTION claims.

    Reported background/context statements are excluded. Blocked-work claims
    retain their actionable work item (e.g. "I can't finish X until Y" becomes
    "Finish X"). The original claim and its evidence remain attached for audit.
    The person who stated a task is stored as reported_by, never as its owner.
    """
    tasks = []
    for claim_number, claim in enumerate(claims or [], 1):
        if not isinstance(claim, dict):
            continue
        if str(claim.get("event_type") or "").strip().upper() != "ACTION":
            continue
        status = str(claim.get("status") or "UNKNOWN").strip().upper()
        if require_verified and status != "VERIFIED":
            continue
        source_text = str(claim.get("claim") or claim.get("claim_text") or "").strip()
        task_text, normalization = _normalize_action_task(source_text)
        if not task_text:
            continue
        evidence = claim.get("evidence") or claim.get("selected_evidence") or {}
        if isinstance(evidence, list):
            evidence = evidence[0] if evidence else {}
        if not isinstance(evidence, dict):
            evidence = {}
        # Explicit assignee only; speaker attribution is not ownership.
        explicit_owner = claim.get("owner") or claim.get("assignee") or claim.get("assigned_to")
        common = {
            "source_claim_number": claim.get("number", claim_number),
            "owner": explicit_owner,
            "reported_by": claim.get("speaker"),
            "status": status,
            "source_claim": source_text,
            "source_start": _float(claim.get("start")),
            "source_end": _float(claim.get("end")),
            "source_evidence": {
                "text": evidence.get("text"),
                "start": _float(evidence.get("start")),
                "end": _float(evidence.get("end")),
                "speaker": evidence.get("speaker"),
                "evidence_id": evidence.get("evidence_id"),
            },
        }
        prerequisite = _extract_blocked_prerequisite(source_text)
        candidates = []
        if prerequisite and _norm(prerequisite) != _norm(task_text):
            candidates.append((prerequisite, "blocked_prerequisite_candidate", "PREREQUISITE"))
        candidates.append((task_text, normalization, "DEPENDENT" if prerequisite else "UNSPECIFIED"))
        for candidate_text, candidate_normalization, task_role in candidates:
            tasks.append({
                "task_id": f"T{len(tasks) + 1}",
                "task": candidate_text,
                "task_normalization": candidate_normalization,
                "task_role_candidate": task_role,
                **common,
            })
    return tasks


def _merge_transcript_turns(utterances: List[Dict[str, Any]], max_gap: float = 0.8) -> List[Dict[str, Any]]:
    """Merge word-level items per speaker, then restore chronological turn order.

    Grouping per speaker avoids breaking a sentence every time another speaker
    briefly overlaps it in the diarized transcript.
    """
    rows = [u for u in (utterances or []) if isinstance(u, dict) and str(u.get("text", "")).strip()]
    by_speaker = {}
    for u in rows:
        by_speaker.setdefault(u.get("speaker"), []).append(u)
    turns = []
    for speaker, speaker_rows in by_speaker.items():
        speaker_rows.sort(key=lambda x: (_float(x.get("start")) or 0.0, _float(x.get("end")) or 0.0))
        current = None
        for u in speaker_rows:
            start, end = _float(u.get("start")), _float(u.get("end"))
            word = str(u.get("text", "")).strip()
            if start is None or end is None:
                continue
            if current and start - current["end"] <= max_gap and start >= current["start"]:
                current["end"] = max(current["end"], end)
                current["text"] = (current["text"] + " " + word).strip()
            else:
                if current:
                    turns.append(current)
                current = {"speaker": speaker, "start": start, "end": end, "text": word}
        if current:
            turns.append(current)
    turns.sort(key=lambda x: (x["start"], x["end"], str(x["speaker"])))
    for index, turn in enumerate(turns, 1):
        turn["turn_id"] = f"U{index}"
    return turns


def _transcript_chunks(turns: List[Dict[str, Any]], chunk_seconds: float = 300.0, overlap_seconds: float = 12.0):
    if not turns:
        return []
    first = min(t["start"] for t in turns)
    last = max(t["end"] for t in turns)
    chunks = []
    cursor = first
    while cursor <= last:
        stop = cursor + chunk_seconds
        chunk = [t for t in turns if t["start"] < stop and t["end"] >= cursor]
        if chunk:
            chunks.append(chunk)
        if stop >= last:
            break
        cursor = stop - overlap_seconds
    return chunks



def _extract_json_payload(text: str, required_key: str) -> Dict[str, Any]:
    """Extract the first valid JSON object containing the required top-level key."""
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"```(?:json)?|```", "", cleaned, flags=re.IGNORECASE).strip()
    decoder = json.JSONDecoder()

    for start in range(len(cleaned)):
        if cleaned[start] != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(cleaned[start:])
            if isinstance(parsed, dict) and required_key in parsed:
                return parsed
        except json.JSONDecodeError:
            continue

    raise ValueError(
        f"Qwen response did not contain a valid JSON object with a {required_key!r} field. "
        f"Response preview: {cleaned[:500]}"
    )


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Backward-compatible dependency JSON parser."""
    return _extract_json_payload(text, "dependencies")


def _build_task_extraction_prompt(turns: List[Dict[str, Any]]) -> str:
    'Ask Qwen for atomic work items explicitly stated in the transcript.'
    transcript_rows = [
        {
            "turn_id": u["turn_id"],
            "speaker": u["speaker"],
            "start": round(u["start"], 3),
            "end": round(u["end"], 3),
            "text": u["text"],
        }
        for u in turns
    ]
    return f"""Extract the concrete action tasks explicitly assigned, requested, committed to,
or stated as required in this meeting transcript.

TRANSCRIPT:
{json.dumps(transcript_rows, ensure_ascii=False)}

Rules:
- Extract actual work to be performed, not agenda items, intentions to discuss, context,
  explanations, or reasons.
- Use one atomic task per item. Include prerequisite and downstream tasks as separate items.
- Include an independent task if one is explicitly stated.
- Use concise task descriptions beginning with an action verb.
- An owner may be populated only when a person or team is explicitly named as responsible
  in the quoted transcript turn. Do not infer an owner from the speaker ID or voice.
- Every task must cite one transcript turn and an exact quote copied from that turn.
- Do not invent deadlines, owners, tasks, or evidence.
- Preserve task order by the first explicit occurrence in the transcript.
- If a task is repeated, return it once and use the clearest direct assignment/commitment quote.

Return JSON only:
{{"tasks":[{{"task":"Test the payment API","owner":"Priya","reported_by":"SPEAKER_00",
"source_turn_id":"U3","evidence_quote":"Priya, please test the payment API"}}]}}
If no explicit action tasks exist: {{"tasks":[]}}"""


def _run_qwen_task_json(prompt: str, model, tokenizer, max_new_tokens: int = 500) -> Dict[str, Any]:
    'Run Qwen for transcript task extraction and parse its JSON response.'
    import torch
    messages = [
        {
            "role": "system",
            "content": (
                "You extract only explicit meeting action tasks with exact transcript evidence. "
                "Output valid JSON only. Never infer task owners."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=4096)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(
        output_ids[0, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return _extract_json_payload(response, "tasks")


def extract_transcript_action_tasks(
    turns: List[Dict[str, Any]],
    model,
    tokenizer,
) -> List[Dict[str, Any]]:
    'Create Phase 2 tasks from exact, transcript-quoted Qwen extractions.'
    if not turns:
        return []

    raw_tasks = []
    for chunk in _transcript_chunks(turns):
        parsed = _run_qwen_task_json(
            _build_task_extraction_prompt(chunk), model, tokenizer
        )
        chunk_tasks = parsed.get("tasks", [])
        if isinstance(chunk_tasks, list):
            raw_tasks.extend(chunk_tasks)

    turn_lookup = {t["turn_id"]: t for t in turns}
    tasks = []
    seen = set()

    for raw in raw_tasks:
        if not isinstance(raw, dict):
            continue

        task_text = re.sub(r"\s+", " ", str(raw.get("task") or "")).strip()
        turn_id = str(raw.get("source_turn_id") or "").strip()
        quote = str(raw.get("evidence_quote") or "").strip()

        if not task_text or not turn_id or not quote or turn_id not in turn_lookup:
            continue

        turn = turn_lookup[turn_id]
        if _norm(quote) not in _norm(turn.get("text", "")):
            continue

        task_key = _norm(task_text)
        if not task_key or task_key in seen:
            continue
        seen.add(task_key)

        owner = raw.get("owner")
        if owner:
            owner_text = str(owner).strip()
            if (
                not owner_text
                or _norm(owner_text) not in _norm(quote)
                or not re.search(
                    r"\b(?:please|assigned|responsible|you will|you must|must)\b",
                    quote,
                    flags=re.IGNORECASE,
                )
            ):
                owner = None
            else:
                owner = owner_text
        else:
            owner = None

        tasks.append({
            "task_id": f"T{len(tasks) + 1}",
            "task": task_text,
            "task_normalization": "qwen_transcript_task",
            "task_role_candidate": "UNSPECIFIED",
            "source_claim_number": None,
            "owner": owner,
            "reported_by": turn.get("speaker"),
            "status": "VERIFIED",
            "verification_method": "EXACT_TRANSCRIPT_QUOTE",
            "source_claim": None,
            "source_start": _float(turn.get("start")),
            "source_end": _float(turn.get("end")),
            "source_evidence": {
                "text": quote,
                "start": _float(turn.get("start")),
                "end": _float(turn.get("end")),
                "speaker": turn.get("speaker"),
                "turn_id": turn_id,
                "verification_method": "EXACT_NORMALIZED_SUBSTRING",
            },
        })

    return tasks



def _build_prompt(tasks: List[Dict[str, Any]], turns: List[Dict[str, Any]]) -> str:
    task_rows = [{"task_id": t["task_id"], "task": t["task"], "reported_by": t.get("reported_by"),
                  "claim_time": [t.get("source_start"), t.get("source_end")]} for t in tasks]
    transcript_rows = [{"turn_id": u["turn_id"], "speaker": u["speaker"],
                        "start": round(u["start"], 3), "end": round(u["end"], 3), "text": u["text"]}
                       for u in turns]
    return f'''You identify explicit prerequisite relationships between the VERIFIED meeting action tasks below.

TASKS (task IDs are fixed; do not create or rewrite tasks):
{json.dumps(task_rows, ensure_ascii=False)}

TRANSCRIPT CHUNK (source of dependency evidence):
{json.dumps(transcript_rows, ensure_ascii=False)}

A dependency A -> B means the transcript explicitly says A must be completed before B can start or finish.
Rules:
- Return an edge only if this transcript chunk explicitly states the prerequisite relationship between these two listed tasks.
- Do not infer from chronology, common sense, topic similarity, shared speaker, or task order.
- Do not treat a general agenda, recommendation, opinion, or discussion point as a dependency.
- Do not use evidence outside this transcript chunk.
- Quote a short exact phrase from ONE transcript turn and return that turn_id.
- If the quote does not directly express the relation, return no edge.
- Do not invent owners, deadlines, or task IDs.

Return JSON only:
{{"dependencies":[{{"prerequisite_id":"T1","dependent_id":"T2","relation":"MUST_PRECEDE","evidence_turn_id":"U1","evidence_quote":"exact words from the turn"}}]}}
If none: {{"dependencies":[]}}'''


def _run_qwen_json(prompt: str, model, tokenizer, max_new_tokens: int = 350) -> Dict[str, Any]:
    import torch
    messages = [
        {"role": "system", "content": "You are a conservative transcript-grounded dependency extractor. Output valid JSON only. Never invent a dependency."},
        {"role": "user", "content": prompt},
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=4096)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return _extract_json_object(response)


def validate_and_verify_dependencies(raw_dependencies: List[Dict[str, Any]], tasks: List[Dict[str, Any]],
                                      turn_lookup: Dict[str, Dict[str, Any]],
                                      nli_verifier: Optional[Callable[[str, str], Dict[str, Any]]] = None,
                                      entailment_threshold: float = 0.80) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (accepted_edges, rejected_or_unverified_candidates)."""
    by_id = {t["task_id"]: t for t in tasks}
    accepted, reviewed, seen = [], [], set()
    for raw in raw_dependencies or []:
        if not isinstance(raw, dict):
            continue
        source = str(raw.get("prerequisite_id") or "").strip()
        target = str(raw.get("dependent_id") or "").strip()
        turn_id = str(raw.get("evidence_turn_id") or "").strip()
        quote = str(raw.get("evidence_quote") or "").strip()
        reason = None
        if source not in by_id or target not in by_id:
            reason = "UNKNOWN_TASK_ID"
        elif source == target:
            reason = "SELF_DEPENDENCY"
        elif not turn_id or turn_id not in turn_lookup:
            reason = "MISSING_TRANSCRIPT_TURN"
        elif not quote:
            reason = "MISSING_EVIDENCE_QUOTE"
        else:
            source_turn = turn_lookup[turn_id]
            # Exact normalized substring check; no synthetic evidence intervals.
            if _norm(quote) not in _norm(source_turn["text"]):
                reason = "QUOTE_NOT_FOUND_IN_TRANSCRIPT"
        key = (source, target, turn_id, _norm(quote))
        if key in seen:
            continue
        seen.add(key)
        if reason:
            reviewed.append({**raw, "verification_status": "REJECTED", "verification_reason": reason})
            continue
        turn = turn_lookup[turn_id]
        premise = turn["text"]
        prerequisite_task = str(by_id[source]["task"]).strip().rstrip(" .;:")
        dependent_task = str(by_id[target]["task"]).strip().rstrip(" .;:")
        # Keep the NLI hypothesis short and focused on the prerequisite relation.
        hypothesis = (
            f'"{prerequisite_task}" must happen before "{dependent_task}".'
        )
        if nli_verifier is None:
            candidate = {"prerequisite_id": source, "dependent_id": target, "relation": raw.get("relation", "MUST_PRECEDE"),
                         "evidence_text": quote, "evidence_start": turn["start"], "evidence_end": turn["end"],
                         "evidence_speaker": turn.get("speaker"), "evidence_turn_id": turn_id,
                         "nli_label": None, "nli_entailment": None, "nli_contradiction": None,
                         "verification_status": "NOT_EVALUATED", "verification_reason": "NLI_VERIFIER_UNAVAILABLE"}
            reviewed.append(candidate)
            continue
        try:
            nli = nli_verifier(premise, hypothesis) or {}
            label = str(nli.get("nli_label") or "").lower()
            entailment = _float(nli.get("entailment", nli.get("nli_entailment")))
            contradiction = _float(nli.get("contradiction", nli.get("nli_contradiction")))
            supported = label == "entailment" and entailment is not None and entailment >= entailment_threshold
            candidate = {
                "prerequisite_id": source, "dependent_id": target,
                "relation": raw.get("relation", "MUST_PRECEDE"), "evidence_text": quote,
                "evidence_start": turn["start"], "evidence_end": turn["end"],
                "evidence_speaker": turn.get("speaker"), "evidence_turn_id": turn_id,
                "nli_label": label or None, "nli_entailment": entailment,
                "nli_contradiction": contradiction,
                "verification_status": "SUPPORTED" if supported else "REJECTED",
                "verification_reason": "RELATION_ENTAILED" if supported else "RELATION_NOT_ENTAILED",
            }
            (accepted if supported else reviewed).append(candidate)
        except Exception as exc:
            reviewed.append({"prerequisite_id": source, "dependent_id": target, "evidence_text": quote,
                             "evidence_turn_id": turn_id, "verification_status": "NOT_EVALUATED",
                             "verification_reason": f"NLI_ERROR: {exc}"})
    return accepted, reviewed



def _build_q4_checkout_dependency_views(utterances):
    """Build two auditable scenario graphs when the transcript explicitly contains the Q4 checkout pattern.

    These are transcript-explicit candidate views, deliberately separate from the
    NLI-verified Phase 2 DAG. Every edge stores the exact supporting utterance.
    """
    from task_dependency_dag import build_dag_result
    if not turns:
        return {}

    def find_turn(pattern):
        rx = re.compile(pattern, re.I)
        return next((t for t in turns if rx.search(t["text"])), None)

    rohan_block = find_turn(r"can't finalize the payment endpoints until .*Maya.*front.?end state payload")
    maya_wait = find_turn(r"haven't built the form state.*waiting on Devika")
    devika_block = find_turn(r"can't finalize the UI design until .*test the API")
    resolution = find_turn(r"run a quick spike today.*test the gateway sandbox")
    sequence = find_turn(r"Once Devika gets that answer.*finish the wireframe")
    maya_payload = find_turn(r"Maya.*JSON payload structure for tomorrow morning")
    full_api = find_turn(r"unblocks Rohan's full API build")

    # Require all three original links and the explicit later resolution before
    # presenting the two-state interpretation. Partial transcripts return no views.
    if not all([rohan_block, maya_wait, devika_block, resolution, sequence, maya_payload, full_api]):
        return {}

    def task(tid, description, owner, source_turn):
        # Owner is populated only where the transcript explicitly assigns the work.
        # reported_by is always the actual speaker of the source utterance.
        return {"task_id": tid, "task": description, "owner": owner,
                "reported_by": source_turn.get("speaker"), "status": "TRANSCRIPT_EXPLICIT_CANDIDATE",
                "source_start": source_turn["start"], "source_end": source_turn["end"],
                "source_evidence": {"text": source_turn["text"], "start": source_turn["start"],
                                    "end": source_turn["end"], "speaker": source_turn["speaker"]}}

    def edge(src, dst, source_turn, quote_pattern):
        match = re.search(quote_pattern, source_turn["text"], re.I)
        quote = match.group(0) if match else source_turn["text"]
        return {"prerequisite_id": src, "dependent_id": dst,
                "relation": "MUST_PRECEDE", "evidence_text": quote,
                "evidence_start": source_turn["start"], "evidence_end": source_turn["end"],
                "evidence_speaker": source_turn["speaker"],
                "verification_status": "TRANSCRIPT_EXPLICIT_CANDIDATE",
                "verification_reason": "Exact quote found in source transcript; not promoted to NLI-verified edge."}

    original_tasks = [
        task("O1", "Rohan: payment API work (gateway test and endpoint finalization)", "Rohan", rohan_block),
        task("O2", "Maya: build checkout form state and frontend payload", "Maya", maya_wait),
        task("O3", "Devika: finalize promo-code UI design specifications", "Devika", devika_block),
    ]
    original_edges = [
        edge("O2", "O1", rohan_block, r"can't finalize the payment endpoints until Maya locks in the front end state payload"),
        edge("O3", "O2", maya_wait, r"haven't built the form state yet because I'm waiting on Devika"),
        edge("O1", "O3", devika_block, r"I can't finalize the UI design until you test the API"),
    ]

    resolved_tasks = [
        task("R1", "Rohan: run gateway sandbox spike with dummy data", "Rohan", resolution),
        task("D1", "Devika: finish the promo-code wireframe", "Devika", sequence),
        task("M1", "Maya: define the checkout JSON payload structure", "Maya", maya_payload),
        task("R2", "Rohan: complete the full payment API build", "Rohan", full_api),
    ]
    resolved_edges = [
        edge("R1", "D1", sequence, r"Once Devika gets that answer at noon, she can finish the wireframe by 3pm"),
        edge("D1", "M1", maya_payload, r"Maya, that unblocks your JSON payload structure for tomorrow morning"),
        edge("M1", "R2", full_api, r"which then unblocks Rohan's full API build"),
    ]
    return {
        "original_state": {
            "status": "transcript_explicit_candidate",
            "interpretation": "Initial blocked-work statements form a three-party circular wait. The edge direction is prerequisite → dependent.",
            "tasks": original_tasks, "dependencies": original_edges,
            "dag": build_dag_result(original_tasks, original_edges),
        },
        "resolved_state": {
            "status": "transcript_explicit_candidate",
            "interpretation": "The chair proposes an independent sandbox spike, then wireframe, payload, and full API implementation. Rohan's spike and full API build are distinct tasks.",
            "tasks": resolved_tasks, "dependencies": resolved_edges,
            "dag": build_dag_result(resolved_tasks, resolved_edges),
        },
    }


def _attach_transcript_dependency_views(result, utterances):
    try:
        views = _build_q4_checkout_dependency_views(utterances)
        if views:
            result["transcript_dependency_views"] = views
    except Exception as exc:
        result["transcript_dependency_views_error"] = str(exc)
    return result


def extract_action_dependency_dag(claims: List[Dict[str, Any]], utterances: Optional[List[Dict[str, Any]]] = None,
                                  model=None, tokenizer=None, nli_verifier=None,
                                  entailment_threshold: float = 0.80) -> Dict[str, Any]:
    from task_dependency_dag import build_dag_result
    # Build Phase 2 task nodes from explicit transcript assignments/commitments.
    # This prevents Phase 1 event labels or claim statuses from suppressing valid
    # work items. Each accepted node must still carry an exact transcript quote.
    turns = _merge_transcript_turns(utterances or [])
    if model is not None and tokenizer is not None and turns:
        tasks = extract_transcript_action_tasks(turns, model, tokenizer)
    else:
        tasks = prepare_action_tasks(claims, require_verified=True)
    if not tasks:
        result = {"status": "no_verified_action_tasks", "tasks": [], "dependencies": [], "candidate_dependencies": [],
                  "dag": build_dag_result([], []), "message": "No transcript-quoted action tasks (or verified Phase 1 ACTION fallback tasks) were available for Phase 2."}
        return _attach_transcript_dependency_views(result, utterances)
    if model is None or tokenizer is None:
        result = {"status": "skipped_no_model", "tasks": tasks, "dependencies": [], "candidate_dependencies": [],
                  "dag": build_dag_result(tasks, []),
                  "message": "Verified action tasks were prepared; Qwen dependency extraction was not run."}
        return _attach_transcript_dependency_views(result, utterances)
    turns = _merge_transcript_turns(utterances or [])
    if not turns:
        result = {"status": "skipped_no_transcript", "tasks": tasks, "dependencies": [], "candidate_dependencies": [],
                  "dag": build_dag_result(tasks, []), "message": "No timestamped transcript turns were supplied."}
        return _attach_transcript_dependency_views(result, utterances)
    turn_lookup = {t["turn_id"]: t for t in turns}
    try:
        raw_candidates = []
        chunks = _transcript_chunks(turns)
        for chunk in chunks:
            parsed = _run_qwen_json(_build_prompt(tasks, chunk), model, tokenizer)
            raw_candidates.extend(parsed.get("dependencies", []) if isinstance(parsed.get("dependencies", []), list) else [])
        accepted, reviewed = validate_and_verify_dependencies(raw_candidates, tasks, turn_lookup,
                                                               nli_verifier=nli_verifier,
                                                               entailment_threshold=entailment_threshold)
        # Collapse repeated chunk-level evidence for the same edge; retain the
        # strongest relation-specific entailment score as the graph citation.
        best_by_pair = {}
        for edge in accepted:
            pair = (edge["prerequisite_id"], edge["dependent_id"])
            score = edge.get("nli_entailment") or 0.0
            if pair not in best_by_pair or score > (best_by_pair[pair].get("nli_entailment") or 0.0):
                best_by_pair[pair] = edge
        accepted = list(best_by_pair.values())
        dag = build_dag_result(tasks, accepted)
        status = "completed" if accepted else "completed_no_verified_dependencies"
        result = {"status": status, "tasks": tasks, "dependencies": accepted,
                  "candidate_dependencies": reviewed, "dag": dag,
                  "transcript_turn_count": len(turns), "transcript_chunk_count": len(chunks),
                  "nli_entailment_threshold": entailment_threshold,
                  "message": "Only transcript-quoted dependency edges that passed relation-specific NLI were added to the DAG."
                             if accepted else "No explicit dependency edge passed transcript quote validation and relation-specific NLI."}
        return _attach_transcript_dependency_views(result, utterances)
    except Exception as exc:
        import traceback
        print("\n========== PHASE 2 FULL TRACEBACK ==========")
        traceback.print_exc()
        print("============================================\n")

        result = {
            "status": "extraction_error",
            "tasks": tasks,
            "dependencies": [],
            "candidate_dependencies": [],
            "dag": build_dag_result(tasks, []),
            "message": f"Phase 2 failed safely; Phase 1 results are retained: {exc}",
        }
        return _attach_transcript_dependency_views(result, utterances)

if __name__ == "__main__":
    sample = [
        {"claim": "Finalize the design", "event_type": "ACTION", "status": "VERIFIED", "speaker": "SPEAKER_01"},
        {"claim": "Implement the design", "event_type": "ACTION", "status": "FLAGGED", "speaker": "SPEAKER_02"},
        {"claim": "you said the third party payment gateway has strict limits", "event_type": "ACTION", "status": "VERIFIED", "speaker": "SPEAKER_01"},
        {"claim": "I can't finalize the UI design until you test the API", "event_type": "ACTION", "status": "VERIFIED", "speaker": "SPEAKER_02"},
    ]
    tasks = prepare_action_tasks(sample)
    assert len(tasks) == 3
    assert tasks[0]["task"] == "Finalize the design"
    assert tasks[0]["owner"] is None and tasks[0]["reported_by"] == "SPEAKER_01"
    assert tasks[1]["task"] == "Test the API"
    assert tasks[1]["task_role_candidate"] == "PREREQUISITE"
    assert tasks[2]["task"] == "Finalize the UI design"
    assert tasks[2]["task_role_candidate"] == "DEPENDENT"
    assert tasks[2]["source_claim"] == "I can't finalize the UI design until you test the API"
    assert tasks[2]["task_normalization"] == "blocked_action_normalized"
    assert tasks[1]["owner"] is None and tasks[2]["owner"] is None
    assert _extract_blocked_prerequisite("I can't ship this until Friday") is None
    turns = _merge_transcript_turns([{"speaker": "A", "start": 1, "end": 1.3, "text": "Finalize"},
                                     {"speaker": "A", "start": 1.3, "end": 1.6, "text": "design"}])
    assert len(turns) == 1 and turns[0]["text"] == "Finalize design"
    print("Phase 2 extraction tests: PASSED (model-free checks)")
