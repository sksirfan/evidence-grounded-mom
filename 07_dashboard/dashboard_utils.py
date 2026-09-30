
# ============================================================
# DASHBOARD UTILITIES
# Verifiable Meeting Minutes
# ============================================================

import json
import html


# ============================================================
# 1. JSON UTILITIES
# ============================================================

def load_json(path):
    """
    Load a JSON file from disk.
    """

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# 2. TIMESTAMP UTILITIES
# ============================================================

def format_timestamp(seconds):
    """
    Convert seconds to MM:SS format.

    Examples:
        84.66  -> 01:24
        105.24 -> 01:45
    """

    if seconds is None:
        return "N/A"

    seconds = float(seconds)

    minutes = int(
        seconds // 60
    )

    secs = int(
        seconds % 60
    )

    return f"{minutes:02d}:{secs:02d}"


# ============================================================
# 3. TEXT UTILITIES
# ============================================================

def safe_html(text):
    """
    Escape text before displaying it as HTML.
    """

    if text is None:
        return ""

    return html.escape(
        str(text)
    )


# ============================================================
# 4. VERIFICATION UTILITIES
# ============================================================

def get_verification_status(result):
    """
    Return the verification status of a claim.
    """

    return result.get(
        "status",
        "UNKNOWN"
    )


def get_verification_reason(result):
    """
    Return the verification reason.
    """

    return result.get(
        "reason",
        ""
    )


def get_claim_metadata(result):
    """
    Extract commonly used claim metadata.
    """

    return {
        "claim": result.get(
            "claim",
            ""
        ),
        "speaker": result.get(
            "claim_speaker",
            "Unknown"
        ),
        "event_type": result.get(
            "claim_type",
            "Unknown"
        ),
        "status": result.get(
            "status",
            "UNKNOWN"
        ),
        "reason": result.get(
            "reason",
            ""
        )
    }


# ============================================================
# 5. EVIDENCE UTILITIES
# ============================================================

def get_supporting_evidence(result):
    """
    Extract supporting evidence from the
    content-consistency check.
    """

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

    return details.get(
        "supporting_candidates",
        []
    )


def format_evidence_timestamp(evidence):
    """
    Return an evidence timestamp in MM:SS format.
    """

    start = evidence.get(
        "start",
        0
    )

    end = evidence.get(
        "end",
        0
    )

    return (
        f"{format_timestamp(start)}"
        f"–"
        f"{format_timestamp(end)}"
    )


# ============================================================
# 6. CONSISTENCY CHECK UTILITIES
# ============================================================

CONSISTENCY_CHECKS = [
    (
        "content_consistency",
        "Content"
    ),
    (
        "speaker_consistency",
        "Speaker"
    ),
    (
        "timestamp_consistency",
        "Timestamp"
    ),
    (
        "event_type_consistency",
        "Event Type"
    )
]


def get_consistency_checks(result):
    """
    Return the four multi-attribute checks.
    """

    checks = result.get(
        "checks",
        {}
    )

    return [
        (
            key,
            display_name,
            checks.get(
                key,
                {}
            )
        )
        for key, display_name
        in CONSISTENCY_CHECKS
    ]


def is_timestamp_check(check_key):
    """
    Identify the timestamp consistency check.
    """

    return (
        check_key ==
        "timestamp_consistency"
    )


# ============================================================
# 7. SUMMARY UTILITIES
# ============================================================

def get_verification_summary(
    verification_data
):
    """
    Extract dashboard summary metrics.
    """

    return {
        "total_claims": verification_data.get(
            "total_claims",
            0
        ),
        "verified_claims": verification_data.get(
            "verified_claims",
            0
        ),
        "flagged_claims": verification_data.get(
            "flagged_claims",
            0
        ),
        "verification_rate": verification_data.get(
            "verification_rate",
            0.0
        )
    }


# ============================================================
# 8. PDF UTILITIES
# ============================================================

def get_pdf_claim_data(result):
    """
    Extract the information required by the
    PDF generator.
    """

    evidence_start = result.get(
        "evidence_start"
    )

    evidence_end = result.get(
        "evidence_end"
    )

    return {
        "claim": result.get(
            "claim",
            ""
        ),
        "speaker": result.get(
            "claim_speaker",
            "Unknown"
        ),
        "event_type": result.get(
            "claim_type",
            "Unknown"
        ),
        "status": result.get(
            "status",
            "UNKNOWN"
        ),
        "reason": result.get(
            "reason",
            ""
        ),
        "evidence_start": evidence_start,
        "evidence_end": evidence_end,
        "supporting_evidence": get_supporting_evidence(
            result
        )
    }
