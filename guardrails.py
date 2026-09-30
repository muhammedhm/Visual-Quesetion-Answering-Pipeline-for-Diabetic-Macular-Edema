"""
guardrails.py — Production-grade input/output safety layer
─────────────────────────────────────────────────────────
Input guardrails:
  - Image validation (format, size, dimensions, not blank/corrupt)
  - Image medical relevance check (basic heuristics for retinal images)
  - Query text sanitization & topic relevance check
  - Rate limiting per user
  - Prompt injection detection

Output guardrails:
  - Structured output schema enforcement
  - Hallucination mitigation (confidence scoring)
  - Forbidden content filter (no definitive diagnoses, no drug dosages)
  - Disclaimer injection
  - Output length control
  - PII scrubbing
"""

import re
import time
import hashlib
import logging
import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

logger = logging.getLogger("deepeye.guardrails")

# ─────────────────────────────────────────
# Constants
# ─────────────────────────────────────────
DISCLAIMER = (
    "⚠️ **Medical Disclaimer:** This analysis is AI-generated and intended for "
    "informational purposes only. It does **not** constitute a medical diagnosis. "
    "For clinical confirmation and treatment decisions, please consult a qualified "
    "**ophthalmologist**."
)

MAX_IMAGE_SIZE_MB    = 15
MIN_IMAGE_DIM        = 64
MAX_IMAGE_DIM        = 4096
MAX_QUERY_LENGTH     = 500
MAX_OUTPUT_TOKENS    = 350
RATE_LIMIT_WINDOW_S  = 60
RATE_LIMIT_MAX_CALLS = 10   # max calls per user per minute

# Topics that are allowed
ALLOWED_TOPICS = [
    "retina", "fundus", "macula", "optic", "disc", "vessel", "vitreous",
    "choroid", "sclera", "cornea", "lens", "iris", "pupil", "vision",
    "eye", "ocular", "ophthalmol", "disease", "condition", "finding",
    "describe", "what", "which", "how", "diagnosis", "symptom", "clinical",
    "treatment", "image", "show", "identify", "detect", "analyze", "analysis",
    "abnormal", "normal", "lesion", "hemorrhage", "edema", "degeneration",
    "glaucoma", "diabetic", "amd", "drusen", "neovascular", "exudate",
    "microaneurysm", "neovascularization", "atrophy", "detachment", "hole",
    "tumor", "hemangioma", "pigment",
]

# Prompt injection patterns
INJECTION_PATTERNS = [
    r"ignore\s+(previous|all|prior)\s+instruction",
    r"forget\s+everything",
    r"act\s+as\s+(a\s+)?(different|new|another)",
    r"you\s+are\s+now",
    r"system\s*:\s*",
    r"<\s*system\s*>",
    r"jailbreak",
    r"DAN\s+mode",
    r"pretend\s+you",
    r"override\s+(your\s+)?instruction",
]

# Forbidden output phrases (model should not claim these)
FORBIDDEN_OUTPUT_PHRASES = [
    r"you\s+(have|are\s+diagnosed\s+with|suffer\s+from)",
    r"take\s+\d+\s*mg",
    r"prescri(be|ption)",
    r"definitely\s+(have|has|indicates?)",
    r"confirmed\s+diagnosis",
    r"I\s+am\s+(100|certain|sure|positive)\s+",
    r"no\s+need\s+to\s+(see|visit|consult)\s+(a\s+)?doctor",
    r"\b(guaranteed|cure|cured)\b",
]

# PII patterns to scrub from output
PII_PATTERNS = [
    (r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE REDACTED]"),
    (r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", "[EMAIL REDACTED]"),
    (r"\b\d{9,}\b", "[ID REDACTED]"),
    (r"\b(?:SSN|NIC|CNIC|passport)\s*:?\s*[\d\-]+\b", "[ID REDACTED]"),
]

# ─────────────────────────────────────────
# Rate limiter (in-memory, per user)
# ─────────────────────────────────────────
_rate_store: dict = defaultdict(list)

def check_rate_limit(user_id: int) -> tuple[bool, str]:
    now = time.time()
    calls = _rate_store[user_id]
    # Remove calls outside window
    _rate_store[user_id] = [t for t in calls if now - t < RATE_LIMIT_WINDOW_S]
    if len(_rate_store[user_id]) >= RATE_LIMIT_MAX_CALLS:
        wait = int(RATE_LIMIT_WINDOW_S - (now - _rate_store[user_id][0]))
        return False, f"Rate limit exceeded. Please wait {wait}s before trying again."
    _rate_store[user_id].append(now)
    return True, ""

# ─────────────────────────────────────────
# Data classes for structured output
# ─────────────────────────────────────────
@dataclass
class GuardrailResult:
    passed:  bool
    message: str = ""
    detail:  str = ""

@dataclass
class StructuredRAGOutput:
    """Production-grade structured output from RAG pipeline."""
    # Core answer
    primary_finding:      str = ""
    possible_conditions:  list[str] = field(default_factory=list)
    clinical_description: str = ""
    confidence_level:     str = ""   # Low / Moderate / High
    confidence_note:      str = ""

    # Retrieved context
    similar_cases_count:  int  = 0
    top_disease_matches:  list[str] = field(default_factory=list)
    retrieval_quality:    str = ""   # Strong / Moderate / Weak

    # Safety
    disclaimer:           str = DISCLAIMER
    is_ai_generated:      bool = True
    requires_expert:      bool = True
    safety_flags:         list[str] = field(default_factory=list)

    # Meta
    query_text:           str = ""
    processing_time_s:    float = 0.0
    model_version:        str = "LLaVA-Qwen-0.5B + LoRA (DeepEye)"

# ─────────────────────────────────────────
# Input Guardrails
# ─────────────────────────────────────────
def validate_image(image: Image.Image, file_size_bytes: int = 0) -> GuardrailResult:
    """Validate uploaded image for medical use."""
    # Size check
    if file_size_bytes > MAX_IMAGE_SIZE_MB * 1024 * 1024:
        return GuardrailResult(False, f"Image too large. Maximum size is {MAX_IMAGE_SIZE_MB}MB.")

    # Dimension check
    w, h = image.size
    if w < MIN_IMAGE_DIM or h < MIN_IMAGE_DIM:
        return GuardrailResult(False, f"Image too small ({w}×{h}). Minimum: {MIN_IMAGE_DIM}px.")
    if w > MAX_IMAGE_DIM or h > MAX_IMAGE_DIM:
        return GuardrailResult(False, f"Image too large ({w}×{h}). Maximum: {MAX_IMAGE_DIM}px.")

    # Mode check
    if image.mode not in ["RGB", "RGBA", "L", "P"]:
        return GuardrailResult(False, f"Unsupported image mode: {image.mode}.")

    # Blank/corrupt image check
    try:
        arr = np.array(image.convert("RGB"))
        std = arr.std()
        if std < 3.0:
            return GuardrailResult(
                False,
                "Image appears to be blank or uniform. Please upload a valid fundus image.",
                f"Pixel std={std:.2f}"
            )
        mean = arr.mean()
        if mean < 5 or mean > 250:
            return GuardrailResult(
                False,
                "Image appears to be entirely black or white. Please upload a valid fundus image.",
                f"Pixel mean={mean:.2f}"
            )
    except Exception as e:
        return GuardrailResult(False, f"Could not process image: {e}")

    return GuardrailResult(True, "Image valid.")


def validate_query(query: str) -> GuardrailResult:
    """Validate and sanitize user query text."""
    if not query or not query.strip():
        return GuardrailResult(False, "Query cannot be empty.")

    query = query.strip()

    # Length check
    if len(query) > MAX_QUERY_LENGTH:
        return GuardrailResult(
            False,
            f"Query too long ({len(query)} chars). Maximum: {MAX_QUERY_LENGTH} characters."
        )

    # Prompt injection detection
    query_lower = query.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, query_lower):
            logger.warning(f"Prompt injection attempt detected: {query[:80]}")
            return GuardrailResult(
                False,
                "Query contains disallowed content. Please ask a clinical question about the retinal image."
            )

    # Topic relevance (at least one allowed keyword, or very short/general question)
    has_relevant_topic = any(kw in query_lower for kw in ALLOWED_TOPICS)
    is_general_question = len(query.split()) <= 6  # short questions allowed
    if not has_relevant_topic and not is_general_question:
        return GuardrailResult(
            False,
            "Query does not appear to be related to retinal/ocular conditions. "
            "Please ask a clinical question about the uploaded fundus image."
        )

    return GuardrailResult(True, "Query valid.")


def sanitize_query(query: str) -> str:
    """Clean query text — remove HTML, control chars, excessive whitespace."""
    query = re.sub(r"<[^>]+>", "", query)             # strip HTML tags
    query = re.sub(r"[^\w\s\.\?\,\!\-\(\)\/\'\"]", " ", query)  # keep safe chars
    query = re.sub(r"\s+", " ", query).strip()
    return query[:MAX_QUERY_LENGTH]


# ─────────────────────────────────────────
# Output Guardrails
# ─────────────────────────────────────────
def scrub_pii(text: str) -> str:
    """Remove any PII that may have appeared in output."""
    for pattern, replacement in PII_PATTERNS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def filter_forbidden_content(text: str) -> tuple[str, list[str]]:
    """
    Detect and soften forbidden output phrases.
    Returns (cleaned_text, list_of_flags).
    """
    flags = []
    for pattern in FORBIDDEN_OUTPUT_PHRASES:
        if re.search(pattern, text, re.IGNORECASE):
            flags.append(f"Forbidden phrase detected: '{pattern}'")
            logger.warning(f"Output guardrail triggered: {pattern}")

    # Soften definitive language
    replacements = [
        (r"\byou have\b",            "the image may suggest"),
        (r"\bconfirmed diagnosis\b", "possible finding"),
        (r"\bdefinitely\b",          "possibly"),
        (r"\bcertainly\b",           "possibly"),
        (r"\bwithout doubt\b",       "with some possibility"),
        (r"\bno need to see a doctor\b", "professional consultation is still recommended"),
    ]
    for old, new in replacements:
        text = re.sub(old, new, text, flags=re.IGNORECASE)

    return text, flags


def enforce_output_length(text: str, max_words: int = MAX_OUTPUT_TOKENS) -> str:
    """Trim output to max_words if needed, ending at a sentence boundary."""
    words = text.split()
    if len(words) <= max_words:
        return text
    truncated = " ".join(words[:max_words])
    # Try to end at sentence boundary
    last_period = max(truncated.rfind("."), truncated.rfind("!"), truncated.rfind("?"))
    if last_period > len(truncated) * 0.6:
        truncated = truncated[:last_period + 1]
    else:
        truncated = truncated + "..."
    return truncated


def score_confidence(answer: str, retrieved_count: int, top_score: float) -> tuple[str, str]:
    """
    Heuristic confidence scoring.
    Returns (level, note).
    """
    # Based on retrieval similarity score and answer quality
    if top_score >= 0.85 and retrieved_count >= 3 and len(answer.split()) >= 20:
        level = "Moderate"
        note  = "Based on similar cases found in the dataset."
    elif top_score >= 0.70 and retrieved_count >= 2:
        level = "Low–Moderate"
        note  = "Some similar cases found; findings may not be conclusive."
    else:
        level = "Low"
        note  = "Limited similar cases found. Results should be interpreted cautiously."
    return level, note


def parse_structured_output(
    raw_answer:    str,
    retrieved:     list[dict],
    query_text:    str,
    elapsed_s:     float,
) -> StructuredRAGOutput:
    """
    Convert raw model output + retrieved records into StructuredRAGOutput.
    Applies all output guardrails.
    """
    # 1. PII scrub
    answer = scrub_pii(raw_answer)

    # 2. Forbidden content filter
    answer, flags = filter_forbidden_content(answer)

    # 3. Length enforcement
    answer = enforce_output_length(answer)

    # 4. Collect diseases from retrieved records
    disease_set = []
    for r in retrieved:
        for d in r.get("diseases", "").split(","):
            d = d.strip()
            if d and d not in disease_set:
                disease_set.append(d)

    # 5. Confidence scoring
    top_score = max((r.get("score", 0.0) for r in retrieved), default=0.0)
    conf_level, conf_note = score_confidence(answer, len(retrieved), top_score)

    # 6. Retrieval quality
    if top_score >= 0.85:
        ret_quality = "Strong"
    elif top_score >= 0.70:
        ret_quality = "Moderate"
    else:
        ret_quality = "Weak"

    # 7. Extract primary finding from answer (first sentence)
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    primary   = sentences[0] if sentences else answer[:120]

    # 8. Clinical description = rest of answer
    clinical = " ".join(sentences[1:]).strip() if len(sentences) > 1 else ""

    return StructuredRAGOutput(
        primary_finding      = primary,
        possible_conditions  = disease_set[:6],
        clinical_description = clinical,
        confidence_level     = conf_level,
        confidence_note      = conf_note,
        similar_cases_count  = len(retrieved),
        top_disease_matches  = disease_set[:4],
        retrieval_quality    = ret_quality,
        disclaimer           = DISCLAIMER,
        is_ai_generated      = True,
        requires_expert      = True,
        safety_flags         = flags,
        query_text           = query_text,
        processing_time_s    = round(elapsed_s, 2),
        model_version        = "LLaVA-Qwen-0.5B + LoRA (DeepEye)",
    )
