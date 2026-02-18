"""
ESRA / ESS / RAS scoring utilities.

- RAS: set Jaccard over (verb, object) tuples (v1.8: more robust than multiset Jaccard).
- ESS: our innovation for Root Cause, Jaccard over extracted defect concepts.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Sequence, Tuple

import spacy

try:
    from .extraction_phrase import extract_repair_action_tuples, extract_root_cause_concepts
except ImportError:  # pragma: no cover
    from extraction_phrase import extract_repair_action_tuples, extract_root_cause_concepts


def compute_ras(
    reference_repair_suggestion: str,
    predicted_repair_suggestion: str,
    *,
    nlp=None,
    empty_score: float = 1.0,
) -> float:
    """
    Repair Action Score (RAS).

    v1.8: extract (verb, object) tuples and compute set Jaccard similarity.
    """
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")
    ref_pairs = set(extract_repair_action_tuples(reference_repair_suggestion, nlp))
    pred_pairs = set(extract_repair_action_tuples(predicted_repair_suggestion, nlp))
    return set_jaccard_similarity(ref_pairs, pred_pairs, empty_score=empty_score)


def compute_ess(
    reference_root_cause: str,
    predicted_root_cause: str,
    *,
    nlp=None,
    empty_score: float = 0.0,
) -> float:
    """
    Error State Score (ESS) for Root Cause.

    - Extract defect concepts (noun-phrase-like key units)
    - Compute set Jaccard similarity
    """
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")
    ref = set(extract_root_cause_concepts(reference_root_cause, nlp))
    pred = set(extract_root_cause_concepts(predicted_root_cause, nlp))
    return set_jaccard_similarity(ref, pred, empty_score=empty_score)


def compute_esra(
    reference_root_cause: str,
    predicted_root_cause: str,
    reference_repair_suggestion: str,
    predicted_repair_suggestion: str,
    *,
    nlp=None,
    empty_score_ess: float = 0.0,
    empty_score_ras: float = 1.0,
) -> float:
    """ESRA = (ESS + RAS) / 2."""
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")
    ess = compute_ess(
        reference_root_cause,
        predicted_root_cause,
        nlp=nlp,
        empty_score=empty_score_ess,
    )
    ras = compute_ras(
        reference_repair_suggestion,
        predicted_repair_suggestion,
        nlp=nlp,
        empty_score=empty_score_ras,
    )
    return (ess + ras) / 2.0


def set_jaccard_similarity(
    a: Iterable[object],
    b: Iterable[object],
    *,
    empty_score: float = 0.0,
) -> float:
    set_a = set(a)
    set_b = set(b)
    union = set_a | set_b
    if not union:
        return float(empty_score)
    return len(set_a & set_b) / len(union)


def generalized_jaccard_similarity(
    a: Sequence[Tuple[str, str]],
    b: Sequence[Tuple[str, str]],
    *,
    empty_score: float = 1.0,
) -> float:
    """
    Generalized Jaccard for multisets (Counter-based).

    Matches VulAdvisor's `generalized_jaccard_similarity` behavior.
    """
    c1 = Counter(a)
    c2 = Counter(b)
    intersection_sum = sum((c1 & c2).values())
    union_sum = sum((c1 | c2).values())
    if union_sum == 0:
        return float(empty_score)
    return intersection_sum / union_sum


if __name__ == "__main__":
    rc_ref = "Root cause: missing bounds check when copying user-controlled length."
    rc_pred = "The bug is due to a missing bounds check on the copy length."
    rs_ref = "Add a bounds check before copying and validate the length."
    rs_pred = "Add checks to validate length before copying data."

    nlp = spacy.load("en_core_web_sm")
    print("ESS:", compute_ess(rc_ref, rc_pred, nlp=nlp))
    print("RAS:", compute_ras(rs_ref, rs_pred, nlp=nlp))
    print("ESRA:", compute_esra(rc_ref, rc_pred, rs_ref, rs_pred, nlp=nlp))
