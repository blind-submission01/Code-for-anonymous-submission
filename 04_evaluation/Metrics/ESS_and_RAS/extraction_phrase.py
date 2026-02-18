"""
Repair action 与 root cause 抽取算法实现.
"""
from __future__ import annotations

import re
import string
import unicodedata
from typing import Iterable, List, Sequence, Set, Tuple

import spacy
from spacy.tokens import Doc, Token


TARGET_OBJ_DEPS = {"dobj", "iobj", "pobj", "nsubjpass"}
ARTICLES = {"a", "an", "the"}
NON_CONTENT_WORDS = {
    "this",
    "that",
    "these",
    "those",
    "some",
    "any",
    "new",
    "old",
    "proper",
    "appropriate",
    "various",
}

# Verb synonym normalization for RAS.
#
# Notes:
# - This is intentionally NOT a whitelist: verbs not listed here will be kept as-is (lowercased lemma).
# - Canonical verbs are coarse-grained to increase alignability across paraphrases.
VERB_SYNONYM_CANONICAL = {
    # add / introduce
    "add": "add",
    "introduce": "add",
    "append": "add",
    "insert": "add",
    "include": "add",
    "create": "add",
    "implement": "add",
    "define": "add",
    "register": "add",
    "enable": "add",
    "inject": "add",
    # remove / delete
    "remove": "remove",
    "delete": "remove",
    "drop": "remove",
    "strip": "remove",
    "eliminate": "remove",
    "purge": "remove",
    "disable": "remove",
    "deprecate": "remove",
    # replace / swap
    "replace": "replace",
    "swap": "replace",
    "substitute": "replace",
    "rename": "replace",
    "convert": "replace",
    "migrate": "replace",
    # fix / correct
    "fix": "fix",
    "resolve": "fix",
    "correct": "fix",
    "repair": "fix",
    "patch": "fix",
    "address": "fix",
    "mitigate": "fix",
    # change / modify / update
    "change": "change",
    "modify": "change",
    "adjust": "change",
    "alter": "change",
    "refactor": "change",
    "rewrite": "change",
    "rework": "change",
    "restructure": "change",
    "update": "update",
    "upgrade": "update",
    "bump": "update",
    "refresh": "update",
    # validate / check / enforce
    "validate": "validate",
    "check": "validate",
    "verify": "validate",
    "ensure": "validate",
    "confirm": "validate",
    "test": "validate",
    "assert": "validate",
    "guard": "validate",
    "enforce": "validate",
    "restrict": "validate",
    "limit": "validate",
    # sanitize / escape / encode
    "sanitize": "sanitize",
    "escape": "sanitize",
    "encode": "sanitize",
    "filter": "sanitize",
    "normalize": "sanitize",
    "canonicalize": "sanitize",
    # handle / support
    "handle": "handle",
    "support": "handle",
    "process": "handle",
    "parse": "handle",
    "catch": "handle",
    "avoid": "handle",
    "prevent": "handle",
    # raise / throw
    "raise": "raise",
    "throw": "raise",
    # dependency / config style (kept but normalized for matching)
    "rely": "depend",
    "depend": "depend",
    "require": "depend",
}

# Negative / absence markers used for Root Cause concept extraction.
# Expanded based on token statistics observed in `Evaluation/test_swelite_rc_rs.jsonl` (root_cause).
# Note: we compare against spaCy lemmas (`token.lemma_.lower()`), so include lemma forms (e.g. "miss").
NEGATIVE_TERMS = {
    # Explicit negation / absence
    "no",
    "not",
    "none",
    "without",
    "lack",
    "absent",
    "non",
    "never",
    # Missing / omission
    "miss",
    "missing",
    "omit",
    "omitted",
    "omission",
    "ignore",
    "ignored",
    # Failure / incorrectness
    "fail",
    "failed",
    "failure",
    "wrong",
    "incorrect",
    "invalid",
    "inconsistent",
    "unexpected",
    "broken",
    "incomplete",
    "insufficient",
    "improper",
    "mismatch",
    # Un-* error states (common in root-cause descriptions)
    "unable",
    "unhandled",
    "unrecognized",
    "unsupported",
    "nonexistent",
    "undefined",
    "null",
    "unsafe",
    "unescaped",
    "uninitialized",
    "unsanitized",
    "unchecked",
    "unvalidated",
}

_CODE_SPAN_RE = re.compile(r"`([^`]+)`")


def _coerce_doc(text: str | Doc, nlp) -> Doc:
    if isinstance(text, Doc):
        return text
    return nlp(text)


def _normalize_action_verb(verb_lemma: str) -> str:
    lemma = (verb_lemma or "").strip().lower()
    if not lemma:
        return ""
    return VERB_SYNONYM_CANONICAL.get(lemma, lemma)

def _preprocess_repair_action_text(text: str) -> tuple[str, dict[str, str]]:
    """
    Preprocess text for RAS phrase extraction:
      - Extract inline-code spans delimited by backticks as atomic placeholder tokens.
      - Remove all punctuation outside those code spans.
      - Keep placeholders so spaCy sees them as single tokens; later map back.
    """
    code_map: dict[str, str] = {}

    def _repl(m: re.Match) -> str:
        key = f"CODETOKEN{len(code_map)}"
        code_map[key] = m.group(1)
        return f" {key} "

    text = _CODE_SPAN_RE.sub(_repl, text)
    # Remove punctuation outside code spans (which are already replaced by placeholders).
    # Also strip unicode punctuation/symbols to reduce noisy tokens from code-like text.
    cleaned_chars: list[str] = []
    for ch in text:
        cat0 = unicodedata.category(ch)[:1]
        if ch in string.punctuation or cat0 in {"P", "S"}:
            cleaned_chars.append(" ")
        else:
            cleaned_chars.append(ch)
    text = "".join(cleaned_chars)
    text = re.sub(r"\s+", " ", text).strip()
    return text, code_map


def _mapped_text(token: Token, code_map: dict[str, str]) -> str:
    return code_map.get(token.text, token.text)


def _is_code_like_text(text: str) -> bool:
    if not text:
        return False
    if any(mark in text for mark in ("`", "::", "/", "_", ".", "#")):
        return True
    if re.search(r"[a-z][A-Z]", text):
        return True
    if any(ch.isdigit() for ch in text):
        return True
    return False


def _normalize_text_pieces(text: str) -> List[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    raw = raw.strip("`'\"()[]{}")

    # If it's a file path, prefer basename stem (more stable across contexts).
    if "/" in raw:
        raw = raw.split("/")[-1]
    for ext in (".py", ".js", ".ts", ".java", ".c", ".cc", ".cpp", ".h", ".hpp", ".rs", ".go"):
        if raw.endswith(ext):
            raw = raw[: -len(ext)]
            break

    raw = raw.replace("::", " ")
    raw = raw.replace("/", " ")
    raw = raw.replace("_", " ")
    raw = raw.replace("-", " ")
    raw = raw.replace(".", " ")
    raw = re.sub(r"[(),;:]", " ", raw)

    pieces = [p.lower().strip() for p in raw.split()]
    filtered: List[str] = []
    for p in pieces:
        if not p:
            continue
        if p in ARTICLES or p in NON_CONTENT_WORDS:
            continue
        if len(p) <= 1:
            continue
        filtered.append(p)
    return filtered


def _normalize_object_head(obj: Token) -> str:
    if obj.is_stop:
        return ""
    if obj.pos_ in {"NOUN", "PROPN"}:
        lemma = obj.lemma_.lower().strip()
        if lemma and lemma not in ARTICLES and lemma not in NON_CONTENT_WORDS:
            return lemma
    return ""


def _normalize_object_phrase(tokens: Sequence[Token], obj: Token, code_map: dict[str, str]) -> str:
    """
    v2-style normalization:
      - favor a single stable head (noun lemma / identifier head)
      - strip articles & common non-content words
      - reduce code-like spans to stable pieces for matching
    """
    pieces: List[str] = []
    for t in tokens:
        if t.is_stop:
            continue
        lemma = t.lemma_.lower().strip()
        if lemma in ARTICLES or lemma in NON_CONTENT_WORDS:
            continue

        mapped = _mapped_text(t, code_map)
        if _is_code_like_text(mapped):
            pieces.extend(_normalize_text_pieces(mapped))
            continue

        if t.pos_ in {"NOUN", "PROPN"} and lemma:
            pieces.append(lemma)

    # Prefer last identifier-like piece when code tokens exist; otherwise fallback to noun head.
    code_head = pieces[-1] if pieces else ""
    head = _normalize_object_head(obj)
    return code_head or head


def extract_repair_action_tuples(sentence: str | Doc, nlp) -> List[Tuple[str, str]]:
    """
    对齐 VulAdvisor 的 Repair Action 抽取:
      - 遍历 VERB
      - 找到其关联 object（含被动）
      - object 合并 compound

    v1.8 tweaks:
      - verb synonym normalization (no whitelist)
      - extract both (compound phrase) and (normalized head phrase) for each object
    """
    code_map: dict[str, str] = {}
    if isinstance(sentence, Doc):
        doc = sentence
    else:
        processed, code_map = _preprocess_repair_action_text(sentence)
        doc = nlp(processed)

    pairs: List[Tuple[str, str]] = []
    for token in doc:
        if token.pos_ != "VERB":
            continue
        verb = _normalize_action_verb(token.lemma_)
        if not verb:
            continue
        for obj in _iter_objects(token):
            if obj.like_num:
                continue
            for phrase in _build_object_phrases(obj, code_map):
                pairs.append((verb, phrase))

    return sorted(set(pairs))


def extract_repair_actions(sentence: str | Doc, nlp) -> List[str]:
    """基于依存解析抽取 (verb, object) 短语（用于可读展示，去重排序）。"""
    pairs = extract_repair_action_tuples(sentence, nlp)
    return sorted({f"{v} {o}".strip() for v, o in pairs})


def _iter_objects(verb: Token) -> Sequence[Token]:
    objs: List[Token] = []
    for child in verb.children:
        if child.is_punct:
            continue
        if child.dep_ in TARGET_OBJ_DEPS:
            objs.append(child)
        if child.dep_ == "prep":
            objs.extend(_find_pobjs(child))
    return objs


def _find_pobjs(prep: Token) -> List[Token]:
    return [gc for gc in prep.children if gc.dep_ == "pobj" and not gc.is_punct]


def _build_object_phrases(obj: Token, code_map: dict[str, str]) -> Set[str]:
    """
    Return a set of object phrases containing:
      1) compound phrase (surface-ish, preserves code span text via code_map)
      2) normalized head phrase (v2-style)
    """
    phrases: Set[str] = set()
    compounds = [t for t in obj.children if t.dep_ == "compound"]
    compounds_sorted = sorted(compounds, key=lambda t: t.i)
    prefix = " ".join(_mapped_text(t, code_map) for t in compounds_sorted).strip()
    core = _mapped_text(obj, code_map).strip()
    compound_phrase = f"{prefix} {core}".strip()
    if compound_phrase:
        phrases.add(re.sub(r"\s+", " ", compound_phrase).strip().lower())

    normalized = _normalize_object_phrase([*compounds_sorted, obj], obj, code_map)
    if normalized:
        phrases.add(normalized.strip().lower())

    return {p for p in phrases if p}


def extract_root_cause_concepts(sentence: str | Doc, nlp) -> List[str]:
    """
    抽取关键缺陷概念 (短语).

    设计目标（ESS 的 key units）:
      - 捕获名词短语的核心技术概念（compound/amod）
      - 捕获否定约束（missing/no/lack/without）
      - 适度归一化，减少形态变化造成的 mismatch
    """
    doc = _coerce_doc(sentence, nlp)
    concepts: Set[str] = set()
    for token in doc:
        if token.pos_ not in {"NOUN", "PROPN"}:
            continue
        if token.is_stop:
            continue

        modifiers: List[Token] = []
        for child in token.children:
            child_lemma = child.lemma_.lower()
            if child.dep_ in {"compound", "amod"}:
                modifiers.append(child)
            elif child_lemma in NEGATIVE_TERMS:
                modifiers.append(child)
            elif child.dep_ in {"det", "neg"} and child_lemma in NEGATIVE_TERMS:
                modifiers.append(child)

        if modifiers:
            modifiers_sorted = sorted(modifiers, key=lambda t: t.i)
            phrase = " ".join([m.lemma_.lower() for m in modifiers_sorted] + [token.lemma_.lower()])
            concepts.add(phrase.strip())

        concepts.add(token.lemma_.lower())
    return sorted(concepts)


def extract_phrases(
    sentence: str,
    strategy: str,
    nlp=None,
) -> List[str]:
    """统一接口，根据策略返回短语列表."""
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")
    if strategy == "extract_repair_actions":
        return extract_repair_actions(sentence, nlp)
    if strategy == "extract_root_cause_concepts":
        return extract_root_cause_concepts(sentence, nlp)
    raise ValueError(f"Unknown strategy: {strategy}")


if __name__ == "__main__":
    root_casue = "Console and stream outputs that carry arbitrary text from Python (including panic strings from polars) are sometimes being wrapped in `CellOutput` without an explicit MIME type. In `marimo/_messaging/console_output_worker.py` and `marimo/_pyodide/streams.py`, `CellOutput` is constructed with only `channel=...` and `data=data`, omitting `mimetype=\"text/plain\"`.\n\nThis allows the `CellOutput` machinery or frontend renderer to treat the payload as HTML (or otherwise non-escaped rich content) instead of plain text. When the error string contains `<` and `::` (e.g. `core::iter::adapters::genericshunt<i,r` from a Rust panic), the frontend attempts to interpret portions of this string as tag names and uses them in `document.createElement(...)`. Because the resulting tag name is invalid, the browser throws `Failed to execute 'createElement' on 'Document': The tag name provided ('core::iter::adapters::genericshunt<i,r') is not a valid name.` and marimo shows \"Something went wrong\".\n\nOther paths (e.g. `CellOutput.stdout`/`stderr` in `marimo/_messaging/cell_output.py`) already use `mimetype=\"text/plain\"`, so those outputs are correctly escaped. The bug is limited to the paths where `CellOutput` is instantiated without specifying a plain-text MIME type."
    repair_suggestion = "Replace the unsupported `platform_system` marker with the widely supported `sys_platform` marker that older pip versions know about. Concretely:\n1. In `setup.py`, locate the `extras_require` dict entry with key `'socks:platform_system == \"Windows\" and python_version<\"3.3\"'`.\n2. Change the environment marker to use `sys_platform == \"win32\"` instead of `platform_system == \"Windows\"`, e.g. `'socks:sys_platform == \"win32\" and python_version<\"3.3\"'`.\n3. Keep the associated value `['win_inet_pton']` unchanged so that the behavior (extra dependency on Windows for Python < 3.3) is preserved, but now uses a marker understood by both old and new pip."
    print(extract_phrases(repair_suggestion, "extract_repair_actions"))
    print(extract_phrases(root_casue, "extract_root_cause_concepts"))
