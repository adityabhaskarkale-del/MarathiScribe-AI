"""CER / WER with Unicode (NFC) normalization - important for Devanagari matras."""
import re
import unicodedata


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text))
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def corpus_cer(refs, hyps) -> float:
    refs, hyps = [normalize(r) for r in refs], [normalize(h) for h in hyps]
    errors = sum(edit_distance(r, h) for r, h in zip(refs, hyps))
    return errors / max(1, sum(len(r) for r in refs))


def corpus_wer(refs, hyps) -> float:
    refs, hyps = [normalize(r).split() for r in refs], [normalize(h).split() for h in hyps]
    errors = sum(edit_distance(r, h) for r, h in zip(refs, hyps))
    return errors / max(1, sum(len(r) for r in refs))
