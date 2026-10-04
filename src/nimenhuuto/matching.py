from __future__ import annotations

import re
import time
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher


_NON_WORD = re.compile(r"[^\w]+", flags=re.UNICODE)
_UNDERSCORES = re.compile(r"_+")


def normalize_words(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = _UNDERSCORES.sub(" ", normalized)
    return tuple(word for word in _NON_WORD.sub(" ", normalized).split() if word)


@dataclass(frozen=True)
class PhraseMatch:
    target: str
    heard: str
    score: float


class PhraseMatcher:
    """Tolerant matching for Whisper spelling and word-boundary variations."""

    def __init__(self, targets: Sequence[str], threshold: float = 0.86) -> None:
        if not targets:
            raise ValueError("At least one target phrase is required")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.targets = tuple(targets)
        self.threshold = threshold
        self._normalized_targets = tuple(
            (target, normalize_words(target)) for target in self.targets
        )
        if any(not words for _, words in self._normalized_targets):
            raise ValueError("Target phrases cannot be empty after normalization")

    def match(self, transcript: str) -> PhraseMatch | None:
        words = normalize_words(transcript)
        if not words:
            return None

        compact_transcript = "".join(words)
        for target, target_words in self._normalized_targets:
            compact_target = "".join(target_words)
            if compact_target in compact_transcript:
                return PhraseMatch(target=target, heard=target, score=1.0)

        best: PhraseMatch | None = None
        for target, target_words in self._normalized_targets:
            compact_target = "".join(target_words)
            max_ngram = min(len(words), max(3, len(target_words) + 2))
            for size in range(1, max_ngram + 1):
                for start in range(len(words) - size + 1):
                    heard_words = words[start : start + size]
                    candidate = "".join(heard_words)
                    length_ratio = len(candidate) / len(compact_target)
                    if not 0.55 <= length_ratio <= 1.6:
                        continue
                    score = SequenceMatcher(None, compact_target, candidate).ratio()
                    if best is None or score > best.score:
                        best = PhraseMatch(
                            target=target,
                            heard=" ".join(heard_words),
                            score=score,
                        )

        if best is not None and best.score >= self.threshold:
            return best
        return None


class Cooldown:
    def __init__(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("Cooldown cannot be negative")
        self.seconds = seconds
        self._last_triggered: float | None = None

    def try_acquire(self, now: float | None = None) -> bool:
        current = time.monotonic() if now is None else now
        if (
            self._last_triggered is not None
            and current - self._last_triggered < self.seconds
        ):
            return False
        self._last_triggered = current
        return True
