from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nimenhuuto.matching import PhraseMatch, PhraseMatcher


@dataclass(frozen=True)
class Analysis:
    transcript: str
    match: PhraseMatch | None
    recognition_log_probability: float | None


class WhisperKeywordDetector:
    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        compute_type: str,
        language: str,
        targets: Sequence[str],
        match_threshold: float,
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed; run 'pip install -e .'"
            ) from exc

        self.matcher = PhraseMatcher(targets, match_threshold)
        self.language = language
        self.model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
        )

    def analyze(self, audio: np.ndarray | str | Path) -> Analysis:
        source: Any = str(audio) if isinstance(audio, Path) else audio
        segments, _ = self.model.transcribe(
            source,
            language=self.language,
            beam_size=5,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
        )

        segment_list = list(segments)
        transcript = " ".join(segment.text.strip() for segment in segment_list).strip()
        log_probabilities = [
            float(segment.avg_logprob)
            for segment in segment_list
            if segment.avg_logprob is not None and math.isfinite(segment.avg_logprob)
        ]
        average_log_probability = (
            sum(log_probabilities) / len(log_probabilities)
            if log_probabilities
            else None
        )

        best_segment_match: PhraseMatch | None = None
        match_log_probability: float | None = None
        for segment in segment_list:
            segment_match = self.matcher.match(segment.text)
            if segment_match is None:
                continue
            segment_log_probability = (
                float(segment.avg_logprob)
                if segment.avg_logprob is not None
                and math.isfinite(segment.avg_logprob)
                else None
            )
            if (
                best_segment_match is None
                or segment_match.score > best_segment_match.score
                or (
                    segment_match.score == best_segment_match.score
                    and (
                        segment_log_probability
                        if segment_log_probability is not None
                        else float("-inf")
                    )
                    > (
                        match_log_probability
                        if match_log_probability is not None
                        else float("-inf")
                    )
                )
            ):
                best_segment_match = segment_match
                match_log_probability = segment_log_probability

        match = best_segment_match or self.matcher.match(transcript)
        return Analysis(
            transcript=transcript,
            match=match,
            recognition_log_probability=(
                match_log_probability
                if best_segment_match is not None
                else average_log_probability
            ),
        )

