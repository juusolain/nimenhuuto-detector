from __future__ import annotations

import math
import os
from dataclasses import dataclass


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _audio_device(value: str | None) -> int | str | None:
    if value is None or not value.strip():
        return None
    value = value.strip()
    try:
        return int(value)
    except ValueError:
        return value


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str
    telegram_chat_id: str
    target_phrases: tuple[str, ...]
    whisper_model: str
    whisper_device: str
    whisper_compute_type: str
    whisper_language: str
    match_threshold: float
    min_log_probability: float
    sample_rate: int
    window_seconds: float
    stride_seconds: float
    block_seconds: float
    alert_cooldown_seconds: float
    min_audio_dbfs: float
    audio_device: int | str | None
    log_level: str

    @classmethod
    def from_env(cls) -> Settings:
        phrases = tuple(
            phrase.strip()
            for phrase in os.getenv("TARGET_PHRASES", "nimenhuuto,nimen huuto").split(",")
            if phrase.strip()
        )
        settings = cls(
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", "").strip(),
            target_phrases=phrases,
            whisper_model=os.getenv("WHISPER_MODEL", "small").strip(),
            whisper_device=os.getenv("WHISPER_DEVICE", "cpu").strip(),
            whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8").strip(),
            whisper_language=os.getenv("WHISPER_LANGUAGE", "fi").strip(),
            match_threshold=_get_float("MATCH_THRESHOLD", 0.86),
            min_log_probability=_get_float("MIN_LOG_PROBABILITY", -0.8),
            sample_rate=_get_int("SAMPLE_RATE", 16_000),
            window_seconds=_get_float("WINDOW_SECONDS", 6.0),
            stride_seconds=_get_float("STRIDE_SECONDS", 2.0),
            block_seconds=_get_float("BLOCK_SECONDS", 0.25),
            alert_cooldown_seconds=_get_float("ALERT_COOLDOWN_SECONDS", 600.0),
            min_audio_dbfs=_get_float("MIN_AUDIO_DBFS", -50.0),
            audio_device=_audio_device(os.getenv("AUDIO_DEVICE")),
            log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not self.target_phrases:
            raise ValueError("TARGET_PHRASES must contain at least one phrase")
        finite_values = {
            "MATCH_THRESHOLD": self.match_threshold,
            "MIN_LOG_PROBABILITY": self.min_log_probability,
            "WINDOW_SECONDS": self.window_seconds,
            "STRIDE_SECONDS": self.stride_seconds,
            "BLOCK_SECONDS": self.block_seconds,
            "ALERT_COOLDOWN_SECONDS": self.alert_cooldown_seconds,
            "MIN_AUDIO_DBFS": self.min_audio_dbfs,
        }
        for name, value in finite_values.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= self.match_threshold <= 1.0:
            raise ValueError("MATCH_THRESHOLD must be between 0 and 1")
        if self.min_log_probability > 0:
            raise ValueError("MIN_LOG_PROBABILITY must be no greater than zero")
        if self.sample_rate != 16_000:
            raise ValueError("SAMPLE_RATE must be 16000 for Whisper audio")
        if self.window_seconds <= 0 or self.stride_seconds <= 0:
            raise ValueError("WINDOW_SECONDS and STRIDE_SECONDS must be positive")
        if self.stride_seconds > self.window_seconds:
            raise ValueError("STRIDE_SECONDS cannot exceed WINDOW_SECONDS")
        if self.block_seconds <= 0 or self.block_seconds > self.stride_seconds:
            raise ValueError("BLOCK_SECONDS must be positive and no greater than STRIDE_SECONDS")
        if self.alert_cooldown_seconds < 0:
            raise ValueError("ALERT_COOLDOWN_SECONDS cannot be negative")

    def require_telegram(self) -> None:
        if not self.telegram_bot_token or not self.telegram_chat_id:
            raise ValueError(
                "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are required unless --dry-run is used"
            )
