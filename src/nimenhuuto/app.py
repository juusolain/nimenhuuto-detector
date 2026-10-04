from __future__ import annotations

import argparse
import logging
import math
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Protocol

from dotenv import load_dotenv

from nimenhuuto.audio import MicrophoneWindows, dbfs
from nimenhuuto.config import Settings
from nimenhuuto.detector import Analysis, WhisperKeywordDetector
from nimenhuuto.matching import Cooldown
from nimenhuuto.telegram import (
    TelegramDispatcher,
    TelegramError,
    TelegramNotifier,
)


LOGGER = logging.getLogger(__name__)


class AlertSender(Protocol):
    def send(self, text: str) -> None: ...


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect a shouted 'nimenhuuto' and send a Telegram alert."
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="list audio input devices and exit",
    )
    parser.add_argument(
        "--device",
        help="audio device index or name (overrides AUDIO_DEVICE)",
    )
    parser.add_argument(
        "--test-audio",
        type=Path,
        metavar="FILE",
        help="analyze one audio file instead of listening to a microphone",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="log detections without sending Telegram messages",
    )
    return parser


def _device_argument(value: str | None, fallback: int | str | None) -> int | str | None:
    if value is None:
        return fallback
    try:
        return int(value)
    except ValueError:
        return value


def _alert_text() -> str:
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    return f"🔔 Nimenhuuto havaittu!\nAika: {timestamp}"


def _make_detector(settings: Settings) -> WhisperKeywordDetector:
    LOGGER.info(
        "Loading Whisper model %s on %s (%s)",
        settings.whisper_model,
        settings.whisper_device,
        settings.whisper_compute_type,
    )
    return WhisperKeywordDetector(
        model_name=settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
        language=settings.whisper_language,
        targets=settings.target_phrases,
        match_threshold=settings.match_threshold,
    )


def _handle_detection(
    analysis: Analysis,
    *,
    notifier: AlertSender | None,
    cooldown: Cooldown,
    min_log_probability: float,
) -> bool:
    if analysis.match is None:
        return False

    log_probability = analysis.recognition_log_probability
    if log_probability is None or log_probability < min_log_probability:
        LOGGER.info(
            "Rejected low-confidence keyword match (log probability %s, minimum %.2f)",
            "unavailable" if log_probability is None else f"{log_probability:.2f}",
            min_log_probability,
        )
        return False

    if not cooldown.try_acquire():
        LOGGER.info("Suppressed duplicate keyword match")
        return True

    approximate_confidence = min(1.0, math.exp(log_probability))
    LOGGER.info(
        "Detected nimenhuuto (text match %.0f%%, ASR confidence approximately %.0f%%)",
        analysis.match.score * 100,
        approximate_confidence * 100,
    )
    message = _alert_text()
    if notifier is None:
        LOGGER.warning("DRY RUN detection:\n%s", message)
        return True

    try:
        notifier.send(message)
    except TelegramError:
        LOGGER.exception("Detection occurred, but the Telegram alert could not be queued")
    return True


def _run_file(
    path: Path,
    detector: WhisperKeywordDetector,
    notifier: TelegramNotifier | None,
    min_log_probability: float,
) -> int:
    if not path.is_file():
        LOGGER.error("Audio file does not exist: %s", path)
        return 2
    analysis = detector.analyze(path)
    LOGGER.info("Transcript: %s", analysis.transcript or "<no speech>")
    detected = _handle_detection(
        analysis,
        notifier=notifier,
        cooldown=Cooldown(0),
        min_log_probability=min_log_probability,
    )
    if not detected:
        LOGGER.info("Target phrase not detected with sufficient confidence")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv(dotenv_path=Path.cwd() / ".env")
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_devices:
        try:
            print(MicrophoneWindows.list_devices())
            return 0
        except RuntimeError as exc:
            parser.error(str(exc))

    try:
        settings = Settings.from_env()
        if not args.dry_run:
            settings.require_telegram()
    except ValueError as exc:
        parser.error(str(exc))

    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    telegram_notifier = None
    if not args.dry_run:
        telegram_notifier = TelegramNotifier(
            bot_token=settings.telegram_bot_token,
            chat_id=settings.telegram_chat_id,
        )

    try:
        detector = _make_detector(settings)
        if args.test_audio is not None:
            return _run_file(
                args.test_audio,
                detector,
                telegram_notifier,
                settings.min_log_probability,
            )

        device = _device_argument(args.device, settings.audio_device)
        microphone = MicrophoneWindows(
            sample_rate=settings.sample_rate,
            window_seconds=settings.window_seconds,
            stride_seconds=settings.stride_seconds,
            block_seconds=settings.block_seconds,
            device=device,
        )
        cooldown = Cooldown(settings.alert_cooldown_seconds)
        dispatcher_context = (
            TelegramDispatcher(telegram_notifier)
            if telegram_notifier is not None
            else nullcontext(None)
        )
        LOGGER.info(
            "Listening on device %r for %s; press Ctrl+C to stop",
            device if device is not None else "system default",
            ", ".join(repr(value) for value in settings.target_phrases),
        )
        with microphone, dispatcher_context as live_notifier:
            for audio in microphone.windows():
                level = dbfs(audio)
                if level < settings.min_audio_dbfs:
                    LOGGER.debug("Skipping quiet window at %.1f dBFS", level)
                    continue
                analysis = detector.analyze(audio)
                if analysis.transcript:
                    LOGGER.debug("Transcribed a %.1f dBFS audio window", level)
                _handle_detection(
                    analysis,
                    notifier=live_notifier,
                    cooldown=cooldown,
                    min_log_probability=settings.min_log_probability,
                )
    except KeyboardInterrupt:
        LOGGER.info("Stopped")
        return 0
    except (RuntimeError, OSError, ValueError) as exc:
        LOGGER.error("Fatal error: %s", exc)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
