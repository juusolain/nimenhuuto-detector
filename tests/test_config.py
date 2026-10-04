import os
import unittest
from unittest.mock import patch

from nimenhuuto.config import Settings


_RELEVANT_ENVIRONMENT = (
    "TARGET_PHRASES",
    "MATCH_THRESHOLD",
    "MIN_LOG_PROBABILITY",
    "SAMPLE_RATE",
    "WINDOW_SECONDS",
    "STRIDE_SECONDS",
    "BLOCK_SECONDS",
    "AUDIO_DEVICE",
)


class SettingsTests(unittest.TestCase):
    def test_default_settings(self) -> None:
        environment = {key: os.environ.pop(key, None) for key in _RELEVANT_ENVIRONMENT}
        try:
            settings = Settings.from_env()
        finally:
            for key, value in environment.items():
                if value is not None:
                    os.environ[key] = value

        self.assertEqual(settings.target_phrases, ("nimenhuuto", "nimen huuto"))
        self.assertEqual(settings.match_threshold, 0.86)
        self.assertIsNone(settings.audio_device)

    def test_non_finite_number_is_rejected(self) -> None:
        with (
            patch.dict(os.environ, {"WINDOW_SECONDS": "nan"}),
            self.assertRaisesRegex(ValueError, "must be finite"),
        ):
            Settings.from_env()

    def test_non_whisper_sample_rate_is_rejected(self) -> None:
        with (
            patch.dict(os.environ, {"SAMPLE_RATE": "48000"}),
            self.assertRaisesRegex(ValueError, "must be 16000"),
        ):
            Settings.from_env()

    def test_invalid_stride_is_rejected(self) -> None:
        with (
            patch.dict(
                os.environ,
                {"WINDOW_SECONDS": "2", "STRIDE_SECONDS": "3"},
            ),
            self.assertRaisesRegex(ValueError, "cannot exceed"),
        ):
            Settings.from_env()


if __name__ == "__main__":
    unittest.main()
