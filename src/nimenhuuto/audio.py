from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from types import TracebackType
from typing import Any

import numpy as np


LOGGER = logging.getLogger(__name__)


class MicrophoneWindows:
    """Capture audio continuously and yield the newest complete window."""

    def __init__(
        self,
        *,
        sample_rate: int,
        window_seconds: float,
        stride_seconds: float,
        block_seconds: float,
        device: int | str | None,
    ) -> None:
        self.sample_rate = sample_rate
        self.window_samples = round(window_seconds * sample_rate)
        self.stride_samples = round(stride_seconds * sample_rate)
        self.block_samples = round(block_seconds * sample_rate)
        self.device = device
        self._ring = np.zeros(self.window_samples, dtype=np.float32)
        self._write_index = 0
        self._total_samples = 0
        self._last_callback_at: float | None = None
        self._condition = threading.Condition()
        self._stream: Any = None

    @staticmethod
    def list_devices() -> str:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError("sounddevice is not installed; run 'pip install -e .'") from exc
        return str(sd.query_devices())

    def _callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        del frames, time_info
        if status:
            LOGGER.warning("Audio input status: %s", status)

        block = np.asarray(indata[:, 0], dtype=np.float32)
        captured_samples = len(block)
        with self._condition:
            if captured_samples >= self.window_samples:
                self._ring[:] = block[-self.window_samples :]
                self._write_index = 0
            else:
                end = self._write_index + captured_samples
                if end <= self.window_samples:
                    self._ring[self._write_index : end] = block
                else:
                    first_part = self.window_samples - self._write_index
                    self._ring[self._write_index :] = block[:first_part]
                    self._ring[: end - self.window_samples] = block[first_part:]
                self._write_index = end % self.window_samples

            self._total_samples += captured_samples
            self._last_callback_at = time.monotonic()
            self._condition.notify_all()

    def __enter__(self) -> MicrophoneWindows:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError("sounddevice is not installed; run 'pip install -e .'") from exc

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_samples,
            device=self.device,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _is_ready(self, last_yield_sample: int) -> bool:
        if self._total_samples < self.window_samples:
            return False
        return (
            last_yield_sample == 0
            or self._total_samples - last_yield_sample >= self.stride_samples
        )

    def _ensure_stream_is_alive(self) -> None:
        if self._stream is not None and not self._stream.active:
            raise RuntimeError("Microphone input stream stopped unexpectedly")
        if (
            self._last_callback_at is not None
            and time.monotonic() - self._last_callback_at > 5.0
        ):
            raise RuntimeError("Microphone stopped delivering audio")

    def windows(self) -> Iterator[np.ndarray]:
        last_yield_sample = 0

        while True:
            with self._condition:
                while not self._is_ready(last_yield_sample):
                    self._condition.wait(timeout=1.0)
                    self._ensure_stream_is_alive()

                current_sample = self._total_samples
                if last_yield_sample:
                    elapsed_strides = (
                        current_sample - last_yield_sample
                    ) // self.stride_samples
                    if elapsed_strides > 1:
                        LOGGER.warning(
                            "Inference skipped %d audio window(s); consider a smaller model",
                            elapsed_strides - 1,
                        )

                audio = np.concatenate(
                    (
                        self._ring[self._write_index :],
                        self._ring[: self._write_index],
                    )
                )
                last_yield_sample = current_sample

            yield audio


def dbfs(audio: np.ndarray) -> float:
    if audio.size == 0:
        return float("-inf")
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    if rms <= 0.0:
        return float("-inf")
    return 20.0 * np.log10(rms)
