from __future__ import annotations

import json
import logging
import queue
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from types import TracebackType
from typing import Any


LOGGER = logging.getLogger(__name__)
_SENTINEL = object()


class TelegramError(RuntimeError):
    pass


@dataclass(frozen=True)
class TelegramNotifier:
    bot_token: str
    chat_id: str
    timeout_seconds: float = 10.0
    attempts: int = 3

    def send(self, text: str) -> None:
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = json.dumps(
            {
                "chat_id": self.chat_id,
                "text": text,
                "disable_notification": False,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        last_error: Exception | None = None
        for attempt in range(1, self.attempts + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    body: dict[str, Any] = json.loads(response.read().decode("utf-8"))
                if not body.get("ok"):
                    description = body.get("description", "Telegram rejected the alert")
                    raise TelegramError(str(description))
                return
            except (OSError, ValueError, urllib.error.HTTPError, TelegramError) as exc:
                last_error = exc
                if attempt < self.attempts:
                    time.sleep(2 ** (attempt - 1))

        raise TelegramError(f"Could not send Telegram alert: {last_error}") from last_error


class TelegramDispatcher:
    """Send alerts off the audio loop and retain them during network outages."""

    def __init__(
        self,
        notifier: TelegramNotifier,
        *,
        retry_delay_seconds: float = 60.0,
        max_pending: int = 8,
    ) -> None:
        self.notifier = notifier
        self.retry_delay_seconds = retry_delay_seconds
        self._queue: queue.Queue[str | object] = queue.Queue(maxsize=max_pending)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> TelegramDispatcher:
        self._thread = threading.Thread(
            target=self._run,
            name="telegram-alerts",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self._stop.set()
        try:
            self._queue.put_nowait(_SENTINEL)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def send(self, text: str) -> None:
        if self._thread is None or not self._thread.is_alive():
            raise TelegramError("Telegram dispatcher is not running")
        try:
            self._queue.put_nowait(text)
        except queue.Full as exc:
            raise TelegramError("Telegram alert queue is full") from exc

    def _run(self) -> None:
        while not self._stop.is_set():
            item = self._queue.get()
            if item is _SENTINEL:
                return
            assert isinstance(item, str)

            while not self._stop.is_set():
                try:
                    self.notifier.send(item)
                except TelegramError as exc:
                    LOGGER.error(
                        "Telegram delivery failed; retrying in %.0f seconds: %s",
                        self.retry_delay_seconds,
                        exc,
                    )
                    if self._stop.wait(self.retry_delay_seconds):
                        return
                else:
                    LOGGER.info("Telegram alert sent")
                    break
