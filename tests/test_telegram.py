import threading
import unittest
from typing import Any, cast

from nimenhuuto.telegram import TelegramDispatcher, TelegramError


class _FlakyNotifier:
    def __init__(self) -> None:
        self.calls = 0
        self.delivered = threading.Event()

    def send(self, text: str) -> None:
        self.calls += 1
        if self.calls == 1:
            raise TelegramError("temporary failure")
        if text == "alert":
            self.delivered.set()


class TelegramDispatcherTests(unittest.TestCase):
    def test_retries_delivery_in_background(self) -> None:
        notifier = _FlakyNotifier()
        with TelegramDispatcher(
            cast(Any, notifier),
            retry_delay_seconds=0.01,
        ) as dispatcher:
            dispatcher.send("alert")
            self.assertTrue(notifier.delivered.wait(timeout=1.0))

        self.assertEqual(notifier.calls, 2)


if __name__ == "__main__":
    unittest.main()
