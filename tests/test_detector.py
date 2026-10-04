import unittest

from nimenhuuto.matching import Cooldown, PhraseMatcher, normalize_words


class PhraseMatcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.matcher = PhraseMatcher(("nimenhuuto", "nimen huuto"), threshold=0.86)

    def test_normalize_words_handles_case_and_punctuation(self) -> None:
        self.assertEqual(
            normalize_words("  NIMEN—HUUTO! "),
            ("nimen", "huuto"),
        )

    def test_accepts_likely_transcriptions(self) -> None:
        transcripts = (
            "Nimenhuuto!",
            "Nimen huuto alkaa nyt.",
            "Seuraavaksi nimen-huuto.",
            "Nimen huudo",
        )
        for transcript in transcripts:
            with self.subTest(transcript=transcript):
                match = self.matcher.match(transcript)
                self.assertIsNotNone(match)
                assert match is not None
                self.assertGreaterEqual(match.score, 0.86)

    def test_rejects_other_speech(self) -> None:
        transcripts = (
            "Nyt mennään ulos.",
            "Jono liikkuu pian.",
            "Seuraava, olkaa hyvä.",
            "Tämä on aivan tavallinen kuulutus.",
            "Minun huutoni kuuluu.",
            "",
        )
        for transcript in transcripts:
            with self.subTest(transcript=transcript):
                self.assertIsNone(self.matcher.match(transcript))


class CooldownTests(unittest.TestCase):
    def test_suppresses_repeated_windows(self) -> None:
        cooldown = Cooldown(600)

        self.assertTrue(cooldown.try_acquire(now=1000))
        self.assertFalse(cooldown.try_acquire(now=1001))
        self.assertFalse(cooldown.try_acquire(now=1599))
        self.assertTrue(cooldown.try_acquire(now=1600))


if __name__ == "__main__":
    unittest.main()
