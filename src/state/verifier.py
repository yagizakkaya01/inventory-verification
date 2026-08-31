"""Compare a stable scene reading against the expected configuration."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum

from src.state.temporal_smoothing import SceneReading


class Verdict(str, Enum):
    OK = "OK"
    MISSING = "MISSING"                    # expected item(s) absent
    WRONG_ORDER = "WRONG_ORDER"            # right items, wrong left-to-right order
    WRONG_COMBINATION = "WRONG_COMBINATION"  # unexpected / extra items present
    EMPTY = "EMPTY"                        # nothing detected


@dataclass
class VerificationResult:
    verdict: Verdict
    expected: tuple[str, ...]
    observed: tuple[str, ...]
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.verdict is Verdict.OK


class Verifier:
    def __init__(self, expected_sequence: list[str] | tuple[str, ...]) -> None:
        self.expected = tuple(expected_sequence)
        self._expected_counts = Counter(self.expected)

    def check(self, reading: SceneReading | None) -> VerificationResult:
        observed = tuple(reading.sequence) if reading else ()

        if not observed:
            return VerificationResult(Verdict.EMPTY, self.expected, observed,
                                      "no objects detected")

        obs_counts = Counter(observed)

        if obs_counts == self._expected_counts:
            if observed == self.expected:
                return VerificationResult(Verdict.OK, self.expected, observed)
            return VerificationResult(
                Verdict.WRONG_ORDER, self.expected, observed,
                f"expected order {self.expected}, saw {observed}",
            )

        missing = list((self._expected_counts - obs_counts).elements())
        extra = list((obs_counts - self._expected_counts).elements())

        if extra:
            return VerificationResult(
                Verdict.WRONG_COMBINATION, self.expected, observed,
                f"unexpected: {extra}" + (f", missing: {missing}" if missing else ""),
            )
        return VerificationResult(
            Verdict.MISSING, self.expected, observed, f"missing: {missing}",
        )
