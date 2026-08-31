"""State machine over verifier verdicts.

Consumes the (already temporally smoothed) verdict each frame and emits an event
only on a *confirmed* transition, so downstream consumers (alarm, logger, UI)
see edges, not every frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from src.state.verifier import Verdict, VerificationResult


class State(str, Enum):
    INIT = "INIT"
    OK = "OK"
    ERROR = "ERROR"


@dataclass
class Transition:
    frm: State
    to: State
    verdict: Verdict
    detail: str


@dataclass
class StateMachine:
    """`confirm_frames` consecutive agreeing verdicts are needed to change state.

    This is a second debounce on top of TemporalSmoother — cheap insurance that
    a single smoother blip cannot toggle the reported state.
    """

    confirm_frames: int = 3
    state: State = State.INIT
    _pending: State | None = field(default=None, repr=False)
    _pending_count: int = field(default=0, repr=False)

    def update(self, result: VerificationResult) -> Transition | None:
        target = State.OK if result.ok else State.ERROR

        if target == self.state:
            self._pending = None
            self._pending_count = 0
            return None

        if target == self._pending:
            self._pending_count += 1
        else:
            self._pending = target
            self._pending_count = 1

        if self._pending_count >= self.confirm_frames:
            frm, self.state = self.state, target
            self._pending = None
            self._pending_count = 0
            return Transition(frm, self.state, result.verdict, result.detail)
        return None
