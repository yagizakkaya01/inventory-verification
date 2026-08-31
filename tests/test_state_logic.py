"""Unit tests for the state layer — no torch / camera needed.

These are the logic the whole system's correctness hinges on, so they come
first (TDD-friendly: extend as the verdict rules get richer).
"""

from __future__ import annotations

from src.state.state_machine import State, StateMachine
from src.state.temporal_smoothing import SceneReading, TemporalSmoother
from src.state.verifier import Verdict, Verifier

EXPECTED = ["item_a", "item_b", "item_c"]


def r(*names: str) -> SceneReading:
    return SceneReading(sequence=tuple(names))


# --- Verifier -------------------------------------------------------------

def test_verifier_ok():
    assert Verifier(EXPECTED).check(r(*EXPECTED)).verdict is Verdict.OK


def test_verifier_missing():
    res = Verifier(EXPECTED).check(r("item_a", "item_b"))
    assert res.verdict is Verdict.MISSING
    assert "item_c" in res.detail


def test_verifier_wrong_order():
    res = Verifier(EXPECTED).check(r("item_b", "item_a", "item_c"))
    assert res.verdict is Verdict.WRONG_ORDER


def test_verifier_wrong_combination():
    res = Verifier(EXPECTED).check(r("item_a", "item_b", "item_b"))
    assert res.verdict is Verdict.WRONG_COMBINATION


def test_verifier_empty():
    assert Verifier(EXPECTED).check(None).verdict is Verdict.EMPTY


# --- TemporalSmoother ---------------------------------------------------------

def test_smoother_needs_full_window():
    s = TemporalSmoother(window=4, min_agree=3)
    assert s.update(r(*EXPECTED)) is None
    assert s.update(r(*EXPECTED)) is None
    assert s.update(r(*EXPECTED)) is None
    assert s.update(r(*EXPECTED)) == r(*EXPECTED)


def test_smoother_rejects_flicker():
    s = TemporalSmoother(window=5, min_agree=4)
    good = r(*EXPECTED)
    for _ in range(4):
        s.update(good)
    s.update(r("item_a"))          # single bad frame
    assert s.stable == good        # holds previous stable reading


def test_smoother_switches_when_change_is_sustained():
    s = TemporalSmoother(window=4, min_agree=3)
    for _ in range(4):
        s.update(r(*EXPECTED))
    new = r("item_a", "item_b")
    for _ in range(4):
        s.update(new)
    assert s.stable == new


# --- StateMachine -----------------------------------------------------------

def _res(ok: bool):
    return Verifier(EXPECTED).check(r(*EXPECTED) if ok else r("item_a"))


def test_fsm_confirms_before_transition():
    fsm = StateMachine(confirm_frames=3)
    assert fsm.update(_res(True)) is None      # INIT -> OK pending
    assert fsm.update(_res(True)) is None
    t = fsm.update(_res(True))
    assert t is not None and t.to is State.OK


def test_fsm_debounces_single_error():
    fsm = StateMachine(confirm_frames=3)
    for _ in range(3):
        fsm.update(_res(True))
    assert fsm.state is State.OK
    assert fsm.update(_res(False)) is None     # one bad frame, no transition
    assert fsm.update(_res(True)) is None
    assert fsm.state is State.OK


def test_fsm_transitions_to_error_when_sustained():
    fsm = StateMachine(confirm_frames=3)
    for _ in range(3):
        fsm.update(_res(True))
    t = None
    for _ in range(3):
        t = fsm.update(_res(False))
    assert t is not None and t.to is State.ERROR
    assert t.verdict is Verdict.MISSING
