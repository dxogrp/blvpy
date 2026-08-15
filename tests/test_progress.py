"""Tests for BLVPY's private human-readable progress reporter."""

from __future__ import annotations

import logging

import blvpy.progress as progress
from blvpy.progress import ProgressReporter
from blvpy.result import BilevelResult, IterationRecord, Residuals, StartRecord


def _residuals(*, infeasible: bool = False) -> Residuals:
    violation = 2.5e-4 if infeasible else 2.5e-8
    return Residuals(
        primal_equality=violation,
        dual_equality=1e-9,
        recovery=0.0,
        upper_constraints=0.0,
        primal_cone=0.0,
        dual_cone=0.0,
        complementarity=7.5e-7,
        gap_violation=violation,
    )


def _problem(reporter: ProgressReporter) -> None:
    reporter.problem(
        upper_dimension=2,
        lower_dimension=3,
        canonical_variables=4,
        canonical_constraints=8,
        zero=1,
        nonnegative=2,
        soc=(2, 3),
        lower_solver="CLARABEL",
        nonlinear_solver="IPOPT",
        requested_starts=3,
        epsilon_initial=1e-1,
        epsilon_target=1e-6,
        contraction=0.1,
    )


def test_disabled_reporter_is_silent(capfd) -> None:
    reporter = ProgressReporter(enabled=False)

    _problem(reporter)
    reporter.initialization()
    reporter.starts(requested_starts=3, deduplicated_starts=2)
    reporter.failure(RuntimeError("not displayed"), elapsed=0.5)

    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_problem_transcript_uses_stderr_and_stable_plain_sections(capfd) -> None:
    reporter = ProgressReporter(enabled=True)

    _problem(reporter)

    captured = capfd.readouterr()
    assert captured.out == ""
    lines = captured.err.splitlines()
    assert lines[0] == "=" * 79
    assert lines[1].strip() == "BLVPY"
    assert lines[2].strip().startswith("v")
    assert len(lines[0]) == len(lines[1]) == len(lines[2]) == len(lines[3]) == 79
    assert "Problem" in lines[5]
    assert "Dimensions: upper=2 | lower=3" in captured.err
    assert "Cones: zero=1 | nonnegative=2 | soc=[2, 3]" in captured.err
    assert "Epsilon: initial=1.000e-01 | target=1.000e-06 | contraction=1.000e-01" in captured.err
    assert "\x1b" not in captured.err
    assert "\r" not in captured.err


def test_initialization_and_attempt_lines_are_one_based_and_complete(capfd) -> None:
    reporter = ProgressReporter(enabled=True)
    good = _residuals()
    bad = _residuals(infeasible=True)
    rejected_message = "line one\nline two " + "x" * 300

    _problem(reporter)
    reporter.initialization()
    reporter.starts(requested_starts=3, deduplicated_starts=2)
    reporter.projection(start_index=0, total_starts=2, before_violation=float("inf"))
    reporter.restoration(start_index=1, total_starts=2, before_violation=0.25)
    reporter.start(
        start_index=0,
        total_starts=2,
        record=StartRecord(index=0, status="optimal", objective=1.25, residuals=good),
        accepted=True,
        solve_time=0.25,
        num_iters=4,
    )
    reporter.start(
        start_index=1,
        total_starts=2,
        record=StartRecord(index=1, status="failed", residuals=bad, message=rejected_message),
        accepted=False,
    )
    reporter.selected_start(start_index=0, total_starts=2, objective=1.25)
    reporter.continuation()
    reporter.attempt(
        attempt_index=0,
        kind="scheduled",
        epsilon=1e-2,
        record=IterationRecord(
            epsilon=1e-2,
            status="optimal",
            objective=1.0,
            residuals=good,
            solve_time=0.125,
            num_iters=7,
        ),
        accepted=True,
    )
    reporter.attempt(
        attempt_index=1,
        kind="alternative-start",
        epsilon=1e-3,
        start_index=1,
        record=IterationRecord(
            epsilon=1e-3,
            status="solver_error",
            objective=None,
            residuals=bad,
            message="first line\nsecond line",
        ),
        accepted=False,
    )
    reporter.inserting_epsilon(epsilon=3.162e-3, target=1e-3)
    reporter.retry_exhausted(last_successful_epsilon=1e-2, max_retries=8)

    transcript = capfd.readouterr().err
    assert "Starts: requested=3 | unique=2" in transcript
    assert "Projection | start=1/2 | before_violation=inf" in transcript
    assert "Restoration | start=2/2 | before_violation=2.500e-01" in transcript
    assert (
        "Start 1/2 | accepted | status=optimal | objective=1.250e+00"
        " | max_feasibility=2.500e-08 | gap_violation=2.500e-08"
        " | complementarity=7.500e-07 | solve_time=2.500e-01s | iterations=4"
    ) in transcript
    assert "Start 2/2 | rejected | status=failed" in transcript
    assert "line one line two" in transcript
    assert "x" * 170 not in transcript
    assert "Selected start 1/2 | objective=1.250e+00" in transcript
    assert "Attempt 1 | scheduled | epsilon=1.000e-02 | accepted" in transcript
    assert "solve_time=1.250e-01s | iterations=7" in transcript
    assert "Attempt 2 | alternative-start | epsilon=1.000e-03 | start=2 | rejected" in transcript
    assert "message=first line second line" in transcript
    assert "Inserting epsilon=3.162e-03 before retrying target=1.000e-03" in transcript
    assert "Retry budget exhausted | retries=8 | last_successful_epsilon=1.000e-02" in transcript


def test_summary_and_failure_include_terminal_information(capfd) -> None:
    reporter = ProgressReporter(enabled=True)
    residuals = _residuals()
    final = IterationRecord(1e-6, "optimal", 0.125, residuals)
    result = BilevelResult(
        status="optimal",
        objective=0.125,
        iterations=(final,),
        final_iteration=final,
        message="local result",
    )

    _problem(reporter)
    reporter.summary(
        result=result,
        successful_starts=2,
        requested_starts=3,
        accepted_attempts=6,
        attempted_solves=8,
        elapsed=1.5,
    )

    transcript = capfd.readouterr().err
    assert transcript.count("=" * 79) == 2
    assert "Summary" in transcript
    assert "Status: optimal | objective=1.250e-01 | final_epsilon=1.000e-06 | elapsed=1.500e+00s" in transcript
    assert "Residuals: max_feasibility=2.500e-08" in transcript
    assert "gap_violation=2.500e-08 | complementarity=7.500e-07" in transcript
    assert "Progress: accepted=6 | attempted=8 | successful_starts=2/3" in transcript
    assert "Message: local result" in transcript
    assert "Local numerical solution; not a rigorous bilevel certificate." in transcript

    failed = ProgressReporter(enabled=True)
    failed.failure(ValueError("multi\nline"), elapsed=0.25)
    failure = capfd.readouterr().err
    assert failure.splitlines()[0] == "=" * 79
    assert failure.count("=" * 79) == 2
    assert "Status: failed | elapsed=2.500e-01s | error=ValueError: multi line" in failure


def test_unavailable_record_fields_are_omitted_and_nonfinite_values_are_explicit(capfd) -> None:
    reporter = ProgressReporter(enabled=True)
    reporter.initialization()
    reporter.starts(requested_starts=1, deduplicated_starts=1)
    reporter.start(
        start_index=0,
        total_starts=1,
        record=StartRecord(index=0, status="failed", message="no point"),
        accepted=False,
    )
    reporter.projection(start_index=0, total_starts=1, before_violation=float("-inf"))

    transcript = capfd.readouterr().err
    start_line = next(line for line in transcript.splitlines() if "Start 1/1" in line)
    assert "objective=" not in start_line
    assert "solve_time=" not in start_line
    assert "message=no point" in start_line
    assert "before_violation=-inf" in transcript


def test_progress_handler_installation_is_idempotent() -> None:
    logger = logging.getLogger("blvpy")
    progress._install_handler()
    progress._install_handler()

    handlers = [handler for handler in logger.handlers if getattr(handler, progress._HANDLER_MARKER, False)]
    assert len(handlers) == 1
    assert logger.propagate is False
    assert not logging.getLogger().handlers or all(handler not in logging.getLogger().handlers for handler in handlers)


def test_reporting_failure_never_escapes(monkeypatch) -> None:
    reporter = ProgressReporter(enabled=True)

    def fail_to_write(*args, **kwargs) -> None:
        raise RuntimeError("logging failed")

    monkeypatch.setattr(progress._LOGGER, "info", fail_to_write)

    reporter.initialization()
    reporter.starts(requested_starts=1, deduplicated_starts=1)
    reporter.failure(RuntimeError("original"), elapsed=0.0)


def test_displayed_messages_strip_ansi_and_control_sequences(capfd) -> None:
    reporter = ProgressReporter(enabled=True)

    reporter.failure(RuntimeError("\x1b[31mred\x1b[0m\x00text"), elapsed=0.0)

    transcript = capfd.readouterr().err
    assert "error=RuntimeError: red text" in transcript
    assert "\x1b" not in transcript
    assert "\x00" not in transcript
