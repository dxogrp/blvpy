"""Tests for BLVPY's private human-readable progress reporter."""

from __future__ import annotations

import logging

import blvpy.progress as progress
from blvpy.progress import ProgressReporter
from blvpy.result import BilevelResult, IterationRecord, Residuals, RunRecord

_PREFIX = "(BLVPY)"


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


def _problem(
    reporter: ProgressReporter,
    *,
    best_of: int | None = 3,
    requested_runs: int = 3,
) -> None:
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
        best_of=best_of,
        requested_runs=requested_runs,
        epsilon_initial=1e-1,
        epsilon_target=1e-6,
        contraction=0.1,
    )


def _event_block(transcript: str, marker: str) -> str:
    """Return one event row together with its wrapped/detail continuation rows."""

    lines = transcript.splitlines()
    start = next(index for index, line in enumerate(lines) if marker in line)
    stop = start + 1
    while stop < len(lines) and lines[stop].startswith(f"{_PREFIX}   "):
        stop += 1
    return " ".join(lines[start:stop])


def test_disabled_reporter_is_silent(capfd) -> None:
    reporter = ProgressReporter(enabled=False)

    _problem(reporter)
    reporter.initialization()
    reporter.run_begin(run_index=0, total_runs=3, randomized=True)
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
    assert any(line.strip() == "Problem" for line in lines)
    dimensions = _event_block(captured.err, "Dimensions:")
    assert "upper=2" in dimensions
    assert "lower=3" in dimensions
    assert "canonical_variables=4" in dimensions
    assert "canonical_constraints=8" in dimensions
    assert "Cones: zero=1 | nonnegative=2 | soc=[2, 3]" in captured.err
    assert "Search: mode=random | best_of=3" in captured.err
    assert "Epsilon: initial=1.000e-01 | target=1.000e-06 | contraction=1.000e-01" in captured.err
    assert "\x1b" not in captured.err
    assert "\r" not in captured.err
    assert all(len(line) <= 79 for line in lines)


def test_problem_transcript_distinguishes_deterministic_search(capfd) -> None:
    reporter = ProgressReporter(enabled=True)

    _problem(reporter, best_of=None, requested_runs=1)

    transcript = capfd.readouterr().err
    assert "Search: mode=deterministic | runs=1" in transcript
    assert "best_of=" not in transcript


def test_run_and_attempt_lines_are_one_based_and_complete(capfd) -> None:
    reporter = ProgressReporter(enabled=True)
    good = _residuals()
    bad = _residuals(infeasible=True)
    rejected_message = "line one\nline two " + "x" * 300

    _problem(reporter)
    reporter.initialization()
    reporter.run_begin(run_index=0, total_runs=2, randomized=True)
    reporter.projection(run_index=0, total_runs=2, before_violation=float("inf"))
    reporter.restoration(run_index=1, total_runs=2, before_violation=0.25)
    accepted_final = IterationRecord(
        epsilon=1e-6,
        status="optimal",
        objective=1.25,
        residuals=good,
        solve_time=0.25,
        num_iters=4,
    )
    reporter.run(
        run_index=0,
        total_runs=2,
        record=RunRecord(
            index=0,
            initial_values={},
            status="optimal",
            objective=1.25,
            iterations=(accepted_final,),
            final_iteration=accepted_final,
        ),
    )
    reporter.run(
        run_index=1,
        total_runs=2,
        record=RunRecord(
            index=1,
            initial_values={},
            status="failed",
            message=rejected_message,
        ),
    )
    reporter.selected_run(run_index=0, total_runs=2, objective=1.25)
    reporter.continuation()
    reporter.attempt(
        run_index=0,
        total_runs=2,
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
        run_index=1,
        total_runs=2,
        attempt_index=1,
        kind="inserted",
        epsilon=1e-3,
        record=IterationRecord(
            epsilon=1e-3,
            status="solver_error",
            objective=None,
            residuals=bad,
            message="first line\nsecond line",
        ),
        accepted=False,
    )
    reporter.inserting_epsilon(
        run_index=1,
        total_runs=2,
        epsilon=3.162e-3,
        target=1e-3,
    )
    reporter.retry_exhausted(
        run_index=1,
        total_runs=2,
        last_successful_epsilon=1e-2,
        max_retries=8,
    )

    transcript = capfd.readouterr().err
    assert "Run 1/2 | begin | mode=random" in transcript
    assert "Projection | run=1/2 | before_violation=inf" in transcript
    assert "Restoration | run=2/2 | before_violation=2.500e-01" in transcript
    accepted_run = _event_block(transcript, "Run 1/2:")
    assert "succeeded | status=optimal" in accepted_run
    assert "objective=1.250e+00" in accepted_run
    assert "feasibility=2.500e-08" in accepted_run
    assert "gap=2.500e-08" in accepted_run
    assert "complementarity=7.500e-07" in accepted_run
    assert "time=2.500e-01s" in accepted_run
    assert "iters=4" in accepted_run
    rejected_run = _event_block(transcript, "Run 2/2:")
    assert "failed | status=failed" in rejected_run
    assert "line one line two" in rejected_run
    assert "x" * 170 not in transcript
    assert "Selected run 1/2 | objective=1.250e+00" in transcript
    attempt = _event_block(transcript, "Run 1/2, attempt 1 [scheduled]:")
    assert "accepted | eps=1.000e-02" in attempt
    assert "status=optimal" in attempt
    assert "time=1.250e-01s" in attempt
    assert "iters=7" in attempt
    inserted = _event_block(transcript, "Run 2/2, attempt 2 [inserted]:")
    assert "rejected | eps=1.000e-03" in inserted
    assert "status=solver_error" in inserted
    assert "message=first line second line" in inserted
    assert "Inserting epsilon | run=2/2 | epsilon=3.162e-03" in transcript
    assert "target=1.000e-03" in transcript
    exhausted = _event_block(transcript, "Retry budget exhausted")
    assert "run=2/2" in exhausted
    assert "retries=8" in exhausted
    assert "last_successful_epsilon=1.000e-02" in exhausted
    assert all(len(line) <= 79 for line in transcript.splitlines())


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
        successful_runs=2,
        requested_runs=3,
        accepted_attempts=6,
        attempted_solves=8,
        elapsed=1.5,
    )

    transcript = capfd.readouterr().err
    assert transcript.count("=" * 79) == 2
    assert "Summary" in transcript
    status = _event_block(transcript, "Status:")
    assert "status=optimal" in status
    assert "objective=1.250e-01" in status
    assert "final_epsilon=1.000e-06" in status
    assert "elapsed=1.500e+00s" in status
    residuals_block = _event_block(transcript, "Residuals:")
    assert "max_feasibility=2.500e-08" in residuals_block
    assert "gap_violation=2.500e-08" in residuals_block
    assert "complementarity=7.500e-07" in residuals_block
    assert "Progress: accepted=6 | attempted=8 | successful_runs=2/3" in transcript
    assert "Message: local result" in transcript
    assert "Local numerical solution; not a rigorous bilevel certificate." in transcript

    failed = ProgressReporter(enabled=True)
    failed.failure(ValueError("multi\nline"), elapsed=0.25)
    failure = capfd.readouterr().err
    assert failure.splitlines()[0] == "=" * 79
    assert failure.count("=" * 79) == 2
    failed_status = _event_block(failure, "Status:")
    assert "status=failed" in failed_status
    assert "elapsed=2.500e-01s" in failed_status
    assert "error=ValueError: multi line" in failed_status


def test_unavailable_record_fields_are_omitted_and_nonfinite_values_are_explicit(capfd) -> None:
    reporter = ProgressReporter(enabled=True)
    reporter.initialization()
    reporter.run_begin(run_index=0, total_runs=1, randomized=False)
    reporter.run(
        run_index=0,
        total_runs=1,
        record=RunRecord(index=0, initial_values={}, status="failed", message="no point"),
    )
    reporter.projection(run_index=0, total_runs=1, before_violation=float("-inf"))

    transcript = capfd.readouterr().err
    run_block = _event_block(transcript, "Run 1/1:")
    assert "objective=" not in run_block
    assert "time=" not in run_block
    assert "iters=" not in run_block
    assert "message=no point" in run_block
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
    reporter.run_begin(run_index=0, total_runs=1, randomized=False)
    reporter.failure(RuntimeError("original"), elapsed=0.0)


def test_displayed_messages_strip_ansi_and_control_sequences(capfd) -> None:
    reporter = ProgressReporter(enabled=True)

    reporter.failure(RuntimeError("\x1b[31mred\x1b[0m\x00text"), elapsed=0.0)

    transcript = capfd.readouterr().err
    assert "error=RuntimeError: red text" in transcript
    assert "\x1b" not in transcript
    assert "\x00" not in transcript
