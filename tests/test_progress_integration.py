"""Integration tests for BLVPY-owned progress and backend verbosity."""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest

import blvpy.continuation as continuation
from blvpy import BilevelProblem, LowerProblem
from blvpy.continuation import SolveSettings
from blvpy.errors import InitializationError
from blvpy.result import BilevelResult, IterationRecord, Residuals

_ZERO_RESIDUALS = Residuals(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
_IPOPT_OPTIONS = {"hessian_approximation": "limited-memory", "tol": 1e-9}
_PREFIX = "(BLVPY)"


def _event_block(transcript: str, marker: str) -> str:
    lines = transcript.splitlines()
    start = next(index for index, line in enumerate(lines) if marker in line)
    stop = start + 1
    while stop < len(lines) and lines[stop].startswith(f"{_PREFIX}   "):
        stop += 1
    return " ".join(lines[start:stop])


def _quadratic_model(*, constrained: bool = False):
    x = cp.Variable(name="x", bounds=[-3.0, 3.0])
    y = cp.Variable(name="y")
    lower = LowerProblem(cp.Minimize(cp.square(y - x)), parameters=[x])
    constraints = [x + y >= 1.5] if constrained else []
    model = BilevelProblem(
        cp.Minimize(cp.square(x) + cp.square(y)),
        lower,
        outer_constraints=constraints,
    )
    return model, x, y


def _mock_initialization(model: BilevelProblem, x: cp.Variable, y: cp.Variable) -> None:
    lifted = model.lifted_problem
    y.value = float(x.value)
    lifted.primal.value = np.zeros(lifted.primal.size)
    lifted.slack.value = np.zeros(lifted.slack.size)
    lifted.dual.value = np.zeros(lifted.dual.size)


@pytest.mark.parametrize(
    ("verbose", "solver_verbose"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_verbosity_matrix_routes_progress_and_backend_flags(
    monkeypatch,
    capfd,
    verbose: bool,
    solver_verbose: bool,
) -> None:
    model, _, _ = _quadratic_model()
    observed: list[bool] = []

    def fake_run(current, settings, reporter, backend_verbose):
        assert current is model
        assert settings.verbose is verbose
        observed.append(backend_verbose)
        return BilevelResult(status="optimal")

    monkeypatch.setattr(continuation, "_solve_bilevel", fake_run)

    result = continuation.solve_bilevel(
        model,
        SolveSettings(verbose=verbose, solver_verbose=solver_verbose),
    )

    captured = capfd.readouterr()
    assert result.succeeded
    assert observed == [solver_verbose]
    assert captured.out == ""
    assert ("BLVPY" in captured.err) is verbose


@pytest.mark.parametrize(("name", "value"), [("verbose", 1), ("solver_verbose", "yes")])
def test_verbosity_flags_must_be_boolean(name: str, value) -> None:
    model, _, _ = _quadratic_model()
    kwargs = {name: value}

    with pytest.raises(ValueError, match=rf"{name} must be boolean"):
        continuation.solve_bilevel(model, SolveSettings(**kwargs))


def test_invalid_solver_verbosity_reports_failure_when_progress_is_enabled(capfd) -> None:
    model, _, _ = _quadratic_model()

    with pytest.raises(ValueError, match="solver_verbose must be boolean"):
        continuation.solve_bilevel(
            model,
            SolveSettings(verbose=True, solver_verbose="yes"),
        )

    transcript = capfd.readouterr().err
    assert "Summary" in transcript
    failure = _event_block(transcript, "Status:")
    assert "failed" in failure
    assert "error=ValueError: solver_verbose" in failure
    assert "must be boolean" in failure


def test_progress_reports_rejected_start_and_alternative_start_recovery(
    monkeypatch,
    capfd,
) -> None:
    model, x, y = _quadratic_model()
    statuses = [cp.OPTIMAL, cp.OPTIMAL, "solver_error", "solver_error", cp.OPTIMAL]
    objectives = [0.0, 1.0, None, None, 0.0]
    calls = 0

    monkeypatch.setattr(continuation, "_compile_probe", lambda *args, **kwargs: None)
    monkeypatch.setattr(continuation, "compute_residuals", lambda *args, **kwargs: _ZERO_RESIDUALS)
    monkeypatch.setattr(
        continuation,
        "_initialize_lower",
        lambda current, *args, **kwargs: _mock_initialization(current, x, y),
    )

    def fake_solve(current, epsilon, solver, options, solver_verbose):
        nonlocal calls
        current.lifted_problem.epsilon.value = epsilon
        status = statuses[calls]
        objective = objectives[calls]
        calls += 1
        return IterationRecord(
            epsilon=epsilon,
            status=status,
            objective=objective,
            residuals=_ZERO_RESIDUALS,
            solver_name=str(solver),
            message="synthetic rejection" if status == "solver_error" else None,
        )

    monkeypatch.setattr(continuation, "_solve_one", fake_solve)

    result = model.solve(
        epsilon_initial=1e-1,
        epsilon_target=1e-2,
        starts=3,
        seed=8,
        solver="MOCK_NLP",
        restoration=False,
        verbose=True,
    )

    transcript = capfd.readouterr().err
    assert result.succeeded
    assert calls == 5
    assert "rejected | solver_error" in _event_block(transcript, "Start 3/3:")
    first_attempt = _event_block(transcript, "Attempt 1 [scheduled]:")
    assert "rejected | eps=1.000e-02 | solver_error" in first_attempt
    alternative = _event_block(transcript, "Attempt 2 [alternate, start 2]:")
    assert "accepted | eps=1.000e-02 | optimal" in alternative
    assert transcript.index("Start 3/3") < transcript.index("Selected start")
    assert transcript.index("Attempt 1") < transcript.index("Attempt 2") < transcript.index("Summary")


def test_progress_reports_inserted_epsilon_retry_exhaustion_and_failed_result(
    monkeypatch,
    capfd,
) -> None:
    model, x, y = _quadratic_model()
    calls = 0

    monkeypatch.setattr(continuation, "_compile_probe", lambda *args, **kwargs: None)
    monkeypatch.setattr(continuation, "compute_residuals", lambda *args, **kwargs: _ZERO_RESIDUALS)
    monkeypatch.setattr(
        continuation,
        "_initialize_lower",
        lambda current, *args, **kwargs: _mock_initialization(current, x, y),
    )

    def fake_solve(current, epsilon, solver, options, solver_verbose):
        nonlocal calls
        current.lifted_problem.epsilon.value = epsilon
        calls += 1
        status = cp.OPTIMAL if calls == 1 else "solver_error"
        return IterationRecord(
            epsilon=epsilon,
            status=status,
            objective=0.0 if status == cp.OPTIMAL else None,
            residuals=_ZERO_RESIDUALS,
            message=None if status == cp.OPTIMAL else "synthetic continuation failure",
        )

    monkeypatch.setattr(continuation, "_solve_one", fake_solve)

    result = model.solve(
        epsilon_initial=1e-1,
        epsilon_target=1e-2,
        starts=1,
        solver="MOCK_NLP",
        restoration=False,
        max_retries=1,
        verbose=True,
    )

    transcript = capfd.readouterr().err
    assert result.status == "continuation_failed"
    assert "rejected | eps=1.000e-02" in _event_block(transcript, "Attempt 1 [scheduled]:")
    assert "Inserting epsilon=3.162e-02 before retrying target=1.000e-02" in transcript
    assert "rejected | eps=3.162e-02" in _event_block(transcript, "Attempt 2 [inserted]:")
    assert "Retry budget exhausted | retries=1" in transcript
    assert "Status: continuation_failed" in transcript
    assert "Progress: accepted=0 | attempted=2" in transcript


def test_progress_reports_terminal_initialization_exception(capfd) -> None:
    x = cp.Variable(name="x", bounds=[0.0, 1.0])
    y = cp.Variable(name="y")
    lower = LowerProblem(
        cp.Minimize(cp.square(y)),
        [y >= x + 1.0, y <= x],
        parameters=[x],
    )
    model = BilevelProblem(cp.Minimize(cp.square(x) + cp.square(y)), lower)

    with pytest.raises(InitializationError) as caught:
        model.solve(verbose=True)

    transcript = capfd.readouterr().err
    assert str(caught.value) == "Automatic initialization failed. Please initialize variables: x."
    assert "rejected | failed" in _event_block(transcript, "Start 1/1:")
    assert "Summary" in transcript
    failure = _event_block(transcript, "Status:")
    assert "failed" in failure
    assert "error=InitializationError:" in failure
    assert "Automatic initialization failed." in failure


@pytest.mark.ipopt
def test_progress_is_numerically_inert_and_reports_real_restoration(capfd) -> None:
    silent_model, silent_x, silent_y = _quadratic_model(constrained=True)
    silent = silent_model.solve(
        epsilon_initial=1e-2,
        epsilon_target=1e-4,
        solver_options=_IPOPT_OPTIONS,
        verbose=False,
    )
    silent_output = capfd.readouterr()

    verbose_model, verbose_x, verbose_y = _quadratic_model(constrained=True)
    reported = verbose_model.solve(
        epsilon_initial=1e-2,
        epsilon_target=1e-4,
        solver_options=_IPOPT_OPTIONS,
        verbose=True,
    )
    reported_output = capfd.readouterr()

    assert _PREFIX not in silent_output.err
    assert "Restoration | start=1/1" in reported_output.err
    assert "Initialization" in reported_output.err
    assert "Continuation" in reported_output.err
    assert "Summary" in reported_output.err
    assert reported_output.err.index("Initialization") < reported_output.err.index("Projection")
    assert reported_output.err.index("Projection") < reported_output.err.index("Starts: requested=1 | unique=1")
    assert silent.succeeded and reported.succeeded
    assert reported.status == silent.status
    assert reported.epsilon_history == silent.epsilon_history
    assert [record.status for record in reported.iterations] == [record.status for record in silent.iterations]
    assert [record.status for record in reported.starts] == [record.status for record in silent.starts]
    assert reported.objective == pytest.approx(silent.objective, abs=1e-8)
    np.testing.assert_allclose(
        [verbose_x.value, verbose_y.value],
        [silent_x.value, silent_y.value],
        atol=1e-8,
    )
