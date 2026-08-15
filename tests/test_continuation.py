"""Solver-independent initialization and continuation tests."""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest

import blvpy.continuation as continuation
from blvpy import BilevelProblem, LowerProblem
from blvpy.continuation import compute_residuals
from blvpy.errors import InitializationError, SolverUnavailableError
from blvpy.result import IterationRecord, Residuals

_ZERO_RESIDUALS = Residuals(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _quadratic_bilevel(*, bounded: bool = True):
    attributes = {"bounds": [-2.0, 2.0]} if bounded else {}
    x = cp.Variable(name="x", **attributes)
    y = cp.Variable(name="y")
    lower = LowerProblem(cp.Minimize(cp.square(y - x)), parameters=[x])
    model = BilevelProblem(
        cp.Minimize(cp.square(x - 1.0) + cp.square(y + 1.0)),
        lower,
        outer_constraints=[x + y <= 3.0],
    )
    parameter = next(iter(model._parameter_links))
    return model, x, y, parameter


def _mock_compatible_state(
    model: BilevelProblem,
    upper: cp.Variable,
    lower: cp.Variable,
) -> None:
    lower.value = float(upper.value)
    lifted = model._lifted_problem
    lifted.primal.value = np.zeros(lifted.primal.size)
    lifted.slack.value = np.zeros(lifted.slack.size)
    lifted.dual.value = np.zeros(lifted.dual.size)


def test_deterministic_initialization_preserves_explicit_value() -> None:
    model, x, _, _ = _quadratic_bilevel()
    x.value = 0.75

    samples = continuation._generate_upper_initializations(
        model,
        None,
        np.random.default_rng(19),
    )
    other_seed = continuation._generate_upper_initializations(
        model,
        None,
        np.random.default_rng(91),
    )

    assert len(samples) == 1
    assert float(samples[0][x]) == pytest.approx(0.75)
    np.testing.assert_array_equal(samples[0][x], other_seed[0][x])


def test_deterministic_unbounded_upper_defaults_to_zero() -> None:
    model, x, _, _ = _quadratic_bilevel(bounded=False)

    automatic = continuation._generate_upper_initializations(
        model,
        None,
        np.random.default_rng(0),
    )
    assert len(automatic) == 1
    assert float(automatic[0][x]) == pytest.approx(0.0)


def test_deterministic_initialization_uses_midpoint_one_sided_interior_and_zero() -> None:
    x = cp.Variable(
        4,
        name="x",
        bounds=[
            np.array([-3.0, 2.0, -np.inf, -np.inf]),
            np.array([-1.0, np.inf, -4.0, np.inf]),
        ],
    )
    y = cp.Variable(name="y")
    lower = LowerProblem(cp.Minimize(cp.square(y) + cp.sum_squares(x)), parameters=[x])
    model = BilevelProblem(cp.Minimize(cp.sum_squares(x) + cp.square(y)), lower)

    samples = continuation._generate_upper_initializations(
        model,
        None,
        np.random.default_rng(4),
    )

    assert len(samples) == 1
    np.testing.assert_allclose(samples[0][x], np.array([-2.0, 3.0, -5.0, 0.0]))


def test_deterministic_initialization_respects_one_sided_variable_domain() -> None:
    x = cp.Variable(2, name="x", nonneg=True)
    y = cp.Variable(name="y")
    lower = LowerProblem(cp.Minimize(cp.square(y) + cp.sum_squares(x)), parameters=[x])
    model = BilevelProblem(cp.Minimize(cp.sum_squares(x) + cp.square(y)), lower)

    initializations = continuation._generate_upper_initializations(
        model,
        None,
        np.random.default_rng(1),
    )

    assert len(initializations) == 1
    np.testing.assert_array_equal(initializations[0][x], np.ones(2))


def test_best_of_sampling_precedence_and_exact_run_count() -> None:
    sampled = cp.Variable(name="sampled", bounds=[-2.0, 2.0])
    sampled.value = 0.75
    sampled.sample_bounds = (-0.5, 0.5)
    native = cp.Variable(name="native", bounds=[-2.0, -1.0])
    fixed = cp.Variable(name="fixed")
    fixed.value = 7.5
    source = cp.Variable(name="source")
    lower = LowerProblem(cp.Minimize(cp.square(source - sampled)), parameters=[sampled])
    model = BilevelProblem(
        cp.Minimize(cp.square(sampled) + cp.square(native) + cp.square(fixed) + cp.square(source)),
        lower,
    )

    samples = continuation._generate_upper_initializations(
        model,
        8,
        np.random.default_rng(18),
    )

    assert len(samples) == 8
    assert all(-0.5 <= float(sample[sampled]) <= 0.5 for sample in samples)
    assert any(float(sample[sampled]) != pytest.approx(0.75) for sample in samples)
    assert all(-2.0 <= float(sample[native]) <= -1.0 for sample in samples)
    assert [float(sample[fixed]) for sample in samples] == pytest.approx([7.5] * 8)


def test_best_of_does_not_deduplicate_fixed_explicit_values() -> None:
    model, x, _, _ = _quadratic_bilevel(bounded=False)
    x.value = -3.25

    samples = continuation._generate_upper_initializations(
        model,
        4,
        np.random.default_rng(9),
    )

    assert len(samples) == 4
    assert [float(sample[x]) for sample in samples] == pytest.approx([-3.25] * 4)


def test_explicit_best_of_one_uses_random_initialization() -> None:
    model, x, _, _ = _quadratic_bilevel()
    x.value = 0.75
    x.sample_bounds = (-1.0, 1.0)
    expected = np.random.default_rng(27).uniform(-1.0, 1.0)

    samples = continuation._generate_upper_initializations(
        model,
        1,
        np.random.default_rng(27),
    )

    assert len(samples) == 1
    assert float(samples[0][x]) == pytest.approx(expected)
    assert float(samples[0][x]) != pytest.approx(0.75)


def test_seeded_best_of_is_reproducible_without_using_global_rng() -> None:
    model, x, _, _ = _quadratic_bilevel()
    state = np.random.get_state()
    try:
        np.random.seed(812)
        expected_global_draw = np.random.random()
        np.random.seed(812)
        first = continuation._generate_upper_initializations(
            model,
            5,
            np.random.default_rng(19),
        )
        actual_global_draw = np.random.random()
        second = continuation._generate_upper_initializations(
            model,
            5,
            np.random.default_rng(19),
        )
    finally:
        np.random.set_state(state)

    assert actual_global_draw == pytest.approx(expected_global_draw)
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left[x], right[x])


def test_best_of_broadcasts_sampling_bounds_to_matrix_shape() -> None:
    x = cp.Variable((2, 2), name="x")
    x.sample_bounds = (-2.0, np.array([[0.0, 1.0], [2.0, 3.0]]))
    y = cp.Variable(name="y")
    lower = LowerProblem(cp.Minimize(cp.square(y) + cp.sum_squares(x)), parameters=[x])
    model = BilevelProblem(cp.Minimize(cp.sum_squares(x) + cp.square(y)), lower)

    samples = continuation._generate_upper_initializations(
        model,
        6,
        np.random.default_rng(41),
    )

    assert len(samples) == 6
    for sample in samples:
        assert sample[x].shape == (2, 2)
        assert np.all(sample[x] >= -2.0)
        assert np.all(sample[x] <= np.array([[0.0, 1.0], [2.0, 3.0]]))


@pytest.mark.parametrize(
    "sample_bounds",
    [
        (0.0, np.inf),
        (1.0, -1.0),
        (np.zeros(3), np.ones(3)),
        (0.0 + 1.0j, 1.0),
    ],
)
def test_best_of_rejects_invalid_sample_bounds(sample_bounds) -> None:
    model, x, _, _ = _quadratic_bilevel(bounded=False)
    x.sample_bounds = sample_bounds

    with pytest.raises(ValueError, match="sample_bounds.*x"):
        continuation._generate_upper_initializations(
            model,
            2,
            np.random.default_rng(0),
        )


def test_best_of_names_upper_variables_without_randomization_data() -> None:
    model, _, _, _ = _quadratic_bilevel(bounded=False)

    with pytest.raises(
        InitializationError,
        match=r"requires .*value.*sample_bounds.*finite.*x",
    ):
        continuation._generate_upper_initializations(
            model,
            2,
            np.random.default_rng(0),
        )


def test_sampling_bounds_are_not_copied_to_generated_lower_parameters() -> None:
    model, x, _, parameter = _quadratic_bilevel(bounded=False)
    x.sample_bounds = (-4.0, 4.0)

    assert getattr(parameter, "sample_bounds", None) is None


def test_deterministic_start_projects_onto_dcp_upper_constraints() -> None:
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")
    lower = LowerProblem(cp.Minimize(cp.square(y - x)), parameters=[x])
    model = BilevelProblem(
        cp.Minimize(cp.square(x) + cp.square(y)),
        lower,
        outer_constraints=[x >= 2.0],
    )
    sample = continuation._generate_upper_initializations(
        model,
        None,
        np.random.default_rng(0),
    )[0]

    projected = continuation._project_upper_start(model, sample, cp.CLARABEL, {}, False)

    assert float(projected[x]) == pytest.approx(2.0, abs=1e-7)


def test_failed_automatic_initialization_requests_named_values(monkeypatch) -> None:
    model, _, _, _ = _quadratic_bilevel(bounded=False)

    def fail_initialization(*args, **kwargs) -> None:
        raise InitializationError("synthetic initialization failure")

    monkeypatch.setattr(continuation, "_initialize_lower", fail_initialization)

    with pytest.raises(
        InitializationError,
        match=r"^Automatic initialization failed\. Please initialize variables: x\.",
    ):
        model.solve(solver="MOCK_NLP")


def test_failed_supplied_initialization_names_all_upper_variables(monkeypatch) -> None:
    model, x, _, _ = _quadratic_bilevel(bounded=False)
    x.value = 1.25
    monkeypatch.setattr(
        continuation,
        "_initialize_lower",
        lambda *args, **kwargs: (_ for _ in ()).throw(InitializationError("synthetic initialization failure")),
    )

    with pytest.raises(
        InitializationError,
        match=r"^Automatic initialization failed\. Please initialize variables: x\.",
    ):
        model.solve(solver="MOCK_NLP")


def test_default_solve_reports_actionable_missing_ipopt_error(monkeypatch) -> None:
    model, _, _, _ = _quadratic_bilevel()

    def unavailable_ipopt(*args, **kwargs):
        raise SolverUnavailableError(
            "IPOPT is not available. Install its native library, then reinstall "
            "BLVPY so its required cyipopt binding can be built."
        )

    monkeypatch.setattr(continuation, "solve_dnlp", unavailable_ipopt)

    with pytest.raises(SolverUnavailableError, match=r"IPOPT.*required cyipopt"):
        model.solve()


def test_solve_argument_validation_precedes_numerical_backends() -> None:
    model, _, _, _ = _quadratic_bilevel()

    with pytest.raises(ValueError, match="solver must name a CVXPY DNLP backend"):
        model.solve(solver=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="epsilon_target cannot exceed"):
        model.solve(epsilon_initial=1e-3, epsilon_target=1e-2)
    with pytest.raises(ValueError, match="contraction"):
        model.solve(contraction=1.0)


def test_compile_probe_evaluates_solver_neutral_first_order_oracles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cvxpy.reductions.solvers.nlp_solvers.nlp_solver import Oracles

    model, x, _, _ = _quadratic_bilevel()
    x.value = 0.25
    continuation._initialize_lower(model, cp.CLARABEL, {}, False)

    configuration: list[tuple[bool, bool]] = []
    evaluations: list[tuple[str, np.ndarray]] = []
    original_init = Oracles.__init__

    def recording_init(self, problem, verbose=True, use_hessian=True):
        configuration.append((verbose, use_hessian))
        original_init(self, problem, verbose=verbose, use_hessian=use_hessian)

    monkeypatch.setattr(Oracles, "__init__", recording_init)
    for name in ("objective", "constraints", "gradient", "jacobian"):
        original = getattr(Oracles, name)

        def recording_evaluation(self, point, *, _name=name, _original=original):
            evaluations.append((_name, np.array(point, copy=True)))
            return _original(self, point)

        monkeypatch.setattr(Oracles, name, recording_evaluation)

    continuation._compile_probe(model._lifted_problem)

    assert configuration == [(False, False)]
    assert [name for name, _ in evaluations] == ["objective", "constraints", "gradient", "jacobian"]
    for _, point in evaluations[1:]:
        np.testing.assert_array_equal(point, evaluations[0][1])


@pytest.mark.parametrize(
    "solver",
    [
        cp.IPOPT,
        cp.KNITRO,
        cp.UNO,
        cp.COPT,
        "knitro_ipm",
        "knitro_sqp",
        "knitro_alm",
        "uno_ipm",
        "uno_sqp",
    ],
)
def test_selected_nlp_solver_reaches_restoration_continuation_and_records(
    monkeypatch,
    solver: str,
) -> None:
    model, x, y, _ = _quadratic_bilevel()
    infeasible = Residuals(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    restoration_calls: list[tuple[str, dict[str, object], bool]] = []
    solve_calls: list[tuple[float, str, dict[str, object], bool]] = []
    options = {"backend_option": "preserved"}

    monkeypatch.setattr(continuation, "_compile_probe", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        continuation,
        "_initialize_lower",
        lambda current, *args, **kwargs: _mock_compatible_state(current, x, y),
    )
    monkeypatch.setattr(continuation, "compute_residuals", lambda *args, **kwargs: infeasible)

    def fake_restore(
        current,
        epsilon,
        selected_solver,
        selected_options,
        selected_verbose,
        tolerance,
    ) -> None:
        restoration_calls.append((selected_solver, dict(selected_options), selected_verbose))

    def fake_solve(current, epsilon, selected_solver, selected_options, selected_verbose):
        solve_calls.append((epsilon, selected_solver, dict(selected_options), selected_verbose))
        return IterationRecord(
            epsilon,
            cp.OPTIMAL,
            float(current.outer_objective.value),
            _ZERO_RESIDUALS,
            solver_name=str(selected_solver),
        )

    monkeypatch.setattr(continuation, "_restore_feasibility", fake_restore)
    monkeypatch.setattr(continuation, "_solve_one", fake_solve)

    result = model.solve(
        epsilon_initial=1e-1,
        epsilon_target=1e-2,
        solver=solver,
        solver_options=options,
        solver_verbose=True,
        verbose=False,
    )

    assert restoration_calls == [(solver, options, True)]
    assert [epsilon for epsilon, *_ in solve_calls] == pytest.approx([1e-1, 1e-2])
    assert all(call[1:] == (solver, options, True) for call in solve_calls)
    assert all(record.solver_name == str(solver) for record in result.iterations)
    assert options == {"backend_option": "preserved"}


@pytest.mark.parametrize("best_of", [0, -1, True, 1.5, "2"])
def test_best_of_requires_a_positive_integer(best_of) -> None:
    model, _, _, _ = _quadratic_bilevel()

    with pytest.raises(ValueError, match="best_of"):
        model.solve(best_of=best_of)


def test_public_best_of_one_uses_one_random_run(monkeypatch) -> None:
    model, x, y, _ = _quadratic_bilevel(bounded=False)
    x.value = 0.75
    x.sample_bounds = (-1.0, 1.0)
    expected = np.random.default_rng(27).uniform(-1.0, 1.0)

    monkeypatch.setattr(continuation, "_compile_probe", lambda *args, **kwargs: None)
    monkeypatch.setattr(continuation, "compute_residuals", lambda *args, **kwargs: _ZERO_RESIDUALS)
    monkeypatch.setattr(
        continuation,
        "_initialize_lower",
        lambda current, *args, **kwargs: _mock_compatible_state(current, x, y),
    )
    monkeypatch.setattr(
        continuation,
        "_solve_one",
        lambda current, epsilon, solver, options, solver_verbose: IterationRecord(
            epsilon,
            cp.OPTIMAL,
            float(current.outer_objective.value),
            _ZERO_RESIDUALS,
        ),
    )

    result = model.solve(
        epsilon_initial=1e-2,
        epsilon_target=1e-2,
        best_of=1,
        seed=27,
        solver="MOCK_NLP",
        restoration=False,
        verbose=False,
    )

    assert len(result.runs) == 1
    assert float(result.runs[0].initial_values[x]) == pytest.approx(expected)
    assert float(result.runs[0].initial_values[x]) != pytest.approx(0.75)


def test_equal_final_objectives_select_lowest_run_index(monkeypatch) -> None:
    x = cp.Variable(name="x", bounds=[-2.0, 2.0])
    y = cp.Variable(name="y")
    lower = LowerProblem(cp.Minimize(cp.square(y - x)), parameters=[x])
    model = BilevelProblem(cp.Minimize(0.0 * x + 0.0 * y), lower)

    monkeypatch.setattr(
        continuation,
        "_generate_upper_initializations",
        lambda *args, **kwargs: ({x: np.array(-1.0)}, {x: np.array(1.0)}),
    )
    monkeypatch.setattr(continuation, "_compile_probe", lambda *args, **kwargs: None)
    monkeypatch.setattr(continuation, "compute_residuals", lambda *args, **kwargs: _ZERO_RESIDUALS)
    monkeypatch.setattr(
        continuation,
        "_initialize_lower",
        lambda current, *args, **kwargs: _mock_compatible_state(current, x, y),
    )
    monkeypatch.setattr(
        continuation,
        "_solve_one",
        lambda current, epsilon, solver, options, solver_verbose: IterationRecord(
            epsilon,
            cp.OPTIMAL,
            0.0,
            _ZERO_RESIDUALS,
        ),
    )

    result = model.solve(
        epsilon_initial=1e-2,
        epsilon_target=1e-2,
        best_of=2,
        solver="MOCK_NLP",
        restoration=False,
        verbose=False,
    )

    assert result.all_objectives == pytest.approx((0.0, 0.0))
    assert result.selected_run_index == 0
    assert float(x.value) == pytest.approx(-1.0)


def test_best_of_completes_every_run_and_selects_by_target_objective(
    monkeypatch,
) -> None:
    x = cp.Variable(name="x", bounds=[-2.0, 2.0])
    y = cp.Variable(name="y")
    lower = LowerProblem(cp.Minimize(cp.square(y - x)), parameters=[x])
    model = BilevelProblem(cp.Minimize(cp.square(x - 2.0) + 0.0 * y), lower)
    calls: list[tuple[int, float]] = []

    monkeypatch.setattr(
        continuation,
        "_generate_upper_initializations",
        lambda *args, **kwargs: ({x: np.array(-1.0)}, {x: np.array(1.0)}),
    )
    monkeypatch.setattr(continuation, "_compile_probe", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        continuation,
        "compute_residuals",
        lambda *args, **kwargs: _ZERO_RESIDUALS,
    )
    monkeypatch.setattr(
        continuation,
        "_initialize_lower",
        lambda current, *args, **kwargs: _mock_compatible_state(current, x, y),
    )

    def fake_solve(current, epsilon, solver, options, solver_verbose):
        branch = -1 if float(x.value) < 0.0 else 1
        calls.append((branch, epsilon))
        initial_objective = 0.0 if branch < 0 else 10.0
        objective = initial_objective if np.isclose(epsilon, 1e-1) else float(current.outer_objective.value)
        return IterationRecord(
            epsilon,
            cp.OPTIMAL,
            objective,
            _ZERO_RESIDUALS,
            solver_name=str(solver),
        )

    monkeypatch.setattr(continuation, "_solve_one", fake_solve)

    result = model.solve(
        epsilon_initial=1e-1,
        epsilon_target=1e-2,
        best_of=2,
        solver="MOCK_NLP",
        restoration=False,
        verbose=False,
    )

    assert calls == pytest.approx(
        [(-1, 1e-1), (1, 1e-1), (-1, 1e-2), (1, 1e-2)],
    )
    assert len(result.runs) == 2
    assert all(run.epsilon_history == pytest.approx((1e-1, 1e-2)) for run in result.runs)
    assert result.runs[0].iterations[0].objective == pytest.approx(0.0)
    assert result.runs[1].iterations[0].objective == pytest.approx(10.0)
    assert result.runs[0].objective == pytest.approx(9.0)
    assert result.runs[1].objective == pytest.approx(1.0)
    assert result.selected_run_index == 1
    assert result.selected_run is result.runs[1]
    assert float(x.value) == pytest.approx(1.0)
    assert float(result.variable_values[x]) == pytest.approx(1.0)
    assert result.objective == pytest.approx(1.0)
    assert result.iterations == result.selected_run.iterations


def test_failed_best_of_run_does_not_contaminate_later_run(monkeypatch) -> None:
    model, x, y, parameter = _quadratic_bilevel()
    attempted: list[float] = []

    monkeypatch.setattr(
        continuation,
        "_generate_upper_initializations",
        lambda *args, **kwargs: ({x: np.array(-1.0)}, {x: np.array(1.0)}),
    )
    monkeypatch.setattr(continuation, "_compile_probe", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        continuation,
        "compute_residuals",
        lambda *args, **kwargs: _ZERO_RESIDUALS,
    )

    def initialize(current, *args, **kwargs) -> None:
        attempted.append(float(x.value))
        parameter.value = x.value
        if float(x.value) < 0.0:
            y.value = -99.0
            parameter.value = -99.0
            raise InitializationError("synthetic first-run failure")
        assert float(parameter.value) == pytest.approx(1.0)
        _mock_compatible_state(current, x, y)

    monkeypatch.setattr(continuation, "_initialize_lower", initialize)
    monkeypatch.setattr(
        continuation,
        "_solve_one",
        lambda current, epsilon, solver, options, solver_verbose: IterationRecord(
            epsilon,
            cp.OPTIMAL,
            float(current.outer_objective.value),
            _ZERO_RESIDUALS,
        ),
    )

    result = model.solve(
        epsilon_initial=1e-2,
        epsilon_target=1e-2,
        best_of=2,
        solver="MOCK_NLP",
        restoration=False,
        verbose=False,
    )

    assert attempted == pytest.approx([-1.0, 1.0])
    assert [run.status for run in result.runs] == ["initialization_failed", cp.OPTIMAL]
    assert result.runs[0].iterations == ()
    assert result.selected_run_index == 1
    assert float(x.value) == pytest.approx(1.0)
    assert float(parameter.value) == pytest.approx(1.0)


def test_every_random_initialization_is_projected_and_recorded(monkeypatch) -> None:
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")
    lower = LowerProblem(cp.Minimize(cp.square(y - x)), parameters=[x])
    model = BilevelProblem(
        cp.Minimize(cp.square(x) + cp.square(y)),
        lower,
        outer_constraints=[x >= 0.0],
    )

    monkeypatch.setattr(
        continuation,
        "_generate_upper_initializations",
        lambda *args, **kwargs: ({x: np.array(-1.0)}, {x: np.array(-2.0)}),
    )
    monkeypatch.setattr(continuation, "_compile_probe", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        continuation,
        "compute_residuals",
        lambda *args, **kwargs: _ZERO_RESIDUALS,
    )
    monkeypatch.setattr(
        continuation,
        "_initialize_lower",
        lambda current, *args, **kwargs: _mock_compatible_state(current, x, y),
    )
    monkeypatch.setattr(
        continuation,
        "_solve_one",
        lambda current, epsilon, solver, options, solver_verbose: IterationRecord(
            epsilon,
            cp.OPTIMAL,
            float(current.outer_objective.value),
            _ZERO_RESIDUALS,
        ),
    )

    result = model.solve(
        epsilon_initial=1e-2,
        epsilon_target=1e-2,
        best_of=2,
        solver="MOCK_NLP",
        restoration=False,
        verbose=False,
    )

    assert len(result.runs) == 2
    assert [float(run.initial_values[x]) for run in result.runs] == pytest.approx(
        [0.0, 0.0],
        abs=1e-7,
    )


def test_all_best_of_initialization_failures_are_aggregated(monkeypatch) -> None:
    model, x, _, _ = _quadratic_bilevel()
    monkeypatch.setattr(
        continuation,
        "_generate_upper_initializations",
        lambda *args, **kwargs: ({x: np.array(-1.0)}, {x: np.array(1.0)}),
    )
    monkeypatch.setattr(
        continuation,
        "_initialize_lower",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            InitializationError("synthetic initialization failure"),
        ),
    )

    with pytest.raises(
        InitializationError,
        match="All best-of runs failed at the initial epsilon",
    ) as caught:
        model.solve(best_of=2, solver="MOCK_NLP", verbose=False)

    notes = "\n".join(getattr(caught.value, "__notes__", ()))
    assert "run 1" in notes
    assert "run 2" in notes
    assert "synthetic initialization failure" in notes


def test_partial_best_of_selects_smallest_attained_epsilon(monkeypatch) -> None:
    model, x, y, _ = _quadratic_bilevel()
    calls: list[tuple[int, float]] = []

    monkeypatch.setattr(
        continuation,
        "_generate_upper_initializations",
        lambda *args, **kwargs: ({x: np.array(-1.0)}, {x: np.array(1.0)}),
    )
    monkeypatch.setattr(continuation, "_compile_probe", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        continuation,
        "compute_residuals",
        lambda *args, **kwargs: _ZERO_RESIDUALS,
    )
    monkeypatch.setattr(
        continuation,
        "_initialize_lower",
        lambda current, *args, **kwargs: _mock_compatible_state(current, x, y),
    )

    def fake_solve(current, epsilon, solver, options, solver_verbose):
        branch = -1 if float(x.value) < 0.0 else 1
        calls.append((branch, epsilon))
        accepted = np.isclose(epsilon, 1e-1) or (branch < 0 and np.isclose(epsilon, 1e-2))
        return IterationRecord(
            epsilon,
            cp.OPTIMAL if accepted else "solver_error",
            float(current.outer_objective.value) if accepted else None,
            _ZERO_RESIDUALS,
        )

    monkeypatch.setattr(continuation, "_solve_one", fake_solve)

    result = model.solve(
        epsilon_initial=1e-1,
        epsilon_target=1e-3,
        contraction=0.1,
        best_of=2,
        solver="MOCK_NLP",
        restoration=False,
        max_retries=0,
        verbose=False,
    )

    assert result.status == "continuation_failed"
    assert [branch for branch, _ in calls] == [-1, 1, -1, -1, 1]
    assert [epsilon for _, epsilon in calls] == pytest.approx(
        [1e-1, 1e-1, 1e-2, 1e-3, 1e-2],
    )
    assert [run.final_epsilon for run in result.runs] == pytest.approx([1e-2, 1e-1])
    assert result.selected_run_index == 0
    assert result.final_epsilon == pytest.approx(1e-2)
    assert float(x.value) == pytest.approx(-1.0)


def test_best_of_retry_schedule_is_local_to_each_run(monkeypatch) -> None:
    model, x, y, _ = _quadratic_bilevel()
    calls: list[tuple[int, float]] = []
    failed_negative_target = False

    monkeypatch.setattr(
        continuation,
        "_generate_upper_initializations",
        lambda *args, **kwargs: ({x: np.array(-1.0)}, {x: np.array(1.0)}),
    )
    monkeypatch.setattr(continuation, "_compile_probe", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        continuation,
        "compute_residuals",
        lambda *args, **kwargs: _ZERO_RESIDUALS,
    )
    monkeypatch.setattr(
        continuation,
        "_initialize_lower",
        lambda current, *args, **kwargs: _mock_compatible_state(current, x, y),
    )

    def fake_solve(current, epsilon, solver, options, solver_verbose):
        nonlocal failed_negative_target
        branch = -1 if float(x.value) < 0.0 else 1
        calls.append((branch, epsilon))
        fail = branch < 0 and np.isclose(epsilon, 1e-2) and not failed_negative_target
        if fail:
            failed_negative_target = True
        return IterationRecord(
            epsilon,
            "solver_error" if fail else cp.OPTIMAL,
            None if fail else float(current.outer_objective.value),
            _ZERO_RESIDUALS,
        )

    monkeypatch.setattr(continuation, "_solve_one", fake_solve)

    result = model.solve(
        epsilon_initial=1e-1,
        epsilon_target=1e-2,
        best_of=2,
        solver="MOCK_NLP",
        restoration=False,
        max_retries=2,
        verbose=False,
    )

    inserted = np.sqrt(1e-3)
    assert calls == pytest.approx(
        [
            (-1, 1e-1),
            (1, 1e-1),
            (-1, 1e-2),
            (-1, inserted),
            (-1, 1e-2),
            (1, 1e-2),
        ],
    )
    assert all(run.succeeded for run in result.runs)
    assert result.runs[0].attempted_epsilon_history == pytest.approx(
        (1e-1, 1e-2, inserted, 1e-2),
    )
    assert result.runs[1].attempted_epsilon_history == pytest.approx((1e-1, 1e-2))


def test_compute_residuals_matches_independent_canonical_calculation() -> None:
    model, x, y, parameter = _quadratic_bilevel()
    lifted = model._lifted_problem
    canonical = model.canonicalize()
    x.value = 0.4
    parameter.value = x.value
    data = canonical.apply_numeric({parameter: x.value})

    primal = cp.Variable(canonical.canonical_size)
    slack = cp.Variable(canonical.constraint_size)
    equality = data.A @ primal + slack == data.b
    lower = cp.Problem(
        cp.Minimize(data.c @ primal),
        [equality, *canonical.cone_layout.primal_constraints(slack)],
    )
    lower.solve(solver=cp.CLARABEL)
    assert lower.status in cp.settings.SOLUTION_PRESENT

    lifted.primal.value = primal.value
    lifted.slack.value = slack.value
    lifted.dual.value = equality.dual_value
    y.value = canonical.recover_numeric(primal.value)[y.id]
    epsilon = 1e-5
    actual = compute_residuals(model, epsilon)

    u = np.asarray(primal.value).reshape(-1)
    s = np.asarray(slack.value).reshape(-1)
    dual = np.asarray(equality.dual_value).reshape(-1)
    expected_primal = np.linalg.norm(data.A @ u + s - data.b)
    expected_dual = np.linalg.norm(data.A.T @ dual + data.c)
    expected_complementarity = float(s @ dual)
    assert actual.primal_equality == pytest.approx(expected_primal)
    assert actual.dual_equality == pytest.approx(expected_dual)
    assert actual.recovery == pytest.approx(0.0)
    assert actual.upper_constraints == pytest.approx(0.0)
    assert actual.primal_cone == pytest.approx(canonical.cone_layout.primal_distance(s))
    assert actual.dual_cone == pytest.approx(canonical.cone_layout.dual_distance(dual))
    assert actual.complementarity == pytest.approx(expected_complementarity)
    assert actual.gap_violation == pytest.approx(max(expected_complementarity - epsilon, 0.0))


def test_fixed_upper_initialization_recovers_direct_lower_solution() -> None:
    x = cp.Variable(name="x", bounds=[-2.0, 2.0])
    y = cp.Variable(2, name="y")
    lower = LowerProblem(cp.Minimize(cp.sum_squares(y - cp.hstack([x, -x]))), parameters=[x])
    model = BilevelProblem(cp.Minimize(cp.square(x) + cp.sum_squares(y)), lower)
    parameter = next(iter(model._parameter_links))
    x.value = 0.65

    continuation._initialize_lower(model, cp.CLARABEL, {}, False)
    initialized = np.array(y.value, copy=True)

    parameter.value = x.value
    model._cvxpy_lower_problem.solve(solver=cp.CLARABEL)
    assert model._cvxpy_lower_problem.status in cp.settings.SOLUTION_PRESENT
    np.testing.assert_allclose(initialized, y.value, atol=2e-5)


def test_compute_residuals_reports_each_independent_failure() -> None:
    model, x, y, parameter = _quadratic_bilevel()
    lifted = model._lifted_problem
    canonical = model.canonicalize()
    x.value = 2.0
    parameter.value = x.value
    lifted.primal.value = np.zeros(canonical.canonical_size)
    lifted.slack.value = -np.ones(canonical.constraint_size)
    lifted.dual.value = np.ones(canonical.constraint_size)
    y.value = 10.0

    residuals = compute_residuals(model, epsilon=0.0)

    assert residuals.primal_equality > 0
    assert residuals.dual_equality > 0
    assert residuals.recovery > 0
    assert residuals.upper_constraints > 0
    assert residuals.primal_cone > 0
    assert residuals.complementarity < 0
    assert residuals.gap_violation == 0


def test_failed_step_inserts_intermediate_epsilon_then_retries(monkeypatch) -> None:
    model, x, y, _ = _quadratic_bilevel()
    calls: list[float] = []
    failed_target_once = False
    zero = Residuals(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    monkeypatch.setattr(continuation, "_compile_probe", lambda *args, **kwargs: None)

    def fake_initialize(current, *args, **kwargs) -> None:
        lifted = current._lifted_problem
        y.value = float(x.value)
        lifted.primal.value = np.zeros(lifted.primal.size)
        lifted.slack.value = np.zeros(lifted.slack.size)
        lifted.dual.value = np.zeros(lifted.dual.size)

    def fake_solve(current, epsilon, solver, options, verbose):
        nonlocal failed_target_once
        calls.append(epsilon)
        if epsilon == pytest.approx(1e-2) and not failed_target_once:
            failed_target_once = True
            return IterationRecord(epsilon, "solver_error", None, zero, solver_name=str(solver))
        current._lifted_problem.epsilon.value = epsilon
        return IterationRecord(
            epsilon,
            cp.OPTIMAL,
            float(current.outer_objective.value),
            zero,
            solver_name=str(solver),
        )

    monkeypatch.setattr(continuation, "_initialize_lower", fake_initialize)
    monkeypatch.setattr(continuation, "_solve_one", fake_solve)
    monkeypatch.setattr(continuation, "compute_residuals", lambda *args, **kwargs: zero)

    result = model.solve(
        epsilon_initial=1e-1,
        epsilon_target=1e-2,
        seed=7,
        solver="MOCK_NLP",
        restoration=False,
    )

    assert result.status == cp.OPTIMAL
    # The original failed target is recorded, then an intermediate value is
    # inserted and the target is retried. Floating contraction may also cause
    # the pre-existing schedule to contain a value numerically equal to target.
    assert calls[0] == pytest.approx(1e-1)
    assert calls[1] == pytest.approx(1e-2)
    assert any(value == pytest.approx(np.sqrt(1e-3)) for value in calls)
    assert calls[-1] == pytest.approx(1e-2)
    assert len(calls) == 4
    assert result.epsilon_history == pytest.approx((1e-1, np.sqrt(1e-3), 1e-2))
    assert result.attempted_epsilon_history == pytest.approx(tuple(calls))
    assert result.final_epsilon == pytest.approx(1e-2)
    assert result.runs[0].status == cp.OPTIMAL


def test_retry_budget_stops_repeated_bisection_and_returns_consistent_point(
    monkeypatch,
) -> None:
    model, x, y, _ = _quadratic_bilevel()
    zero = Residuals(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    calls: list[float] = []

    monkeypatch.setattr(continuation, "_compile_probe", lambda *args, **kwargs: None)
    monkeypatch.setattr(continuation, "compute_residuals", lambda *args, **kwargs: zero)

    def fake_initialize(current, *args, **kwargs) -> None:
        lifted = current._lifted_problem
        y.value = float(x.value)
        lifted.primal.value = np.zeros(lifted.primal.size)
        lifted.slack.value = np.zeros(lifted.slack.size)
        lifted.dual.value = np.zeros(lifted.dual.size)

    def fake_solve(current, epsilon, solver, options, verbose):
        calls.append(epsilon)
        status = cp.OPTIMAL if epsilon > 2e-2 else "solver_error"
        current._lifted_problem.epsilon.value = epsilon
        return IterationRecord(epsilon, status, 0.0 if status == cp.OPTIMAL else None, zero)

    monkeypatch.setattr(continuation, "_initialize_lower", fake_initialize)
    monkeypatch.setattr(continuation, "_solve_one", fake_solve)

    result = model.solve(
        epsilon_initial=1e-1,
        epsilon_target=1e-2,
        solver="MOCK_NLP",
        restoration=False,
        max_retries=2,
    )

    assert result.status == "continuation_failed"
    assert len(calls) == 5
    assert result.final_epsilon == pytest.approx(np.sqrt(1e-3))
    assert result.residuals == zero
