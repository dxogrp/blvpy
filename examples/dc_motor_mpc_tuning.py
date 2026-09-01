import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Learning MPC Weights for DC Motor Speed Control

    A model predictive controller balances speed tracking against large or
    rapidly changing motor-voltage commands. Here, the upper problem learns
    two control-cost weights from an expert response, while the lower problem
    computes the corresponding constrained motor trajectory.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import cvxpy as cp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from blvpy import BilevelProblem, LowerProblem

    plt.style.use(Path(__file__).resolve().parent / "zhlatex.mplstyle")
    return BilevelProblem, LowerProblem, Path, cp, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bilevel formulation

    Let $x(t)=(\omega(t),i(t))$ contain shaft speed and armature current, and let
    $u(t)$ be the applied voltage. The motor dynamics are discretized as
    $x(t+1)=Ax(t)+Bu(t)$. For fixed voltage-effort and voltage-slew weights
    $\delta,\eta>0$, one MPC update solves

    \[
    \begin{array}{ll}
    S(\delta,\eta)=\mathop{\mathrm{argmin}}_{x,u}
      & \displaystyle
        \sum_{t=1}^{N}(\omega(t)-1)^2
        +0.04\sum_{t=1}^{N}(i(t)-0.1)^2 \\
      & \displaystyle\quad
        +\delta\sum_{t=0}^{N-1}(u(t)-0.2)^2
        +\eta\sum_{t=0}^{N-1}(u(t)-u(t-1))^2
        +5(\omega(N)-1)^2 \\
    \mathop{\mathrm{subject\ to}}
      & x(0)=(0,0),\qquad x(t+1)=Ax(t)+Bu(t),\\
      & |u(t)|\leq2,\qquad |i(t)|\leq0.5,\\
      & -0.05\leq\omega(t)\leq1.25,
    \end{array}
    \]

    with previous input $u(-1)=0$. Given expert samples
    $(\hat{x},\hat{u})$, the upper problem is

    \[
    \begin{array}{ll}
    \mathop{\mathrm{minimize}}_{\delta,\eta,x,u}
      & \lVert x-\hat{x}\rVert_F^2
        +0.1\lVert u-\hat{u}\rVert_2^2\\
    \mathop{\mathrm{subject\ to}}
      & 0.01\leq\delta\leq1,\\
      & 0.01\leq\eta\leq1,\\
      & (x,u)\in S(\delta,\eta).
    \end{array}
    \]
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Synthetic instance

    We use a 24-step horizon with sample time $0.05$, target equilibrium
    $(\omega,i,u)=(1,0.1,0.2)$, normalized motor constants, and operating
    limits. The exact synthetic expert response uses
    hidden weights $(\delta^{\rm E},\eta^{\rm E})=(0.035,0.16)$.
    """)
    return


@app.cell
def _(np):
    J = 0.01
    b = 0.01
    K_t = 0.1
    K_e = 0.1
    L = 0.5
    R = 1.0
    sample_time = 0.05
    A = np.eye(2) + sample_time * np.array(
        [
            [-b / J, K_t / J],
            [-K_e / L, -R / L],
        ]
    )
    B = sample_time * np.array([0.0, 1.0 / L])

    N = 24
    x_initial = np.zeros(2)
    u_previous = 0.0
    x_reference = np.array([1.0, 0.1])
    u_reference = 0.2
    u_max = 2.0
    i_max = 0.5
    omega_min = -0.05
    omega_max = 1.25
    current_weight = 0.04
    terminal_weight = 5.0
    expert_delta = 0.035
    expert_eta = 0.16
    baseline_delta = 0.5
    baseline_eta = 0.5
    delta_bounds = (0.01, 1.0)
    eta_bounds = (0.01, 1.0)

    assert np.allclose(x_reference, A @ x_reference + B * u_reference)
    return (
        A,
        B,
        J,
        N,
        baseline_delta,
        baseline_eta,
        current_weight,
        delta_bounds,
        eta_bounds,
        expert_delta,
        expert_eta,
        i_max,
        omega_max,
        omega_min,
        sample_time,
        terminal_weight,
        u_max,
        u_previous,
        u_reference,
        x_initial,
        x_reference,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Expert and untuned responses

    Independent CVXPY solves generate the expert trajectory and the untuned
    baseline by fixing their MPC weights.
    """)
    return


@app.cell
def _(
    A,
    B,
    N,
    cp,
    current_weight,
    expert_delta,
    expert_eta,
    i_max,
    np,
    omega_max,
    omega_min,
    terminal_weight,
    u_max,
    u_previous,
    u_reference,
    x_initial,
    x_reference,
):
    def solve_fixed_mpc(fixed_delta, fixed_eta, x_initial_mpc, u_previous_mpc):
        _x = cp.Variable((2, N + 1))
        _u = cp.Variable(N)
        _u_change = cp.diff(cp.hstack([u_previous_mpc, _u]))
        _cost = (
            cp.sum_squares(_x[0, 1:] - x_reference[0])
            + current_weight * cp.sum_squares(_x[1, 1:] - x_reference[1])
            + fixed_delta * cp.sum_squares(_u - u_reference)
            + fixed_eta * cp.sum_squares(_u_change)
            + terminal_weight * cp.square(_x[0, -1] - x_reference[0])
        )
        _x_next = A @ _x[:, :-1] + cp.outer(B, _u)
        _problem = cp.Problem(
            cp.Minimize(_cost),
            [
                _x[:, 0] == x_initial_mpc,
                _x[:, 1:] == _x_next,
                cp.abs(_u) <= u_max,
                cp.abs(_x[1, :]) <= i_max,
                _x[0, :] >= omega_min,
                _x[0, :] <= omega_max,
            ],
        )
        _problem.solve(
            solver=cp.CLARABEL,
            tol_gap_abs=1e-10,
            tol_gap_rel=1e-10,
            tol_feas=1e-10,
        )
        assert _problem.status == cp.OPTIMAL
        return (
            np.asarray(_x.value, dtype=float),
            np.asarray(_u.value, dtype=float),
            float(_problem.value),
        )

    x_hat, u_hat, expert_lower_objective = solve_fixed_mpc(
        fixed_delta=expert_delta,
        fixed_eta=expert_eta,
        x_initial_mpc=x_initial,
        u_previous_mpc=u_previous,
    )
    return expert_lower_objective, solve_fixed_mpc, u_hat, x_hat


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Specify and solve the bilevel model

    `delta` and `eta` are upper-level variables and parameters of the lower MPC
    problem. We initialize them and the trajectory from the feasible untuned
    baseline.
    """)
    return


@app.cell
def _(
    A,
    B,
    BilevelProblem,
    LowerProblem,
    N,
    baseline_delta,
    baseline_eta,
    cp,
    current_weight,
    delta_bounds,
    eta_bounds,
    i_max,
    omega_max,
    omega_min,
    solve_fixed_mpc,
    terminal_weight,
    u_hat,
    u_max,
    u_previous,
    u_reference,
    x_hat,
    x_initial,
    x_reference,
):
    x_baseline, u_baseline, baseline_lower_objective = solve_fixed_mpc(
        baseline_delta,
        baseline_eta,
        x_initial,
        u_previous,
    )

    delta = cp.Variable(
        nonneg=True,
        bounds=[delta_bounds[0], delta_bounds[1]],
        name="delta",
    )
    eta = cp.Variable(
        nonneg=True,
        bounds=[eta_bounds[0], eta_bounds[1]],
        name="eta",
    )
    x = cp.Variable((2, N + 1), name="x")
    u = cp.Variable(N, name="u")

    delta.value = baseline_delta
    eta.value = baseline_eta
    x.value = x_baseline
    u.value = u_baseline

    u_change = cp.diff(cp.hstack([u_previous, u]))
    mpc_objective = (
        cp.sum_squares(x[0, 1:] - x_reference[0])
        + current_weight * cp.sum_squares(x[1, 1:] - x_reference[1])
        + delta * cp.sum_squares(u - u_reference)
        + eta * cp.sum_squares(u_change)
        + terminal_weight * cp.square(x[0, -1] - x_reference[0])
    )
    x_next = A @ x[:, :-1] + cp.outer(B, u)
    controller = LowerProblem(
        cp.Minimize(mpc_objective),
        [
            x[:, 0] == x_initial,
            x[:, 1:] == x_next,
            cp.abs(u) <= u_max,
            cp.abs(x[1, :]) <= i_max,
            x[0, :] >= omega_min,
            x[0, :] <= omega_max,
        ],
        parameters=[delta, eta],
    )
    imitation_loss = cp.sum_squares(x - x_hat) + 0.1 * cp.sum_squares(u - u_hat)
    problem = BilevelProblem(cp.Minimize(imitation_loss), controller)

    assert problem.is_dblp()
    return (
        baseline_lower_objective,
        delta,
        eta,
        problem,
        u,
        u_baseline,
        x,
        x_baseline,
    )


@app.cell
def _(problem):
    epsilon_target = 1e-5
    result = problem.solve(
        epsilon_initial=1e-2,
        epsilon_target=epsilon_target,
        solver_options={"tol": 1e-10, "max_iter": 3000},
        verbose=False,
    )
    diagnostics = problem.gap_diagnostics(result)
    return diagnostics, epsilon_target, result


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Independent learned-weight reference

    A final CVXPY solve fixes the learned weights and independently checks the
    recovered trajectory and lower objective.
    """)
    return


@app.cell
def _(delta, eta, solve_fixed_mpc, u_previous, x_initial):
    learned_delta = float(delta.value)
    learned_eta = float(eta.value)
    x_independent, u_independent, reference_lower_objective = solve_fixed_mpc(
        learned_delta,
        learned_eta,
        x_initial,
        u_previous,
    )
    return (
        learned_delta,
        learned_eta,
        reference_lower_objective,
        u_independent,
        x_independent,
    )


@app.cell(hide_code=True)
def _(
    A,
    B,
    baseline_delta,
    baseline_eta,
    baseline_lower_objective,
    delta_bounds,
    diagnostics,
    epsilon_target,
    eta_bounds,
    expert_delta,
    expert_eta,
    expert_lower_objective,
    i_max,
    learned_delta,
    learned_eta,
    mo,
    np,
    omega_max,
    omega_min,
    reference_lower_objective,
    result,
    u,
    u_baseline,
    u_hat,
    u_independent,
    u_max,
    x,
    x_baseline,
    x_hat,
    x_independent,
):
    x_learned = np.asarray(x.value, dtype=float)
    u_learned = np.asarray(u.value, dtype=float)
    baseline_imitation_loss = float(np.sum(np.square(x_baseline - x_hat)) + 0.1 * np.sum(np.square(u_baseline - u_hat)))
    learned_imitation_loss = float(np.sum(np.square(x_learned - x_hat)) + 0.1 * np.sum(np.square(u_learned - u_hat)))
    x_learned_next = A @ x_learned[:, :-1] + B.reshape(2, 1) * u_learned.reshape(1, -1)
    dynamics_residual = x_learned[:, 1:] - x_learned_next

    assert result.succeeded, result.message
    assert np.isfinite(result.objective)
    assert np.isclose(result.final_epsilon, epsilon_target, rtol=0.0, atol=1e-15)
    assert result.residuals.max_violation <= 2e-7
    assert result.complementarity <= 1.1 * epsilon_target
    assert abs(diagnostics.source_gap) <= 1.1 * epsilon_target
    assert delta_bounds[0] <= learned_delta <= delta_bounds[1]
    assert eta_bounds[0] <= learned_eta <= eta_bounds[1]
    assert abs(learned_delta - expert_delta) <= 1e-5
    assert abs(learned_eta - expert_eta) <= 1e-5
    assert np.max(np.abs(dynamics_residual)) <= 1e-7
    assert np.max(np.abs(u_learned)) <= u_max + 1e-7
    assert np.max(np.abs(x_learned[1, :])) <= i_max + 1e-7
    assert np.min(x_learned[0, :]) >= omega_min - 1e-7
    assert np.max(x_learned[0, :]) <= omega_max + 1e-7
    assert np.allclose(x_learned, x_independent, atol=1e-5)
    assert np.allclose(u_learned, u_independent, atol=1e-5)
    assert abs(reference_lower_objective - expert_lower_objective) <= 1e-5
    assert learned_imitation_loss <= 1e-8
    assert learned_imitation_loss <= 1e-6 * baseline_imitation_loss

    _baseline_weights = f"({baseline_delta:.3f}, {baseline_eta:.3f})"
    _expert_weights = f"({expert_delta:.3f}, {expert_eta:.3f})"
    _learned_weights = f"({learned_delta:.6f}, {learned_eta:.6f})"

    mo.md(rf"""
    ## Result

    | quantity | value |
    | --- | ---: |
    | status | `{result.status}` |
    | baseline weights $(\delta,\eta)$ | {_baseline_weights} |
    | expert weights $(\delta^{{\rm E}},\eta^{{\rm E}})$ | {_expert_weights} |
    | learned weights $(\delta^\star,\eta^\star)$ | {_learned_weights} |
    | baseline imitation loss | {baseline_imitation_loss:.6f} |
    | learned imitation loss | {learned_imitation_loss:.3e} |
    | baseline lower objective | {baseline_lower_objective:.6f} |
    | learned fixed-weight lower objective | {reference_lower_objective:.6f} |
    | upper objective | {float(result.objective):.3e} |
    | final epsilon | {result.final_epsilon:.3e} |
    | maximum lifted residual | {result.residuals.max_violation:.3e} |
    | complementarity | {result.complementarity:.3e} |
    | source gap | {diagnostics.source_gap:.3e} |

    BLVPY recovers both hidden weights and reduces the trajectory-matching loss
    from {baseline_imitation_loss:.3f} to essentially zero. The independent
    fixed-weight MPC solve reproduces the learned state and voltage sequences.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Receding-horizon validation

    To validate closed-loop MPC, we repeat the controller for 36 samples,
    applying only the first voltage from each fresh 24-step solve. Independent
    CVXPY rollouts compare the expert, untuned, and learned weights. An
    unannounced load-torque pulse acts from $0.6$ to $0.8$ seconds, so each
    controller must correct after observing the disturbed motor state.
    """)
    return


@app.cell
def _(
    A,
    B,
    J,
    baseline_delta,
    baseline_eta,
    expert_delta,
    expert_eta,
    i_max,
    learned_delta,
    learned_eta,
    np,
    omega_max,
    omega_min,
    sample_time,
    solve_fixed_mpc,
    u_max,
    u_previous,
    x_initial,
    x_reference,
):
    rollout_steps = 36
    rollout_time = sample_time * np.arange(rollout_steps + 1)
    rollout_input_time = rollout_time[:-1]
    disturbance_start = 0.6
    disturbance_end = 0.8
    load_torque = np.zeros(rollout_steps)
    load_torque[(rollout_input_time >= disturbance_start) & (rollout_input_time < disturbance_end)] = 0.012
    load_disturbance_vector = np.array([-sample_time / J, 0.0])

    def rollout_mpc(fixed_delta, fixed_eta):
        _x = np.zeros((2, rollout_steps + 1))
        _u = np.zeros(rollout_steps)
        _x[:, 0] = x_initial
        _u_previous = u_previous
        for _step in range(rollout_steps):
            _, _u_plan, _ = solve_fixed_mpc(
                fixed_delta,
                fixed_eta,
                _x[:, _step],
                _u_previous,
            )
            _u[_step] = _u_plan[0]
            _x[:, _step + 1] = A @ _x[:, _step] + B * _u[_step] + load_disturbance_vector * load_torque[_step]
            _u_previous = _u[_step]
        return _x, _u

    x_expert_rollout, u_expert_rollout = rollout_mpc(expert_delta, expert_eta)
    x_baseline_rollout, u_baseline_rollout = rollout_mpc(baseline_delta, baseline_eta)
    x_learned_rollout, u_learned_rollout = rollout_mpc(learned_delta, learned_eta)
    baseline_tracking_mse = float(np.mean(np.square(x_baseline_rollout[0, :] - x_reference[0])))
    learned_tracking_mse = float(np.mean(np.square(x_learned_rollout[0, :] - x_reference[0])))

    assert np.max(np.abs(x_learned_rollout - x_expert_rollout)) <= 1e-4
    assert np.max(np.abs(u_learned_rollout - u_expert_rollout)) <= 1e-4
    assert learned_tracking_mse <= 0.9 * baseline_tracking_mse
    assert abs(x_learned_rollout[0, -1] - x_reference[0]) <= 1e-3
    assert np.max(np.abs(u_learned_rollout)) <= u_max + 1e-7
    assert np.max(np.abs(x_learned_rollout[1, :])) <= i_max + 1e-7
    assert np.min(x_learned_rollout[0, :]) >= omega_min - 1e-7
    assert np.max(x_learned_rollout[0, :]) <= omega_max + 1e-7
    return (
        baseline_tracking_mse,
        disturbance_end,
        disturbance_start,
        learned_tracking_mse,
        u_baseline_rollout,
        u_expert_rollout,
        u_learned_rollout,
        x_baseline_rollout,
        x_expert_rollout,
        x_learned_rollout,
    )


@app.cell(hide_code=True)
def _(baseline_tracking_mse, learned_tracking_mse, mo):
    tracking_improvement = 100.0 * (baseline_tracking_mse - learned_tracking_mse) / baseline_tracking_mse
    mo.md(rf"""
    ## Closed-loop interpretation

    The learned trajectory overlaps the expert, reducing speed-tracking MSE by
    {tracking_improvement:.1f}% relative to the untuned controller.
    The initial plan is the single 24-step expert trajectory used
    for training and does not anticipate the shaded load-torque pulse. The
    closed-loop controllers replan after the resulting speed drop. Horizontal
    lines mark the targets and operating limits.
    """)
    return


@app.cell(hide_code=True)
def _(
    N,
    Path,
    disturbance_end,
    disturbance_start,
    i_max,
    np,
    omega_max,
    omega_min,
    plt,
    sample_time,
    u_baseline_rollout,
    u_expert_rollout,
    u_hat,
    u_learned_rollout,
    u_max,
    u_reference,
    x_baseline_rollout,
    x_expert_rollout,
    x_hat,
    x_learned_rollout,
    x_reference,
):
    control_figure, control_axes = plt.subplots(
        3,
        1,
        figsize=(7.4, 7.2),
        sharex=True,
        layout="constrained",
    )
    prediction_index = np.arange(N + 1)
    prediction_input_index = prediction_index[:-1]
    rollout_index = np.arange(x_baseline_rollout.shape[1])
    rollout_input_index = np.arange(u_baseline_rollout.size)
    disturbance_start_index = round(disturbance_start / sample_time)
    disturbance_end_index = round(disturbance_end / sample_time)

    def draw_state_comparison(axis, x_index):
        axis.plot(
            prediction_index,
            x_hat[x_index],
            color="C2",
            linestyle="-.",
            linewidth=1.0,
            marker="x",
            markersize=4.0,
            markevery=3,
            zorder=5,
            label="Demonstration",
        )
        axis.plot(
            rollout_index,
            x_baseline_rollout[x_index],
            color="C1",
            linestyle=":",
            linewidth=2.0,
            label="Untuned",
        )
        axis.plot(
            rollout_index,
            x_learned_rollout[x_index],
            color="C0",
            linewidth=2.2,
            label="Learned",
        )
        axis.plot(
            rollout_index,
            x_expert_rollout[x_index],
            color="black",
            linestyle="--",
            linewidth=1.1,
            marker="o",
            markersize=3.5,
            markerfacecolor="white",
            markevery=4,
            label="Expert",
        )

    draw_state_comparison(control_axes[0], 0)
    control_axes[0].axhline(x_reference[0], color="0.4", linestyle="--", linewidth=0.8)
    control_axes[0].axhline(omega_min, color="0.65", linestyle="-.", linewidth=0.8)
    control_axes[0].axhline(omega_max, color="0.65", linestyle="-.", linewidth=0.8)
    control_axes[0].set(ylabel=r"$\omega(t)$", ylim=(-0.1, 1.32))
    control_axes[0].legend(
        ncol=4,
        frameon=False,
        fontsize=12,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
    )

    draw_state_comparison(control_axes[1], 1)
    control_axes[1].axhline(x_reference[1], color="0.4", linestyle="--", linewidth=0.8)
    control_axes[1].axhline(-i_max, color="0.65", linestyle="-.", linewidth=0.8)
    control_axes[1].axhline(i_max, color="0.65", linestyle="-.", linewidth=0.8)
    control_axes[1].set(ylabel=r"$i(t)$", ylim=(-0.55, 0.55))

    control_axes[2].step(
        prediction_input_index,
        u_hat,
        where="post",
        color="C2",
        linestyle="-.",
        linewidth=1.0,
        marker="x",
        markersize=4.0,
        markevery=3,
        zorder=5,
    )
    control_axes[2].step(
        rollout_input_index,
        u_baseline_rollout,
        where="post",
        color="C1",
        linestyle=":",
        linewidth=2.0,
    )
    control_axes[2].step(
        rollout_input_index,
        u_learned_rollout,
        where="post",
        color="C0",
        linewidth=2.2,
    )
    control_axes[2].step(
        rollout_input_index,
        u_expert_rollout,
        where="post",
        color="black",
        linestyle="--",
        linewidth=1.1,
        marker="o",
        markersize=3.5,
        markerfacecolor="white",
        markevery=4,
    )
    control_axes[2].axhline(u_reference, color="0.4", linestyle="--", linewidth=0.8)
    control_axes[2].axhline(-u_max, color="0.65", linestyle="-.", linewidth=0.8)
    control_axes[2].axhline(u_max, color="0.65", linestyle="-.", linewidth=0.8)
    control_axes[2].set(
        xlabel="$t$",
        ylabel=r"$u(t)$",
        xlim=(rollout_index[0], rollout_index[-1]),
        ylim=(-2.12, 2.12),
    )

    for _axis in control_axes:
        _axis.axvspan(
            disturbance_start_index,
            disturbance_end_index,
            color="0.88",
            alpha=0.7,
            zorder=0,
        )
        _axis.grid(alpha=0.2, linewidth=0.6)

    control_axes[0].text(
        0.5 * (disturbance_start_index + disturbance_end_index),
        1.18,
        "Load torque",
        ha="center",
        va="center",
        fontsize=8.5,
        color="0.35",
    )

    figure_dir = Path(__file__).resolve().parent / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / "dc_motor_mpc_tuning.pdf"
    control_figure.savefig(figure_path, bbox_inches="tight")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
