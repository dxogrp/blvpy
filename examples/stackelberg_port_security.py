import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Stackelberg Security Allocation

    A defender allocates limited security resources across a set of targets. An
    attacker observes the protection policy and selects the most attractive
    target. The defender therefore acts as a Stackelberg leader and anticipates
    the attacker's best response when allocating protection.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import cvxpy as cp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    from blvpy import BilevelProblem, LowerProblem

    plt.style.use(Path(__file__).resolve().parent / "zhlatex.mplstyle")
    return BilevelProblem, LowerProblem, Patch, Path, cp, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bilevel formulation

    Suppose there are $n$ targets. Let $p\in\mathbf{R}^n$, with
    $0\preceq p\preceq\mathbf{1}$, denote their protection probabilities, and
    let $q\in\mathbf{R}^n$, with $0\preceq q\preceq\mathbf{1}$, denote their
    attack probabilities. For a fixed defender allocation $p$, the attacker
    solves

    \[
    \begin{array}{rl}
        S(p) = \mathop{\mathrm{argmin}}_{q} &
          -\displaystyle\sum_{i=1}^n(r_i-c_ip_i)q_i\\
        \text{subject to} &
          q\succeq0,\quad\mathbf{1}^Tq=1,
    \end{array}
    \]

    where $r,c\in\mathbf{R}^n$ are the attacker reward and defender coverage
    effectiveness vectors. Thus $r_i-c_ip_i$ is the attacker's payoff for
    target $i$ after observing defender coverage.

    The defender adapts its protection strategy by solving

    \[
    \begin{array}{ll}
    \mathop{\mathrm{minimize}}_{p,q} &
      \displaystyle\sum_{i=1}^n L_i(1-p_i)q_i
      +\gamma\lVert p\rVert_2^2\\
    \mathop{\mathrm{subject\ to}} &
      0\preceq p\preceq\mathbf{1},\quad\mathbf{1}^Tp\leq B,\\
      &q\in S(p).
    \end{array}
    \]

    The first upper-level term is the expected defender loss: $L_i$ is the
    uncovered loss at target $i$, and $(1-p_i)q_i$ is its probability of being
    attacked without protection. The quadratic term discourages concentrated
    policies. The vectors $r,c,L$, penalty $\gamma>0$, and resource budget
    $B>0$ are given. This instance uses $B=2.5$ and $\gamma=0.03$.

    The lower linear program can have multiple optimal attack targets. BLVPY's
    optimistic semantics implement the strong-Stackelberg convention: among
    tied responses in $S(p)$, the one most favorable to the defender may be
    selected.
    """)
    return


@app.cell(hide_code=True)
def _(B, L, mo, n, r):
    mo.md(rf"""
    ## Synthetic security instance

    The instance contains {n} synthetic targets.
    Uncovered attacker rewards range from
    `{r.min():.1f}` to `{r.max():.1f}`, while defender losses range from
    `{L.min():.1f}` to `{L.max():.1f}`. The security budget of `{B:.1f}` is
    insufficient to cover every target fully.
    """)
    return


@app.cell
def _(np):
    r = np.array([4.0, 3.2, 5.0, 2.6, 4.4, 3.8, 5.5, 3.0])
    c = np.array([3.0, 3.5, 2.2, 2.0, 4.4, 2.8, 3.2, 4.0])
    L = np.array([8.0, 5.0, 10.0, 4.0, 7.0, 6.0, 12.0, 5.0])
    B = 2.5
    gamma = 0.03
    n = r.size
    target_indices = np.arange(1, n + 1)
    return B, L, c, gamma, n, r, target_indices


@app.cell(hide_code=True)
def _(L, Path, c, np, plt, r, target_indices):
    payoff_figure_dir = Path(__file__).resolve().parent / "figures"
    payoff_figure_dir.mkdir(parents=True, exist_ok=True)
    payoff_figure, payoff_axes = plt.subplots(
        3,
        1,
        figsize=(5.5, 4.5),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    payoff_values = (r, c, L)
    payoff_labels = (r"$r$", r"$c$", r"$L$")
    maximum_payoff = max(np.max(values) for values in payoff_values)
    for payoff_axis, values, label in zip(
        payoff_axes,
        payoff_values,
        payoff_labels,
        strict=True,
    ):
        payoff_axis.bar(target_indices, values, width=0.72, facecolor="w", edgecolor="k", linewidth=1.0, hatch="....")
        payoff_axis.set(ylabel=label, ylim=(0.0, 1.08 * maximum_payoff))
    payoff_axes[-1].set_xticks(target_indices)
    payoff_axes[-1].set_xlabel("$i$")

    payoff_figure_path = payoff_figure_dir / "stackelberg_port_security_payoffs.pdf"
    payoff_figure.savefig(payoff_figure_path, bbox_inches="tight")
    plt.show()
    return


@app.cell
def _(B, BilevelProblem, L, LowerProblem, c, cp, gamma, n, np, r):
    p = cp.Variable(n, bounds=[0.0, 1.0], name="p")
    p.value = np.full(n, B / n)
    q = cp.Variable(n, nonneg=True, name="q")

    attacker_payoff = r - cp.multiply(c, p)
    attacker_problem = LowerProblem(
        cp.Minimize(-attacker_payoff @ q),
        [cp.sum(q) == 1.0],
        parameters=[p],
    )
    uncovered_loss = cp.multiply(L, 1.0 - p)
    expected_defender_loss = uncovered_loss @ q
    problem = BilevelProblem(
        cp.Minimize(expected_defender_loss + gamma * cp.sum_squares(p)),
        attacker_problem,
        upper_constraints=[cp.sum(p) <= B],
    )
    assert problem.is_dblp()
    problem.validate()
    return p, problem, q


@app.cell
def _(problem):
    epsilon_target = 1e-5
    result = problem.solve(
        epsilon_initial=1e-2,
        epsilon_target=epsilon_target,
        solver_options={"max_iter": 1000},
        verbose=False,
    )
    diagnostics = problem.gap_diagnostics(result)
    return diagnostics, epsilon_target, result


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Baseline protection policies

    We compare the optimized policy with two transparent baselines. The
    no-protection policy leaves every target uncovered. The uniform policy
    divides the complete resource budget equally.
    """)
    return


@app.cell
def _(B, L, c, n, np, r):
    def _reference_response(p_value):
        payoff_value = r - c * p_value
        best_payoff = float(np.max(payoff_value))
        tied_targets = np.flatnonzero(np.isclose(payoff_value, best_payoff, atol=1e-10, rtol=0.0))
        tied_losses = L[tied_targets] * (1.0 - p_value[tied_targets])
        selected_target = int(tied_targets[np.argmin(tied_losses)])
        q_value = np.zeros(n)
        q_value[selected_target] = 1.0
        defender_outcome = float((L * (1.0 - p_value)) @ q_value)
        attacker_outcome = float(payoff_value @ q_value)
        return q_value, attacker_outcome, defender_outcome

    baseline_p = np.vstack(
        (
            np.zeros(n),
            np.full(n, B / n),
        )
    )
    baseline_q = np.empty_like(baseline_p)
    baseline_attacker_utility = np.empty(baseline_p.shape[0])
    baseline_defender_loss = np.empty(baseline_p.shape[0])
    for baseline_index, fixed_p in enumerate(baseline_p):
        (
            baseline_q[baseline_index],
            baseline_attacker_utility[baseline_index],
            baseline_defender_loss[baseline_index],
        ) = _reference_response(fixed_p)
    return (
        baseline_attacker_utility,
        baseline_defender_loss,
        baseline_p,
        baseline_q,
    )


@app.cell(hide_code=True)
def _(
    B,
    L,
    baseline_defender_loss,
    c,
    diagnostics,
    epsilon_target,
    gamma,
    mo,
    np,
    p,
    q,
    r,
    result,
    target_indices,
):
    optimized_p = np.array(p.value, copy=True)
    optimized_q = np.array(q.value, copy=True)
    attacker_payoff_by_target = r - c * optimized_p
    optimized_defender_loss = float(np.sum(L * (1.0 - optimized_p) * optimized_q))
    protection_regularizer = float(gamma * np.sum(np.square(optimized_p)))
    attacked_target_index = int(np.argmax(optimized_q))
    attacked_target_number = int(target_indices[attacked_target_index])
    maximum_attacker_utility = float(np.max(attacker_payoff_by_target))
    optimized_attacker_utility = float(attacker_payoff_by_target @ optimized_q)
    active_attack_targets = np.flatnonzero(optimized_q > 1e-6)
    near_optimal_target_count = int(
        np.count_nonzero(attacker_payoff_by_target >= maximum_attacker_utility - 1.1 * epsilon_target)
    )
    used_budget = float(np.sum(optimized_p))
    uniform_loss_reduction = 100.0 * (baseline_defender_loss[1] - optimized_defender_loss) / baseline_defender_loss[1]

    assert result.succeeded, result.message
    assert np.isfinite(result.objective)
    assert np.isclose(result.final_epsilon, epsilon_target)
    assert result.residuals.max_violation <= 1e-7
    assert abs(diagnostics.source_gap) <= 1.1 * epsilon_target
    assert np.all((optimized_p >= -1e-8) & (optimized_p <= 1.0 + 1e-8))
    assert used_budget <= B + 1e-7
    assert np.isclose(np.sum(optimized_q), 1.0, atol=1e-7)
    assert np.all(optimized_q >= -1e-8)
    assert np.all(attacker_payoff_by_target[active_attack_targets] >= maximum_attacker_utility - 1.1 * epsilon_target)
    assert optimized_defender_loss < np.min(baseline_defender_loss) - 1.0

    mo.md(rf"""
    ## Result

    | quantity | value |
    | --- | ---: |
    | status | `{result.status}` |
    | most likely attack target | $i={attacked_target_number}$ |
    | maximum attacker utility | {maximum_attacker_utility:.6f} |
    | optimized expected attacker utility | {optimized_attacker_utility:.6f} |
    | no-protection defender loss | {baseline_defender_loss[0]:.6f} |
    | uniform-protection defender loss | {baseline_defender_loss[1]:.6f} |
    | optimized expected defender loss | {optimized_defender_loss:.6f} |
    | $\gamma\lVert p^\star\rVert_2^2$ | {protection_regularizer:.6f} |
    | $\mathbf{{1}}^Tp^\star$ | {used_budget:.6f} / {B:.1f} |
    | final epsilon | {result.final_epsilon:.3e} |
    | maximum lifted residual | {result.residuals.max_violation:.3e} |
    | signed source gap | {diagnostics.source_gap:.3e} |

    The policy makes {near_optimal_target_count} targets approximate best
    responses. Under optimistic strong-Stackelberg semantics, the most likely
    attack is target {attacked_target_number}, the leader-favorable choice among
    those responses. The optimized policy reduces defender loss by
    `{uniform_loss_reduction:.1f}%` relative to uniform protection.
    """)
    return (
        optimized_attacker_utility,
        optimized_defender_loss,
        optimized_p,
        optimized_q,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Policy and outcome comparisons

    The mirrored policy chart compares both sides of the game under no
    protection, uniform protection, and optimized protection. The defender
    probabilities $p$ extend right, while the attacker probabilities $q$ extend
    left on the same scale. The outcome panel compares expected defender loss
    and expected attacker utility across the same three setups.
    """)
    return


@app.cell(hide_code=True)
def _(
    Patch,
    Path,
    baseline_attacker_utility,
    baseline_defender_loss,
    baseline_p,
    baseline_q,
    np,
    optimized_attacker_utility,
    optimized_defender_loss,
    optimized_p,
    optimized_q,
    plt,
    target_indices,
):
    figure_dir = Path(__file__).resolve().parent / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    target_axis = np.arange(target_indices.size)
    setup_names = ("None", "Uniform", "Optimized")
    p_policies = np.vstack((baseline_p, optimized_p))
    q_policies = np.vstack((baseline_q, optimized_q))

    figure = plt.figure(figsize=(7.5, 4), layout="constrained")
    figure_grid = figure.add_gridspec(1, 2, width_ratios=(1.25, 0.75))
    policy_axis = figure.add_subplot(figure_grid[0])
    outcome_axis = figure.add_subplot(figure_grid[1])

    policy_height = 0.25
    setup_colors = ("gray", "C0", "C3")
    for setup_index, setup_color in enumerate(setup_colors):
        policy_offset = (setup_index - 1) * policy_height
        policy_axis.barh(
            target_axis + policy_offset,
            p_policies[setup_index],
            height=policy_height,
            color=setup_color,
            hatch="......",
        )
        policy_axis.barh(
            target_axis + policy_offset,
            -q_policies[setup_index],
            height=policy_height,
            color=setup_color,
            alpha=0.72,
            hatch="//////",
        )
    policy_axis.axvline(0.0, color="k", linewidth=0.8)
    policy_axis.set_yticks(target_axis, target_indices)
    policy_axis.set_xticks((-1.0, -0.5, 0.0, 0.5, 1.0), ("1.0", "0.5", "0", "0.5", "1.0"))
    policy_axis.set(
        xlabel="Probability",
        ylabel="$i$",
        xlim=(-1.08, 1.08),
    )
    policy_axis.invert_yaxis()
    policy_legend_handles = [
        Patch(facecolor=color, edgecolor="none", label=name)
        for name, color in zip(setup_names, setup_colors, strict=True)
    ]
    policy_axis.legend(
        handles=policy_legend_handles,
        frameon=False,
        fontsize=13,
        ncols=1,
        loc="upper left",
    )
    policy_axis.text(
        0.95,
        0.05,
        r"$p$",
        transform=policy_axis.transAxes,
        ha="right",
        va="bottom",
    )
    policy_axis.text(
        0.05,
        0.05,
        r"$q$",
        transform=policy_axis.transAxes,
        ha="left",
        va="bottom",
    )

    outcome_positions = np.arange(len(setup_names))
    outcome_width = 0.36
    defender_outcomes = np.append(baseline_defender_loss, optimized_defender_loss)
    attacker_outcomes = np.append(baseline_attacker_utility, optimized_attacker_utility)
    defender_bars = outcome_axis.bar(
        outcome_positions - outcome_width / 2,
        defender_outcomes,
        width=outcome_width,
        color=setup_colors,
        edgecolor="k",
        linewidth=1.1,
        hatch="....",
    )
    attacker_bars = outcome_axis.bar(
        outcome_positions + outcome_width / 2,
        attacker_outcomes,
        width=outcome_width,
        color=setup_colors,
        edgecolor="k",
        linewidth=1.1,
        hatch="////",
    )
    outcome_axis.bar_label(defender_bars, fmt="%.2f", padding=3, fontsize=11)
    outcome_axis.bar_label(attacker_bars, fmt="%.2f", padding=3, fontsize=11)
    outcome_axis.set_xticks(outcome_positions, setup_names, fontsize=13)
    outcome_axis.set(ylim=(0.0, 1.16 * np.max(defender_outcomes)), ylabel="Loss/Utility")
    outcome_legend_handles = (
        Patch(
            facecolor="white",
            edgecolor="black",
            hatch="....",
            label=r"$\sum_{i=1}^n L_i(1-p_i)q_i$",
        ),
        Patch(
            facecolor="white",
            edgecolor="black",
            hatch="////",
            label=r"$\sum_{i=1}^n(r_i-c_i p_i)q_i$",
        ),
    )
    outcome_axis.legend(
        handles=outcome_legend_handles,
        frameon=False,
        fontsize=12,
        ncols=1,
    )

    figure_path = figure_dir / "stackelberg_port_security.pdf"
    figure.savefig(figure_path, bbox_inches="tight")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
