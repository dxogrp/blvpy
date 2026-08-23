import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Stackelberg Patrol Allocation

    A port authority allocates limited patrol coverage across several maritime
    targets. An attacker observes the long-run coverage policy and selects the
    most attractive target. The authority therefore acts as a Stackelberg
    leader: it must anticipate the attacker's best response when assigning its
    patrol resources.
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

    Let $c_i\in[0,1]$ be the probability that the port authority covers target
    $i$, and let $a_i$ be the probability that the attacker selects it. For a
    fixed coverage policy $c$, the lower problem computes

    \[
    \begin{array}{rl}
        a(c) = \mathop{\mathrm{argmin}}_{a} &
          -\sum_i(r_i-e_ic_i)a_i\\
          \text{subject to} &
          a\succeq0,\quad\mathbf{1}^T a=1.
    \end{array}
    \]

    Here $r_i-e_ic_i$ is the attacker's payoff for target $i$ after observing
    coverage. The negative sign expresses the attacker's payoff maximization in
    BLVPY's lower-level minimization form.

    The upper problem then solves

    \[
    \begin{array}{ll}
    \mathop{\mathrm{minimize}}_{c,a} &
      \displaystyle\sum_i L_i(1-c_i)a_i
      +\rho\lVert c\rVert_2^2\\
    \mathop{\mathrm{subject\ to}} &
      0 \preceq c \preceq \mathbf{1},\quad\mathbf{1}^T c \leq B\\
      &a\in a(c).
    \end{array}
    \]

    The first upper-level term is the expected defender loss: $L_i$ is the
    uncovered loss at target $i$, and $(1-c_i)a_i$ is its attack probability
    without coverage. The quadratic term discourages concentrated patrol
    policies. This instance uses patrol budget $B=2.5$ and penalty
    $\rho=0.03$.

    The lower linear program can have multiple optimal attack targets. BLVPY's
    optimistic semantics implement the strong-Stackelberg convention: among
    tied values in $a(c)$, the response most favorable to the authority may be
    selected.
    """)
    return


@app.cell(hide_code=True)
def _(attacker_reward, defender_loss, mo, patrol_budget, target_count):
    mo.md(f"""
    ## Synthetic port instance

    The instance contains {target_count} heterogeneous synthetic targets.
    Uncovered attacker rewards range from
    `{attacker_reward.min():.1f}` to `{attacker_reward.max():.1f}`, while defender
    losses range from `{defender_loss.min():.1f}` to
    `{defender_loss.max():.1f}`. The patrol budget of `{patrol_budget:.1f}` is
    insufficient to cover every target fully. The payoff figure below stacks
    $r_i$, $e_i$, and $L_i$ vertically against the target index and uses a common
    payoff scale for direct comparison.
    """)
    return


@app.cell
def _(np):
    attacker_reward = np.array([4.0, 3.2, 5.0, 2.6, 4.4, 3.8, 5.5, 3.0])
    protection_effect = np.array([3.0, 3.5, 2.2, 2.0, 4.4, 2.8, 3.2, 4.0])
    defender_loss = np.array([8.0, 5.0, 10.0, 4.0, 7.0, 6.0, 12.0, 5.0])
    patrol_budget = 2.5
    coverage_penalty = 0.03
    target_count = attacker_reward.size
    target_indices = np.arange(1, target_count + 1)
    return (
        attacker_reward,
        coverage_penalty,
        defender_loss,
        patrol_budget,
        protection_effect,
        target_count,
        target_indices,
    )


@app.cell(hide_code=True)
def _(
    Path,
    attacker_reward,
    defender_loss,
    np,
    plt,
    protection_effect,
    target_indices,
):
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
    payoff_values = (attacker_reward, protection_effect, defender_loss)
    payoff_labels = (r"$r$", r"$e$", r"$L$")
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
def _(
    BilevelProblem,
    LowerProblem,
    attacker_reward,
    coverage_penalty,
    cp,
    defender_loss,
    np,
    patrol_budget,
    protection_effect,
    target_count,
):
    coverage = cp.Variable(target_count, bounds=[0.0, 1.0], name="patrol_coverage")
    coverage.value = np.full(target_count, patrol_budget / target_count)
    attack_probability = cp.Variable(target_count, nonneg=True, name="attack_probability")

    attacker_utility = attacker_reward - cp.multiply(protection_effect, coverage)
    attacker_problem = LowerProblem(
        cp.Minimize(-attacker_utility @ attack_probability),
        [cp.sum(attack_probability) == 1.0],
        parameters=[coverage],
    )
    loss_if_attacked = cp.multiply(defender_loss, 1.0 - coverage)
    expected_defender_loss = loss_if_attacked @ attack_probability
    problem = BilevelProblem(
        cp.Minimize(expected_defender_loss + coverage_penalty * cp.sum_squares(coverage)),
        attacker_problem,
        upper_constraints=[cp.sum(coverage) <= patrol_budget],
    )
    assert problem.is_dblp()
    problem.validate()
    return attack_probability, coverage, problem


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
    ## Baseline patrol policies

    We compare the optimized policy with two transparent baselines. The
    no-patrol policy leaves every target uncovered. The uniform policy divides
    the complete resource budget equally. For each fixed policy, the reference
    attacker selects a utility-maximizing target, with defender-favorable
    tie-breaking matching the strong-Stackelberg convention.
    """)
    return


@app.cell
def _(
    attacker_reward,
    defender_loss,
    np,
    patrol_budget,
    protection_effect,
    target_count,
):
    def _reference_response(coverage_value):
        utility_value = attacker_reward - protection_effect * coverage_value
        best_utility = float(np.max(utility_value))
        tied_targets = np.flatnonzero(np.isclose(utility_value, best_utility, atol=1e-10, rtol=0.0))
        tied_losses = defender_loss[tied_targets] * (1.0 - coverage_value[tied_targets])
        selected_target = int(tied_targets[np.argmin(tied_losses)])
        response = np.zeros(target_count)
        response[selected_target] = 1.0
        defender_outcome = float((defender_loss * (1.0 - coverage_value)) @ response)
        attacker_outcome = float(utility_value @ response)
        return response, attacker_outcome, defender_outcome

    baseline_coverage = np.vstack(
        (
            np.zeros(target_count),
            np.full(target_count, patrol_budget / target_count),
        )
    )
    baseline_attack = np.empty_like(baseline_coverage)
    baseline_attacker_utility = np.empty(baseline_coverage.shape[0])
    baseline_defender_loss = np.empty(baseline_coverage.shape[0])
    for baseline_index, fixed_coverage in enumerate(baseline_coverage):
        (
            baseline_attack[baseline_index],
            baseline_attacker_utility[baseline_index],
            baseline_defender_loss[baseline_index],
        ) = _reference_response(fixed_coverage)
    return (
        baseline_attack,
        baseline_attacker_utility,
        baseline_coverage,
        baseline_defender_loss,
    )


@app.cell(hide_code=True)
def _(
    attack_probability,
    attacker_reward,
    baseline_defender_loss,
    coverage,
    coverage_penalty,
    defender_loss,
    diagnostics,
    epsilon_target,
    mo,
    np,
    patrol_budget,
    protection_effect,
    result,
    target_indices,
):
    optimized_coverage = np.array(coverage.value, copy=True)
    optimized_attack = np.array(attack_probability.value, copy=True)
    attacker_utility_by_target = attacker_reward - protection_effect * optimized_coverage
    optimized_defender_loss = float(np.sum(defender_loss * (1.0 - optimized_coverage) * optimized_attack))
    patrol_regularizer = float(coverage_penalty * np.sum(np.square(optimized_coverage)))
    attacked_target_index = int(np.argmax(optimized_attack))
    attacked_target_number = int(target_indices[attacked_target_index])
    maximum_attacker_utility = float(np.max(attacker_utility_by_target))
    optimized_attacker_utility = float(attacker_utility_by_target @ optimized_attack)
    active_attack_targets = np.flatnonzero(optimized_attack > 1e-6)
    near_optimal_target_count = int(
        np.count_nonzero(attacker_utility_by_target >= maximum_attacker_utility - 1.1 * epsilon_target)
    )
    used_coverage = float(np.sum(optimized_coverage))
    uniform_loss_reduction = 100.0 * (baseline_defender_loss[1] - optimized_defender_loss) / baseline_defender_loss[1]

    assert result.succeeded, result.message
    assert np.isfinite(result.objective)
    assert np.isclose(result.final_epsilon, epsilon_target)
    assert result.residuals.max_violation <= 1e-7
    assert abs(diagnostics.source_gap) <= 1.1 * epsilon_target
    assert np.all((optimized_coverage >= -1e-8) & (optimized_coverage <= 1.0 + 1e-8))
    assert used_coverage <= patrol_budget + 1e-7
    assert np.isclose(np.sum(optimized_attack), 1.0, atol=1e-7)
    assert np.all(optimized_attack >= -1e-8)
    assert np.all(attacker_utility_by_target[active_attack_targets] >= maximum_attacker_utility - 1.1 * epsilon_target)
    assert optimized_defender_loss < np.min(baseline_defender_loss) - 1.0

    mo.md(f"""
    ## Result

    | quantity | value |
    | --- | ---: |
    | status | `{result.status}` |
    | most likely attack target | $i={attacked_target_number}$ |
    | maximum attacker utility | {maximum_attacker_utility:.6f} |
    | optimized expected attacker utility | {optimized_attacker_utility:.6f} |
    | no-patrol defender loss | {baseline_defender_loss[0]:.6f} |
    | uniform-patrol defender loss | {baseline_defender_loss[1]:.6f} |
    | optimized expected defender loss | {optimized_defender_loss:.6f} |
    | patrol regularizer | {patrol_regularizer:.6f} |
    | coverage used | {used_coverage:.6f} / {patrol_budget:.1f} |
    | final epsilon | {result.final_epsilon:.3e} |
    | maximum lifted residual | {result.residuals.max_violation:.3e} |
    | signed source gap | {diagnostics.source_gap:.3e} |

    The policy makes {near_optimal_target_count} targets approximate best
    responses. Under optimistic strong-Stackelberg semantics, the most likely
    attack is target {attacked_target_number}, the leader-favorable choice among
    those responses. The optimized policy reduces defender loss by
    `{uniform_loss_reduction:.1f}%` relative to uniform patrol.
    """)
    return (
        optimized_attack,
        optimized_attacker_utility,
        optimized_coverage,
        optimized_defender_loss,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Policy and outcome comparisons

    The mirrored policy chart compares both sides of the game under no patrol,
    uniform patrol, and optimized patrol. Coverage probabilities extend right,
    while attack probabilities extend left on the same scale. The outcome panel
    compares expected defender loss and expected attacker utility across the
    same three setups.
    """)
    return


@app.cell(hide_code=True)
def _(
    Path,
    baseline_attack,
    baseline_attacker_utility,
    baseline_coverage,
    baseline_defender_loss,
    np,
    optimized_attack,
    optimized_attacker_utility,
    optimized_coverage,
    optimized_defender_loss,
    plt,
    target_indices,
):
    figure_dir = Path(__file__).resolve().parent / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    target_axis = np.arange(target_indices.size)
    setup_names = ("No patrol", "Uniform", "Optimized")
    coverage_policies = np.vstack((baseline_coverage, optimized_coverage))
    attack_policies = np.vstack((baseline_attack, optimized_attack))

    figure = plt.figure(figsize=(7.5, 4), layout="constrained")
    figure_grid = figure.add_gridspec(1, 2, width_ratios=(1.25, 0.75))
    policy_axis = figure.add_subplot(figure_grid[0])
    outcome_axis = figure.add_subplot(figure_grid[1])

    policy_height = 0.25
    setup_colors = ("gray", "C0", "C3")
    for setup_index, (setup_name, setup_color) in enumerate(zip(setup_names, setup_colors, strict=True)):
        policy_offset = (setup_index - 1) * policy_height
        policy_axis.barh(
            target_axis + policy_offset,
            coverage_policies[setup_index],
            height=policy_height,
            color=setup_color,
            label=setup_name,
        )
        policy_axis.barh(
            target_axis + policy_offset,
            -attack_policies[setup_index],
            height=policy_height,
            color=setup_color,
            alpha=0.72,
            hatch="////",
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
    policy_axis.legend(frameon=False, fontsize=13, ncols=1, loc="upper left")
    policy_axis.text(
        0.98, 0.02, "Patrol coverage", transform=policy_axis.transAxes, ha="right", va="bottom", fontsize=13
    )
    policy_axis.text(
        0.02, 0.02, "Attacker response", transform=policy_axis.transAxes, ha="left", va="bottom", fontsize=13
    )

    outcome_positions = np.arange(len(setup_names))
    outcome_width = 0.36
    defender_outcomes = np.append(baseline_defender_loss, optimized_defender_loss)
    attacker_outcomes = np.append(baseline_attacker_utility, optimized_attacker_utility)
    defender_bars = outcome_axis.bar(
        outcome_positions - outcome_width / 2,
        defender_outcomes,
        width=outcome_width,
        facecolor="w",
        edgecolor="k",
        linewidth=1.1,
        label=r"$\sum_i L_i(1-c_i)a_i$",
    )
    attacker_bars = outcome_axis.bar(
        outcome_positions + outcome_width / 2,
        attacker_outcomes,
        width=outcome_width,
        facecolor="w",
        edgecolor="k",
        linewidth=1.1,
        hatch="////",
        label=r"$\sum_i(r_i-e_i c_i)a_i$",
    )
    outcome_axis.bar_label(defender_bars, fmt="%.2f", padding=3, fontsize=11)
    outcome_axis.bar_label(attacker_bars, fmt="%.2f", padding=3, fontsize=11)
    outcome_axis.set_xticks(outcome_positions, setup_names, fontsize=13)
    outcome_axis.set(ylim=(0.0, 1.16 * np.max(defender_outcomes)), ylabel="Loss/Utility")
    outcome_axis.legend(frameon=False, fontsize=12, ncols=1)

    figure_path = figure_dir / "stackelberg_port_security.pdf"
    figure.savefig(figure_path, bbox_inches="tight")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
