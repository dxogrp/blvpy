import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Time-of-Use Pricing for Residential Demand Response

    A utility would like to reduce the daily demand peak, but customers retain
    control of when they consume flexible energy. This creates a bilevel
    problem: the **upper problem** chooses a 24-hour price vector for the
    utility, while the **lower problem** models how a representative customer
    reschedules consumption while balancing cost against discomfort.
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
    ## Customer response and utility objective

    Let $p_t$ be the price, $\ell_t$ the customer's load, and
    $\bar\ell_t$ the preferred load in hour $t$. For a published price vector,
    the customer solves

    \[
    \begin{array}{ll}
    \mathop{\mathrm{minimize}}_{\ell}
      & p^T\ell+\gamma\lVert\ell-\bar\ell\rVert_2^2 \\
    \mathop{\mathrm{subject\ to}}
      & \mathbf{1}^T\ell=E,\\
      & \ell_{\min}\leq\ell_t\leq\ell_{\max}.
    \end{array}
    \]

    The energy equality models shifting rather than curtailment. The quadratic
    term represents inconvenience from departing from the preferred schedule
    and makes the lower response unique.

    The upper problem introduces a peak epigraph $z$ and anticipates the lower
    response:

    \[
    \begin{array}{ll}
    \mathop{\mathrm{minimize}}_{p,z,\ell}
      & z+\eta\lVert p\rVert_2^2+\kappa\lVert Dp\rVert_2^2\\
    \mathop{\mathrm{subject\ to}}
      & \ell_t\leq z,\quad 0\leq z\leq 4,\\
      & 0\leq p_t\leq 0.8,\\
      & \ell\text{ solves the customer problem for }p.
    \end{array}
    \]

    The magnitude penalty discourages unnecessarily high prices, while the
    difference operator $D$ discourages abrupt hour-to-hour tariff changes.
    The loose upper bound on $z$ supplies a practical scale for this synthetic
    example; the optimized peak lies well below it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Synthetic daily profile

    The preferred schedule combines a modest morning rise with a larger
    evening peak. All 24 hourly decisions remain in the model.
    """)
    return


@app.cell
def _(np):
    hours = np.arange(24, dtype=float)
    preferred_load = (
        0.75
        + 0.75 * np.exp(-0.5 * np.square((hours - 8.0) / 1.8))
        + 1.65 * np.exp(-0.5 * np.square((hours - 19.0) / 2.1))
    )
    daily_energy = float(np.sum(preferred_load))
    minimum_hourly_load = 0.45
    maximum_hourly_load = 3.2
    discomfort_weight = 0.45
    return (
        daily_energy,
        discomfort_weight,
        hours,
        maximum_hourly_load,
        minimum_hourly_load,
        preferred_load,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Specify and solve the bilevel model

    `price` is the utility's vector decision and is listed as a lower-level
    parameter. The `.value` assignments provide a neutral solving initial point.
    """)
    return


@app.cell
def _(
    BilevelProblem,
    LowerProblem,
    cp,
    daily_energy,
    discomfort_weight,
    maximum_hourly_load,
    minimum_hourly_load,
    np,
    preferred_load,
):
    price = cp.Variable(24, name="hourly_price")
    peak_load = cp.Variable(name="peak_load")
    load = cp.Variable(24, name="hourly_load")
    price.value = np.full(24, 0.4)
    peak_load.value = 2.0

    lower_problem = LowerProblem(
        cp.Minimize(price @ load + discomfort_weight * cp.sum_squares(load - preferred_load)),
        [
            cp.sum(load) == daily_energy,
            load >= minimum_hourly_load,
            load <= maximum_hourly_load,
        ],
        parameters=[price],
    )
    upper_objective = peak_load + 0.025 * cp.sum_squares(price) + 0.12 * cp.sum_squares(price[1:] - price[:-1])
    problem = BilevelProblem(
        cp.Minimize(upper_objective),
        lower_problem,
        upper_constraints=[
            price >= 0.0,
            price <= 0.8,
            peak_load >= 0.0,
            peak_load <= 4.0,
            load <= peak_load,
        ],
    )
    return load, peak_load, price, problem


@app.cell
def _(problem):
    epsilon_target = 1e-5
    result = problem.solve(
        epsilon_initial=1e-2,
        epsilon_target=epsilon_target,
        verbose=False,
    )
    diagnostics = problem.gap_diagnostics(result)
    return diagnostics, result


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Flat-price reference response

    With a zero, flat price the energy payment does not encourage shifting.
    We solve that fixed-tariff customer problem independently with CVXPY to
    obtain a reference peak and utility objective.
    """)
    return


@app.cell
def _(
    cp,
    daily_energy,
    discomfort_weight,
    maximum_hourly_load,
    minimum_hourly_load,
    np,
    preferred_load,
):
    baseline_load_variable = cp.Variable(24, name="baseline_hourly_load")
    baseline_price = np.zeros(24)
    baseline_problem = cp.Problem(
        cp.Minimize(
            baseline_price @ baseline_load_variable
            + discomfort_weight * cp.sum_squares(baseline_load_variable - preferred_load)
        ),
        [
            cp.sum(baseline_load_variable) == daily_energy,
            baseline_load_variable >= minimum_hourly_load,
            baseline_load_variable <= maximum_hourly_load,
        ],
    )
    baseline_problem.solve(solver=cp.CLARABEL)
    baseline_load = np.asarray(baseline_load_variable.value, dtype=float)
    baseline_peak = float(np.max(baseline_load))
    return baseline_load, baseline_peak


@app.cell(hide_code=True)
def _(
    baseline_peak,
    daily_energy,
    diagnostics,
    load,
    mo,
    np,
    peak_load,
    price,
    result,
):
    assert result.succeeded, result.message

    optimized_load = np.asarray(load.value, dtype=float)
    optimized_price = np.asarray(price.value, dtype=float)
    optimized_peak = float(np.max(optimized_load))
    assert abs(float(np.sum(optimized_load)) - daily_energy) <= 1e-5, (
        "The optimized load does not preserve total daily energy."
    )
    assert optimized_peak < baseline_peak - 0.2, "The optimized tariff did not materially reduce peak demand."

    mo.md(f"""
    ## Result

    | quantity | value |
    | --- | ---: |
    | status | `{result.status}` |
    | flat-price peak load | {baseline_peak:.4f} |
    | optimized peak load | {optimized_peak:.4f} |
    | peak epigraph | {float(peak_load.value):.4f} |
    | utility objective | {float(result.objective):.6f} |
    | final epsilon | {result.final_epsilon:.3e} |
    | maximum lifted residual | {result.residuals.max_violation:.3e} |
    | complementarity | {result.complementarity:.3e} |
    | signed source gap | {diagnostics.source_gap:.3e} |

    The tariff raises prices around the evening peak and shifts energy into
    lower-demand hours without changing total daily consumption. The optimized
    peak is materially below the flat-price reference, even after penalizing
    tariff magnitude and roughness.
    """)
    return optimized_load, optimized_price


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load shifting and the price signal

    The upper panel compares the customer's preferred schedule with its
    optimized response. The lower panel shows the time-of-use price that
    induces the shift.
    """)
    return


@app.cell
def _(
    Path,
    baseline_load,
    hours,
    optimized_load,
    optimized_price,
    plt,
    preferred_load,
):
    figure_dir = Path(__file__).resolve().parent / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.0, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )
    axes[0].plot(hours, preferred_load, linestyle="--", linewidth=2.0, label=r"$\bar{\ell}$")
    axes[0].plot(hours, baseline_load, linestyle=":", linewidth=1.5, label="Flat-price response")
    axes[0].plot(hours, optimized_load, linewidth=2.2, label="Optimized-price response")
    axes[0].set(ylabel=r"$\ell$")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].bar(hours, optimized_price, width=0.85, color="#e41a1c", alpha=0.85)
    axes[1].set(
        xlabel="$t$",
        ylabel="$p$",
        xticks=range(0, 24, 3),
    )
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    figure_path = figure_dir / "demand_response_pricing.pdf"
    fig.savefig(figure_path, bbox_inches="tight")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
