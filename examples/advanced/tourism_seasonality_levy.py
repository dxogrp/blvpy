import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tourism Seasonality Levies with Eurostat Data

    This example uses bilevel optimization to choose monthly tourism levies.
    A tourism authority seeks a less seasonal distribution of accommodation
    nights, while travelers redistribute a fixed annual share of overnight
    stays in response to the levies.
    """)
    return


@app.cell
def _():
    import csv
    from pathlib import Path

    import cvxpy as cp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from blvpy import BilevelProblem, LowerProblem

    plt.style.use(Path(__file__).resolve().parents[1] / "_shared" / "zhlatex.mplstyle")
    return BilevelProblem, LowerProblem, Path, cp, csv, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data source

    The local CSV contains monthly nights spent at tourist accommodation
    establishments in the EU-27 during 2025. It is adapted from Eurostat
    dataset [`tour_occ_nim`](https://doi.org/10.2908/TOUR_OCC_NIM).
    """)
    return


@app.cell
def _(Path, csv, np):
    data_path = Path(__file__).resolve().parents[1] / "_shared" / "data" / "eurostat_tourism_2025.csv"
    with data_path.open(encoding="utf-8", newline="") as data_file:
        data_rows = list(csv.DictReader(data_file))

    months = np.asarray([row["month"] for row in data_rows], dtype="datetime64[M]")
    nights_spent = np.asarray([float(row["nights_spent"]) for row in data_rows])
    expected_months = np.arange(
        np.datetime64("2025-01"),
        np.datetime64("2026-01"),
        dtype="datetime64[M]",
    )

    assert months.size == 12
    assert np.unique(months).size == months.size
    assert np.array_equal(months, expected_months)
    assert np.all(np.isfinite(nights_spent))
    assert np.all(nights_spent > 0.0)

    observed_share = nights_spent / np.sum(nights_spent)
    uniform_share = np.full(months.size, 1.0 / months.size)
    np.testing.assert_allclose(np.sum(observed_share), 1.0, atol=1e-12, rtol=0.0)

    month_numbers = np.arange(1, months.size + 1)
    month_labels = np.array(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    n_months = months.size
    return (
        month_labels,
        month_numbers,
        n_months,
        nights_spent,
        observed_share,
        uniform_share,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bilevel formulation

    Let $q_t$ be the observed share of annual nights in month $t$. The vector
    $u=(1/12)\mathbf{1}$ is an illustrative policy target in which every month
    receives the same share, so $u_t=1/12$ for all $t$.
    The authority chooses a nonnegative levy
    $p_t$ in euros per night. For fixed $p$, travelers choose a distribution
    $y$ by balancing the expected levy against divergence from the observed
    seasonal pattern:

    \[
    \begin{array}{ll}
    S(p)=\mathop{\mathrm{argmin}}_y
      & p^Ty+\theta\displaystyle\sum_{t=1}^{12}y_t\log(y_t/q_t) \\
    \mathop{\mathrm{subject\ to}}
      & \mathbf{1}^Ty=1, \\
      & y\succeq0.
    \end{array}
    \]

    Anticipating this response, the authority solves

    \[
    \begin{array}{ll}
    \mathop{\mathrm{minimize}}_{p,y}
      & \tfrac12\lVert y-u\rVert_2^2
        +\rho\lVert p/p_{\max}\rVert_2^2 \\
    \mathop{\mathrm{subject\ to}}
      & 0\preceq p\preceq p_{\max}\mathbf{1}, \\
      & y\in S(p).
    \end{array}
    \]

    Thus, $\tfrac12\lVert y-u\rVert_2^2$ measures the remaining seasonality:
    it is zero only when the modeled nights are distributed equally across all
    12 months.

    We use $p_{\max}=40$ euros per night, $\theta=20$ euros, and
    $\rho=0.005$.
    These are illustrative behavioral and policy parameters. The response
    keeps total tourism fixed, so the result is neither an estimate of an
    observed policy effect nor a policy recommendation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Specify and solve the model

    Listing `levy` as a `LowerProblem` parameter makes the monthly rates fixed
    for the traveler response while retaining them as upper-level decisions.
    The zero-levy response is exactly the observed distribution, so we use it
    as the initial point and continue from $\epsilon=10^{-2}$ to $10^{-5}$.
    """)
    return


@app.cell
def _(
    BilevelProblem,
    LowerProblem,
    cp,
    n_months,
    np,
    observed_share,
    uniform_share,
):
    levy_cap = 40.0
    response_scale = 20.0
    levy_penalty = 0.005

    levy = cp.Variable(n_months, name="monthly_levy")
    night_share = cp.Variable(n_months, nonneg=True, name="night_share")
    levy.value = np.zeros(n_months)
    night_share.value = observed_share.copy()

    traveler_problem = LowerProblem(
        cp.Minimize(levy @ night_share + response_scale * cp.sum(cp.rel_entr(night_share, observed_share))),
        [cp.sum(night_share) == 1.0],
        parameters=[levy],
    )
    authority_problem = BilevelProblem(
        cp.Minimize(0.5 * cp.sum_squares(night_share - uniform_share) + levy_penalty * cp.sum_squares(levy / levy_cap)),
        traveler_problem,
        upper_constraints=[levy >= 0.0, levy <= levy_cap],
    )
    authority_problem.validate()
    return authority_problem, levy, levy_cap, night_share, response_scale


@app.cell
def _(authority_problem):
    result = authority_problem.solve(
        epsilon_initial=1e-2,
        epsilon_target=1e-5,
        verbose=False,
    )
    return (result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Independent response check

    The entropy-regularized traveler problem has a closed-form response:

    \[
    y_t^*(p)=
    \frac{q_t\exp(-p_t/\theta)}
         {\sum_s q_s\exp(-p_s/\theta)}.
    \]

    We evaluate this expression at the selected levies and compare its lower
    objective with the response returned by BLVPY. This validates the
    lower-level solution without another optimizer.
    """)
    return


@app.cell
def _(
    levy,
    levy_cap,
    night_share,
    np,
    observed_share,
    response_scale,
    result,
    uniform_share,
):
    assert result.succeeded, result.message

    selected_levy = np.asarray(result.variable_values[levy], dtype=float)
    selected_share = np.asarray(result.variable_values[night_share], dtype=float)
    numerical_tolerance = max(1e-8, 10.0 * result.residuals.max_violation)

    assert np.all(selected_levy >= -numerical_tolerance)
    assert np.all(selected_levy <= levy_cap + numerical_tolerance)
    assert np.all(selected_share > 0.0)
    np.testing.assert_allclose(np.sum(selected_share), 1.0, atol=numerical_tolerance, rtol=0.0)

    log_response_weights = np.log(observed_share) - selected_levy / response_scale
    response_weights = np.exp(log_response_weights - np.max(log_response_weights))
    exact_response = response_weights / np.sum(response_weights)

    selected_lower_objective = float(
        selected_levy @ selected_share
        + response_scale * np.sum(selected_share * np.log(selected_share / observed_share))
    )
    exact_lower_objective = float(
        selected_levy @ exact_response
        + response_scale * np.sum(exact_response * np.log(exact_response / observed_share))
    )
    response_objective_gap = selected_lower_objective - exact_lower_objective
    response_l1_error = float(np.linalg.norm(selected_share - exact_response, ord=1))
    response_error_bound = float(np.sqrt(2.0 * (result.final_epsilon + numerical_tolerance) / response_scale))

    assert response_objective_gap >= -numerical_tolerance
    assert response_objective_gap <= result.final_epsilon + numerical_tolerance
    assert response_l1_error <= response_error_bound

    observed_seasonality = 0.5 * float(np.sum(np.square(observed_share - uniform_share)))
    selected_seasonality = 0.5 * float(np.sum(np.square(selected_share - uniform_share)))
    assert selected_seasonality < observed_seasonality
    return (
        observed_seasonality,
        response_l1_error,
        response_objective_gap,
        selected_levy,
        selected_seasonality,
        selected_share,
    )


@app.cell(hide_code=True)
def _(
    mo,
    nights_spent,
    np,
    observed_seasonality,
    response_l1_error,
    response_objective_gap,
    result,
    selected_levy,
    selected_seasonality,
    selected_share,
):
    mo.md(rf"""
    ## Result

    | quantity | value |
    | --- | ---: |
    | status | `{result.status}` |
    | 2025 accommodation nights | {float(np.sum(nights_spent)) / 1e9:.3f} billion |
    | largest modeled monthly share | {float(np.max(selected_share)):.2%} |
    | largest selected levy | €{float(np.max(selected_levy)):.2f} per night |
    | observed seasonality | {observed_seasonality:.6f} |
    | modeled seasonality | {selected_seasonality:.6f} |
    | fixed-levy response objective gap | {response_objective_gap:.3e} |
    | $\ell_1$ distance from exact response | {response_l1_error:.3e} |

    The illustrative levies reduce the squared distance from a uniform monthly
    distribution. The exact-response comparison confirms that the modeled
    shares are within the final lower-level tolerance.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Monthly response and levies

    The top panel compares the observed 2025 distribution with the modeled
    fixed-total response. The bottom panel shows the levies selected by the
    authority for that response.
    """)
    return


@app.cell(hide_code=True)
def _(
    Path,
    month_labels,
    month_numbers,
    observed_share,
    plt,
    selected_levy,
    selected_share,
):
    figure_dir = Path(__file__).resolve().parent / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    figure, (share_axis, levy_axis) = plt.subplots(2, 1, figsize=(6.5, 6.6), sharex=True)
    share_axis.plot(
        month_numbers,
        100.0 * observed_share,
        color="C0",
        linewidth=2,
        marker="o",
        label="Observed 2025",
    )
    share_axis.plot(
        month_numbers,
        100.0 * selected_share,
        color="C2",
        linewidth=2,
        marker="s",
        label="Modeled response",
    )
    share_axis.set(ylabel="Share of annual nights (%)")
    share_axis.grid(alpha=0.25)
    share_axis.legend(frameon=False, fontsize=12)

    levy_axis.bar(month_numbers, selected_levy, color="C3", alpha=0.8)
    levy_axis.set(
        xlabel="Month in 2025",
        ylabel="Levy (euros per night)",
        xticks=month_numbers,
        xticklabels=month_labels,
    )
    levy_axis.grid(axis="y", alpha=0.25)

    figure.tight_layout()
    figure_path = figure_dir / "tourism_seasonality_levy.pdf"
    figure.savefig(figure_path, bbox_inches="tight")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
