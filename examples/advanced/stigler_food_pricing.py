import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Food Pricing with the Stigler Diet

    This example uses the Stigler diet dataset to study a bilevel pricing
    problem: a merchant sets food markups and anticipates the consumer's
    least-cost basket. A baseline linear program identifies the foods whose
    prices the merchant may change and supplies the initial consumer basket.
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

    The offline table is derived from the 77 foods and nine nutrient columns
    in the pinned
    [OR-Tools Stigler example](https://github.com/google/or-tools/blob/100f66e6242ab8bf8d32feb8f3bf086db66ae2b5/ortools/linear_solver/samples/stigler_diet.py).
    OR-Tools distributes that sample under Apache-2.0. Every nutrient
    coefficient is normalized per 1939 dollar spent on a food; the calorie
    column is measured in *thousands* of kcal per dollar. The package units
    are included for readable result labels.
    """)
    return


@app.cell
def _(Path, csv, np):
    data_path = Path(__file__).resolve().parents[1] / "_shared" / "data" / "stigler_diet.csv"
    with data_path.open(encoding="utf-8", newline="") as data_file:
        food_rows = list(csv.DictReader(data_file))

    nutrient_columns = (
        "calories_thousand_kcal_per_dollar",
        "protein_g_per_dollar",
        "calcium_g_per_dollar",
        "iron_mg_per_dollar",
        "vitamin_a_kiu_per_dollar",
        "vitamin_b1_mg_per_dollar",
        "vitamin_b2_mg_per_dollar",
        "niacin_mg_per_dollar",
        "vitamin_c_mg_per_dollar",
    )
    nutrient_minimums = np.array([3.0, 70.0, 0.8, 12.0, 5.0, 1.8, 2.7, 18.0, 75.0])
    nutrient_matrix = np.array(
        [[float(food_row[column]) for food_row in food_rows] for column in nutrient_columns],
        dtype=float,
    )
    food_names = np.array([food_row["food"] for food_row in food_rows], dtype=object)
    food_units = np.array([food_row["unit"] for food_row in food_rows], dtype=object)

    assert food_names.size == 77
    assert nutrient_matrix.shape == (9, 77)
    assert np.all(np.isfinite(nutrient_matrix))
    return food_names, food_units, nutrient_matrix, nutrient_minimums


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Baseline least-cost basket

    Let $x_i\geq0$ be daily dollars spent on food $i$, let
    $A\in\mathbf{R}^{9\times77}$ contain nutrient quantities per dollar, and
    let $b$ contain the daily nutrient minimums. The baseline problem is

    \[
    \begin{array}{ll}
    \mathop{\mathrm{minimize}}_x & \mathbf{1}^Tx \\
    \mathop{\mathrm{subject\ to}} & Ax\succeq b,\quad x\succeq0.
    \end{array}
    \]

    We solve it independently with CVXPY. Its active foods define the set
    of prices available to the merchant, and its solution initializes the
    consumer basket in the bilevel model.
    """)
    return


@app.cell
def _(cp, food_names, np, nutrient_matrix, nutrient_minimums):
    number_of_foods = food_names.size
    baseline_variable = cp.Variable(number_of_foods, nonneg=True, name="baseline_food_spending")
    baseline_problem = cp.Problem(
        cp.Minimize(cp.sum(baseline_variable)),
        [nutrient_matrix @ baseline_variable >= nutrient_minimums],
    )
    baseline_problem.solve(
        solver=cp.CLARABEL,
        tol_gap_abs=1e-10,
        tol_gap_rel=1e-10,
        tol_feas=1e-10,
    )
    baseline_food_spending = np.asarray(baseline_variable.value, dtype=float)
    baseline_annual_cost = 365.0 * float(baseline_problem.value)
    baseline_active_indices = np.flatnonzero(baseline_food_spending > 1e-7)

    assert baseline_problem.status == cp.OPTIMAL
    assert np.min(nutrient_matrix @ baseline_food_spending - nutrient_minimums) >= -1e-8
    return (
        baseline_active_indices,
        baseline_annual_cost,
        baseline_food_spending,
        number_of_foods,
    )


@app.cell(hide_code=True)
def _(
    baseline_active_indices,
    baseline_annual_cost,
    baseline_food_spending,
    food_names,
    food_units,
    mo,
):
    baseline_rows = "\n".join(
        f"| {food_names[index]} | {food_units[index]} | ${365.0 * baseline_food_spending[index]:.6f} |"
        for index in baseline_active_indices
    )
    mo.md(f"""
    ## Baseline basket

    | active food | package unit | annual expenditure |
    | --- | --- | ---: |
    {baseline_rows}

    The independently computed baseline costs **${baseline_annual_cost:.7f}
    per year**. The active foods above define the markup index set $I$ used
    below.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bilevel pricing model

    Let $I$ index the foods in the baseline basket and let
    $m\in\mathbf{R}^{|I|}$ be their fractional markups. A selection matrix
    $P\in\mathbf{R}^{77\times|I|}$ inserts those markups into the full price
    vector. The consumer solves

    \[
    \begin{array}{ll}
    S(m)=\mathop{\mathrm{argmin}}_x
      & (\mathbf{1}+Pm)^Tx \\
    \mathop{\mathrm{subject\ to}}
      & Ax\succeq b, \\
      & x\succeq0.
    \end{array}
    \]

    Anticipating that response, the merchant solves

    \[
    \begin{array}{ll}
    \mathop{\mathrm{minimize}}_{m,x}
      & -m^Tx_I+0.005\lVert m\rVert_2^2 \\
    \mathop{\mathrm{subject\ to}}
      & 0\preceq m\preceq\mathbf{1},\quad x\in S(m).
    \end{array}
    \]

    Here $m^Tx_I$ is daily markup revenue measured in 1939 dollars. The small
    quadratic term regularizes the markup vector. BLVPY uses optimistic
    semantics, so the returned basket is the merchant-favorable response among
    those within the final lower-level tolerance.
    """)
    return


@app.cell
def _(
    BilevelProblem,
    LowerProblem,
    baseline_active_indices,
    baseline_food_spending,
    cp,
    np,
    number_of_foods,
    nutrient_matrix,
    nutrient_minimums,
):
    number_of_markups = baseline_active_indices.size
    selection_matrix = np.zeros((number_of_foods, number_of_markups))
    selection_matrix[baseline_active_indices, np.arange(number_of_markups)] = 1.0

    markup = cp.Variable(number_of_markups, name="markup")
    food_spending = cp.Variable(number_of_foods, nonneg=True, name="food_spending")
    markup.value = np.full(number_of_markups, 0.2)
    food_spending.value = baseline_food_spending

    consumer_price = 1.0 + selection_matrix @ markup
    consumer_problem = LowerProblem(
        cp.Minimize(consumer_price @ food_spending),
        [nutrient_matrix @ food_spending >= nutrient_minimums],
        parameters=[markup],
    )
    merchant_problem = BilevelProblem(
        cp.Minimize(-markup @ food_spending[baseline_active_indices] + 0.005 * cp.sum_squares(markup)),
        consumer_problem,
        upper_constraints=[markup >= 0.0, markup <= 1.0],
    )

    merchant_problem.validate()
    return food_spending, markup, merchant_problem, selection_matrix


@app.cell
def _(merchant_problem):
    result = merchant_problem.solve(
        epsilon_initial=1e-2,
        epsilon_target=1e-5,
        solver_options={"max_iter": 1000},
        verbose=False,
    )
    diagnostics = merchant_problem.gap_diagnostics(result)
    return diagnostics, result


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Independent response checks

    At the selected markups, we solve the consumer LP again with CVXPY. We
    also maximize markup revenue over baskets whose marked cost is within the
    final epsilon of that exact lower optimum. The second LP checks BLVPY's
    optimistic response directly rather than assuming a unique basket.
    """)
    return


@app.cell
def _(
    baseline_active_indices,
    cp,
    food_spending,
    markup,
    np,
    number_of_foods,
    nutrient_matrix,
    nutrient_minimums,
    result,
    selection_matrix,
):
    optimized_markup = np.asarray(markup.value, dtype=float)
    optimized_food_spending = np.asarray(food_spending.value, dtype=float)
    marked_price = 1.0 + selection_matrix @ optimized_markup

    fixed_response_variable = cp.Variable(number_of_foods, nonneg=True, name="fixed_markup_response")
    fixed_response_problem = cp.Problem(
        cp.Minimize(marked_price @ fixed_response_variable),
        [nutrient_matrix @ fixed_response_variable >= nutrient_minimums],
    )
    fixed_response_problem.solve(
        solver=cp.CLARABEL,
        tol_gap_abs=1e-10,
        tol_gap_rel=1e-10,
        tol_feas=1e-10,
    )
    fixed_optimal_cost = float(fixed_response_problem.value)
    returned_consumer_cost = float(marked_price @ optimized_food_spending)

    optimistic_variable = cp.Variable(number_of_foods, nonneg=True, name="optimistic_response")
    optimistic_problem = cp.Problem(
        cp.Maximize(optimized_markup @ optimistic_variable[baseline_active_indices]),
        [
            nutrient_matrix @ optimistic_variable >= nutrient_minimums,
            marked_price @ optimistic_variable <= fixed_optimal_cost + result.final_epsilon,
        ],
    )
    optimistic_problem.solve(
        solver=cp.CLARABEL,
        tol_gap_abs=1e-10,
        tol_gap_rel=1e-10,
        tol_feas=1e-10,
    )

    selected_daily_revenue = float(optimized_markup @ optimized_food_spending[baseline_active_indices])
    optimistic_daily_revenue = float(optimistic_problem.value)
    response_cost_gap = returned_consumer_cost - fixed_optimal_cost
    optimistic_revenue_gap = optimistic_daily_revenue - selected_daily_revenue

    assert result.succeeded, result.message
    assert fixed_response_problem.status == cp.OPTIMAL
    assert optimistic_problem.status == cp.OPTIMAL
    assert -1e-7 <= response_cost_gap <= result.final_epsilon + 1e-7
    assert -1e-7 <= optimistic_revenue_gap <= 2.0 * result.final_epsilon
    assert np.min(nutrient_matrix @ optimized_food_spending - nutrient_minimums) >= -1e-6
    assert np.all((optimized_markup >= -1e-8) & (optimized_markup <= 1.0 + 1e-8))
    return (
        marked_price,
        optimistic_revenue_gap,
        optimized_food_spending,
        optimized_markup,
        response_cost_gap,
        selected_daily_revenue,
    )


@app.cell(hide_code=True)
def _(
    diagnostics,
    mo,
    optimistic_revenue_gap,
    response_cost_gap,
    result,
    selected_daily_revenue,
):
    mo.md(rf"""
    ## Pricing result

    | quantity | value |
    | --- | ---: |
    | status | `{result.status}` |
    | annual markup revenue | ${365.0 * selected_daily_revenue:.6f} |
    | upper objective | {result.objective:.8f} |
    | final epsilon | {result.final_epsilon:.3e} |
    | maximum lifted residual | {result.residuals.max_violation:.3e} |
    | signed source gap | {diagnostics.source_gap:.3e} |
    | fixed-markup consumer cost gap | {response_cost_gap:.3e} |
    | optimistic revenue check gap | {optimistic_revenue_gap:.3e} |

    The merchant raises prices on the baseline basket and induces a small
    substitution toward evaporated milk. The revenue shown is attached to an
    epsilon-optimal, leader-favorable consumer response; it should not be read
    as the outcome of an exact tie-breaking rule.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Markups and basket substitution

    The left panel reports the optimized fractional markups. The right
    panel compares annual contributions to the baseline basket cost with
    contributions to the marked-price basket. It includes every food used by
    either solution, making the substitution visible.
    """)
    return


@app.cell
def _(
    baseline_active_indices,
    baseline_food_spending,
    food_names,
    marked_price,
    np,
    optimized_food_spending,
):
    basket_indices = np.flatnonzero((baseline_food_spending > 1e-7) | (optimized_food_spending > 1e-7))
    display_name_map = {
        "Wheat Flour (Enriched)": "Wheat flour",
        "Evaporated Milk (can)": "Evaporated milk",
        "Liver (Beef)": "Beef liver",
        "Navy Beans, Dried": "Navy beans",
    }
    basket_labels = [display_name_map.get(food_names[index], food_names[index]) for index in basket_indices]
    markup_labels = [display_name_map.get(food_names[index], food_names[index]) for index in baseline_active_indices]
    baseline_annual_contributions = 365.0 * baseline_food_spending[basket_indices]
    optimized_annual_contributions = 365.0 * marked_price[basket_indices] * optimized_food_spending[basket_indices]
    return (
        baseline_annual_contributions,
        basket_labels,
        markup_labels,
        optimized_annual_contributions,
    )


@app.cell(hide_code=True)
def _(
    Path,
    baseline_annual_contributions,
    basket_labels,
    markup_labels,
    np,
    optimized_annual_contributions,
    optimized_markup,
    plt,
):
    figure_dir = Path(__file__).resolve().parent / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, (markup_axis, basket_axis) = plt.subplots(1, 2, figsize=(9.5, 4.8))
    markup_positions = np.arange(len(markup_labels))
    markup_axis.barh(markup_positions, 100.0 * optimized_markup, color="C3", alpha=0.85)
    markup_axis.set(
        xlabel="Markup (percent)",
        yticks=markup_positions,
        yticklabels=markup_labels,
    )
    markup_axis.invert_yaxis()

    basket_positions = np.arange(len(basket_labels))
    bar_width = 0.38
    basket_axis.barh(
        basket_positions - bar_width / 2.0,
        baseline_annual_contributions,
        height=bar_width,
        label="Baseline",
        color="grey",
        alpha=0.8,
    )
    basket_axis.barh(
        basket_positions + bar_width / 2.0,
        optimized_annual_contributions,
        height=bar_width,
        label="Marked price",
        color="C2",
        alpha=0.85,
    )
    basket_axis.set(
        xlabel="Annual basket contribution (1939 dollars)",
        yticks=basket_positions,
        yticklabels=basket_labels,
    )
    basket_axis.invert_yaxis()
    basket_axis.legend(frameon=False, fontsize=12)

    fig.tight_layout()
    figure_path = figure_dir / "stigler_food_pricing.pdf"
    fig.savefig(figure_path, bbox_inches="tight")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
