import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # An analytic quadratic bilevel problem

    This small example introduces BLVPY with a problem whose exact bilevel
    solution can be derived by hand. The upper variable is $x$ and the lower
    decision is $y$.
    """)
    return


@app.cell
def _():
    import cvxpy as cp
    import marimo as mo
    import numpy as np

    from blvpy import BilevelProblem, LowerProblem

    return BilevelProblem, LowerProblem, cp, mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problem formulation

    We solve the optimistic bilevel problem

    \[
    \begin{array}{ll}
    \mathop{\rm minimize}_{x,y} & (x-1)^2 + (y+1)^2 \\
    \mathop{\rm subject\ to} &
      y \in \mathop{\rm argmin}_{z}\ (z-x)^2.
    \end{array}
    \]

    For every fixed $x$, the unique lower solution is $y=x$. Substitution
    gives $2x^2+2$, so the exact bilevel solution is $x=y=0$ with objective
    value $2$.

    BLVPY solves an $\epsilon$-relaxed primal-dual reformulation. At a nonzero
    $\epsilon>0$ the returned point can differ slightly from the exact solution,
    and the difference vanishes as $\epsilon \to 0$. In this example the
    relaxed solution is available analytically:

    \[
      x_\epsilon=\frac{\sqrt{\epsilon}}{2},\qquad
      y_\epsilon=-\frac{\sqrt{\epsilon}}{2}.
    \]
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Specify and solve the model

    `LowerProblem(parameters=[x])` declares that the lower-level optimizer
    treats the upper variable $x$ as fixed data. The original lower variable
    $y$ is shared with the upper objective, which gives optimistic semantics.
    """)
    return


@app.cell
def _(BilevelProblem, LowerProblem, cp):
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")

    lower = LowerProblem(
        cp.Minimize(cp.square(y - x)),
        parameters=[x],
    )
    problem = BilevelProblem(
        cp.Minimize(cp.square(x - 1.0) + cp.square(y + 1.0)),
        lower,
    )
    return problem, x, y


@app.cell(hide_code=True)
def _(mo):
    epsilon_slider = mo.ui.slider(
        steps=[
            1e-12,
            1e-11,
            1e-10,
            1e-9,
            1e-8,
            1e-7,
            1e-6,
            1e-5,
            1e-4,
            1e-3,
            1e-2,
            1e-1,
        ],
        value=1e-6,
        debounce=True,
        show_value=True,
        label=r"Target relaxation $\epsilon_{\mathrm{target}}$",
    )
    mo.vstack(
        [
            mo.md(r"""
            ## Explore the relaxation

            Drag the slider and release it to solve again. Its logarithmically
            spaced values show how the relaxed point approaches the exact
            bilevel solution as $\epsilon_{\mathrm{target}}$ decreases.

            The smallest targets also probe numerical precision: once the
            analytic displacement falls below the nonlinear solver's practical
            accuracy, the returned point may no longer track the square-root
            formula digit for digit.
            """),
            epsilon_slider,
        ]
    )
    return (epsilon_slider,)


@app.cell
def _(epsilon_slider, np, problem, x, y):
    epsilon_target = float(epsilon_slider.value)
    result = problem.solve(epsilon_target=epsilon_target, verbose=False)
    diagnostics = problem.gap_diagnostics(result)
    relaxed_shift = np.sqrt(epsilon_target) / 2.0
    expected_objective = 2.0 * (1.0 - relaxed_shift) ** 2

    assert result.succeeded, result.message
    np.testing.assert_allclose(
        [float(np.asarray(x.value)), float(np.asarray(y.value))],
        [relaxed_shift, -relaxed_shift],
        atol=3e-3,
        rtol=0.0,
    )
    assert abs(result.objective - expected_objective) <= 5e-3, "Upper objective does not match the analytic value."
    return diagnostics, epsilon_target, relaxed_shift, result


@app.cell(hide_code=True)
def _(diagnostics, epsilon_target, mo, relaxed_shift, result, x, y):
    mo.md(rf"""
    ## Result and interpretation

    - Status: `{result.status}`
    - Target relaxation: $\epsilon_{{\mathrm{{target}}}}={epsilon_target:.1e}$
    - Final epsilon: ${result.final_epsilon:.1e}$
    - Upper variable: $x={float(x.value):.6f}$
    - Lower response: $y={float(y.value):.6f}$
    - Analytic relaxed point:
      $(x_\epsilon,y_\epsilon)=({relaxed_shift:.6f},{-relaxed_shift:.6f})$
    - Upper objective: ${result.objective:.6f}$
    - Maximum lifted violation: ${result.residuals.max_violation:.3e}$
    - Complementarity: ${result.complementarity:.3e}$
    - Independently evaluated lower source gap: ${diagnostics.source_gap:.3e}$

    The numerical point agrees with the analytic relaxed solution. Move the
    slider toward smaller values to see it converge to the exact bilevel point
    $(0,0)$ and objective value $2$.
    """)
    return


if __name__ == "__main__":
    app.run()
