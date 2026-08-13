"""An SOCP whose constraint matrix depends on the upper variable.

Requires BLVPy's native IPOPT dependency.
"""

import cvxpy as cp

from blvpy import BilevelProblem


def main() -> None:
    x = cp.Variable(name="x", bounds=[0.5, 2.0])
    y = cp.Variable(2, name="y")
    t = cp.Variable(name="t")
    x_lower = cp.Parameter(nonneg=True, name="x_lower")

    # The row [x, 1] in the lower linear inequality varies with the upper
    # variable. The norm epigraph contributes a second-order cone block.
    lower = cp.Problem(
        cp.Minimize(t),
        [cp.norm(y, 2) <= t, x_lower * y[0] + y[1] >= 1.0],
    )
    problem = BilevelProblem(
        outer_objective=cp.Minimize(cp.square(x - 1.25) + 0.1 * t),
        lower_problem=lower,
        parameter_map={x_lower: x},
    )

    result = problem.solve(starts=10, seed=7)
    print(f"status: {result.status}")
    print(f"x: {x.value:.6f}")
    print(f"y: {y.value}")
    print(f"t: {t.value:.6f}")
    print(f"complementarity: {result.complementarity:.3e}")


if __name__ == "__main__":
    main()
