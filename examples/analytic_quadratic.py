"""A bilevel quadratic with the analytic solution x = y = 0.

Requires BLVPY's native IPOPT dependency.
"""

import cvxpy as cp

from blvpy import BilevelProblem, LowerProblem


def main() -> None:
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")

    # For each x, the unique lower solution is y = x. Substitution into the
    # upper objective gives (x - 1)^2 + (x + 1)^2, minimized by x = 0.
    lower = LowerProblem(cp.Minimize(cp.square(y - x)), parameters=[x])
    problem = BilevelProblem(
        outer_objective=cp.Minimize(cp.square(x - 1.0) + cp.square(y + 1.0)),
        lower_problem=lower,
    )

    result = problem.solve()
    print(f"status: {result.status}")
    print(f"x: {x.value:.6f} (expected 0)")
    print(f"y: {y.value:.6f} (expected 0)")
    print(f"complementarity: {result.complementarity:.3e}")


if __name__ == "__main__":
    main()
