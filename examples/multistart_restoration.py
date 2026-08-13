"""Automatic multistart with an initially violated upper constraint.

Requires BLVpy's native IPOPT dependency.
"""

import cvxpy as cp

from blvpy import BilevelProblem, LowerProblem


def main() -> None:
    x = cp.Variable(name="x", bounds=[-3.0, 3.0])
    y = cp.Variable(name="y")

    lower = LowerProblem(cp.Minimize(cp.square(y - x)), parameters=[x])
    problem = BilevelProblem(
        cp.Minimize(cp.square(x - 1.0) + cp.square(y - 1.0)),
        lower,
        # Some sampled starts violate this and invoke feasibility restoration.
        outer_constraints=[x + y >= 1.5],
    )
    result = problem.solve(starts=12, seed=23, restoration=True)

    print(f"status: {result.status}")
    print(f"successful starts: {sum(start.residuals is not None for start in result.starts)}")
    print(f"x, y: {float(x.value):.6f}, {float(y.value):.6f}")
    print(f"residuals: {result.residuals}")


if __name__ == "__main__":
    main()
