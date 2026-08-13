"""Inspect continuation attempts and reproduce multistart sampling.

If a requested contraction fails, BLVpy records the failed attempt and inserts
an intermediate epsilon automatically. Reusing the seed reproduces starts.
Requires BLVpy's native IPOPT dependency.
"""

import cvxpy as cp

from blvpy import BilevelProblem, LowerProblem


def build() -> BilevelProblem:
    x = cp.Variable(name="x", bounds=[-2.0, 2.0])
    y = cp.Variable(name="y")
    lower = LowerProblem(cp.Minimize(cp.square(y - x)), parameters=[x])
    return BilevelProblem(
        cp.Minimize(cp.square(x - 0.25) + cp.square(y + 0.5)),
        lower,
    )


def main() -> None:
    first = build().solve(starts=8, seed=91, contraction=0.1)
    second = build().solve(starts=8, seed=91, contraction=0.1)

    print(f"accepted epsilon history: {first.epsilon_history}")
    print(f"all attempts/retries: {first.attempted_epsilon_history}")
    print(f"reproducible objectives: {first.objective:.8f}, {second.objective:.8f}")


if __name__ == "__main__":
    main()
