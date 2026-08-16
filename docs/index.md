# BLVPY: Disciplined Bilevel Programming in Python

BLVPY is a [CVXPY](https://www.cvxpy.org/) extension for modeling and
(approximately) solving optimistic bilevel optimization problems.
A bilevel problem contains an optimization problem inside another optimization problem, i.e.,

$$
    \begin{array}{ll}
        \text{minimize} & F_0(x, y)\\
        \text{subject to} & F_i(x, y) \leq 0, \quad i = 1, \ldots, m\\
        & y \in S(x),
    \end{array}
$$

where $x \in \mathbf{R}^n$ contains the upper variables and $y \in
\mathbf{R}^k$ contains lower variables constrained to belong to the set
$S(x)$. For a fixed $x \in \mathbf{R}^n$, the constraint set $S(x)$ is the
solution set of the following lower problem:

$$
    \begin{array}{rl}
      S(x) = \mathop{\rm argmin}_z & f_0(x, z)\\
      \text{subject to} & f_i(x, z) \leq 0, \quad i = 1, \ldots, p.
    \end{array}
$$

## Disciplined bilevel programming

A model is **disciplined bilevel programming (DBP)** compatible when:

* The upper objective and constraint functions $F_i \colon \mathbf{R}^n
  \times \mathbf{R}^k \to \mathbf{R}$, for $i=0,1,\ldots,m$, are
  [DNLP](https://www.cvxpy.org/tutorial/dnlp/index.html)-compatible with upper
  variables $x \in \mathbf{R}^n$ and lower variables $y \in \mathbf{R}^k$.
* The lower objective and constraint functions $f_i \colon \mathbf{R}^n
  \times \mathbf{R}^k \to \mathbf{R}$, for $i=0,1,\ldots,p$, are
  [DPP](https://www.cvxpy.org/tutorial/dpp/index.html)-compatible with lower
  variable $z \in \mathbf{R}^k$ (or $y \in \mathbf{R}^k$). Thus the lower
  problem is a disciplined convex program parameterized by $x \in
  \mathbf{R}^n$.

BLVPY models and solves this supported DBP subset with *optimistic
semantics*. When the lower problem has multiple minimizers, the upper problem
may select the one most favorable to its objective.

```{toctree}
:hidden:
:maxdepth: 2

installation
quickstart
modeling
solving
results
examples
troubleshooting
api
release-notes
```
