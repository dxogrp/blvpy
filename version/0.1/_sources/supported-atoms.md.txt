# Supported atoms

BLVPY checks the lower source expression tree as well as CVXPY's final cone
program. An atom appearing in this table is necessary but not sufficient for
model support; see also {ref}`structural-requirements`.

The table is exhaustive for source nodes that BLVPY audits directly. Real
affine nodes are accepted as a class; every audited nonlinear node appears in
its own row.

```{list-table}
:header-rows: 1
:widths: 36 46 18

* - CVXPY syntax
  - Mathematical expression
  - Conic form
* - Any real affine expression
  - $Ax+b$
  - Affine
* - `cp.abs(x)`
  - $|x|$
  - LP
* - `cp.cummax(x, axis=...)`
  - $\left(\max_{j\leq i}x_j\right)_i$
  - LP
* - `cp.dotsort(x, W)`
  - $\left\langle\operatorname{sort}(\operatorname{vec}x),
    \operatorname{sort}(\operatorname{vec}W)\right\rangle$
  - LP
* - `cp.geo_mean(x, p=..., approx=True)`
  - $\displaystyle\prod_i x_i^{w_i}$, where
    $w=p/(\mathbf{1}^{\mathsf T}p)$
  - SOC
* - `cp.huber(x, M)`
  - $\begin{cases}x^2,&|x|\leq M,\\2M|x|-M^2,&|x|>M\end{cases}$
  - SOC
* - `cp.max(x, axis=...)`
  - $\max_i x_i$
  - LP
* - `cp.maximum(x, y)`
  - $\left(\max\{x_i,y_i\}\right)_i$
  - LP
* - `cp.min(x, axis=...)`
  - $\min_i x_i$
  - LP
* - `cp.minimum(x, y)`
  - $\left(\min\{x_i,y_i\}\right)_i$
  - LP
* - `cp.norm1(x)`
  - $\displaystyle\sum_i |x_i|$
  - LP
* - `cp.norm_inf(x)`
  - $\displaystyle\max_i |x_i|$
  - LP
* - `cp.pnorm(x, p, approx=True)`
  - $\begin{cases}(\sum_i |x_i|^p)^{1/p},&p>1,\\
    (\sum_i x_i^p)^{1/p},&p<1,\ x\geq0\end{cases}$
  - SOC
* - `cp.power(x, p, approx=True)`
  - $x^p$ elementwise
  - SOC
* - `cp.quad_form(x, P)`
  - $x^TPx$
  - SOC
* - `cp.quad_over_lin(x, y)`
  - $\|x\|_2^2/y$
  - SOC
* - `cp.sum_largest(x, k)`
  - $\displaystyle\sum_{i=1}^k x_{[i]}$
  - LP
```

Vector-valued expressions are flattened where needed. Sorting in `dotsort`
uses the same order for both arguments, while $x_{[i]}$ denotes the $i$th
largest entry in `sum_largest`. Reduction indices follow the selected axis;
`abs`, `huber`, and `power` act elementwise. The atom data $W$, $M$, $p$, $P$,
and $k$ must satisfy CVXPY's usual constantness, domain, curvature, and DPP
rules.

## Exact rational representations

CVXPY's approximate geometric-mean, p-norm, and power atoms use rational SOC
representations. BLVPY accepts them only when CVXPY reports a finite
`approx_error` exactly equal to zero. A tiny nonzero value is still an
approximation and is rejected without a numerical tolerance.

Note that using `approx=False` selects CVXPY's exact power-cone representation instead.
However, power cones are outside BLVPY's current cone policy, so those forms remain
unsupported even though they do not use rational approximation.
