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
* - `cp.entr(x)`
  - $-x\log x$ elementwise
  - EXP
* - `cp.exp(x)`
  - $e^x$ elementwise
  - EXP
* - `cp.geo_mean(x, p=..., approx=True)`
  - $\displaystyle\prod_i x_i^{w_i}$, where
    $w=p/(\mathbf{1}^{\mathsf T}p)$
  - SOC
* - `cp.huber(x, M)`
  - $\begin{cases}x^2,&|x|\leq M,\\2M|x|-M^2,&|x|>M\end{cases}$
  - SOC
* - `cp.kl_div(x, y)`
  - $x\log(x/y)-x+y$ elementwise
  - EXP
* - `cp.log(x)`
  - $\log x$ elementwise
  - EXP
* - `cp.log1p(x)`
  - $\log(1+x)$ elementwise
  - EXP
* - `cp.log_sum_exp(x, axis=..., keepdims=...)`
  - $\log\left(\sum_i e^{x_i}\right)$
  - EXP
* - `cp.logistic(x)`
  - $\log(1+e^x)$ elementwise
  - EXP
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
* - `cp.pnorm(x, p, approx=False)`
  - $\begin{cases}(\sum_i |x_i|^p)^{1/p},&p>1,\\
    (\sum_i x_i^p)^{1/p},&p<1,\ x\geq0\end{cases}$
  - LP / SOC / P3D
* - `cp.power(x, p, approx=True)`
  - $x^p$ elementwise
  - SOC
* - `cp.power(x, p, approx=False)`
  - $x^p$ elementwise
  - Affine / SOC / P3D
* - `cp.quad_form(x, P)`
  - $x^TPx$
  - SOC
* - `cp.quad_over_lin(x, y)`
  - $\|x\|_2^2/y$
  - SOC
* - `cp.rel_entr(x, y)`
  - $x\log(x/y)$ elementwise
  - EXP
* - `cp.sum_largest(x, k)`
  - $\displaystyle\sum_{i=1}^k x_{[i]}$
  - LP
* - `cp.xexp(x)`
  - $xe^x$ elementwise
  - EXP / SOC
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

Using `approx=False` for `cp.power` or `cp.pnorm` selects CVXPY's exact
representation, which uses 3D power cones for the general case and simpler
cones for special exponents. BLVPY supports those forms. Direct scalar and
vectorized `cp.PowCone3D` constraints are also supported. Exact `cp.geo_mean`
and direct `cp.PowConeND` constraints produce generalized power cones and
remain unsupported. Convenience wrappers that do not expose an `approx`
argument keep their normal CVXPY representation.

Note that here the exactness describes the canonical graph representation, not the
numerical cone-distance estimates; see
{ref}`nonlinear-cone-distance-estimates`.

## Exponential representations

The listed exponential-family atoms use exact exponential-cone graphs.
BLVPY also supports direct scalar, vector, and matrix `cp.ExpCone`
constraints. Vector and matrix entries become three-row exponential-cone
blocks in CVXPY's canonical element order.

BLVPY audits exactness from the constructed expression graph. Convenience
functions such as `cp.loggamma` and `cp.log_normcdf` return compositions of
primitive atoms rather than distinct source nodes, so BLVPY evaluates those
compositions according to their constituent atoms. Any approximation embodied
in such a composition remains part of the modeled expression. Explicit
quadrature constraints such as `RelEntrConeQuad` and `OpRelEntrConeQuad`
remain unsupported.
