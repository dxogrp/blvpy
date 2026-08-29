# Examples

The gallery consists of standalone [Marimo](https://marimo.io/) notebooks for example applications.

## Modeling primers

- {example}`Analytic quadratic <analytic_quadratic.py>`
  compares epsilon-relaxed solutions with a closed-form lower response.
- {example}`Optimistic linear program <optimistic_lp.py>`
  shows how the upper problem selects among tied lower optimizers.
- {example}`Parameter-dependent SOCP <parameter_dependent_socp.py>`
  derives a geometric response and inspects parameter-dependent canonical data.

- {example}`Best-of local optima <best_of_local_optima.py>`
  compares a deterministic local solve with several complete randomized runs.

## Applications

- {example}`Ridge hyperparameter selection <ridge_hyperparameter.py>`
  selects a training penalty using validation loss.
- {example}`Demand-response pricing <demand_response_pricing.py>`
  designs a time-of-use price while anticipating flexible energy use.
- {example}`Renewable-capacity planning <renewable_capacity_planning.py>`
  trades capacity investment against lower-level electricity dispatch.
- {example}`Traffic tolling <traffic_tolling.py>`
  selects tolls while anticipating a congestion equilibrium.
- {example}`Stackelberg port security <stackelberg_port_security.py>`
  allocates limited patrol coverage against a best-responding attacker.
- {example}`Planar truss sizing <planar_truss_sizing.py>`
  allocates member areas while anticipating elastic equilibrium.

Install and open the gallery from a repository checkout:

```shell
make sync-examples
make marimo
```
