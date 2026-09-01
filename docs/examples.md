# Examples

The gallery consists of standalone [Marimo](https://marimo.io/) notebooks for example applications.
Each link opens an executed, non-interactive HTML snapshot containing the notebook code and outputs.

## Modeling primers

- {example}`Analytic quadratic <analytic_quadratic>`
  compares epsilon-relaxed solutions with a closed-form lower response.
- {example}`Optimistic linear program <optimistic_lp>`
  shows how the upper problem selects among tied lower optimizers.
- {example}`Parameter-dependent SOCP <parameter_dependent_socp>`
  derives a geometric response and inspects parameter-dependent canonical data.

- {example}`Best-of local optima <best_of_local_optima>`
  compares a deterministic local solve with several complete randomized runs.

## Applications

- {example}`Ridge hyperparameter selection <ridge_hyperparameter>`
  selects a training penalty using validation loss.
- {example}`Demand-response pricing <demand_response_pricing>`
  designs a time-of-use price while anticipating flexible energy use.
- {example}`Renewable-capacity planning <renewable_capacity_planning>`
  trades capacity investment against lower-level electricity dispatch.
- {example}`Traffic tolling <traffic_tolling>`
  selects tolls while anticipating a congestion equilibrium.
- {example}`Stackelberg port security <stackelberg_port_security>`
  allocates limited patrol coverage against a best-responding attacker.
- {example}`Planar truss sizing <planar_truss_sizing>`
  allocates member areas while anticipating elastic equilibrium.
- {example}`DC motor MPC tuning <dc_motor_mpc_tuning>`
  learns control-cost weights while anticipating a constrained MPC response.

For live interaction, install and open the gallery from a repository checkout:

```shell
make sync-examples
make marimo
```
