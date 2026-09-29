# Examples

The example collections consist of standalone [Marimo](https://marimo.io/)
notebooks. Each link opens an executed, non-interactive HTML snapshot containing
the notebook code and outputs.

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
  trades capacity investment against lower-level electricity dispatch using a
  compact synthetic daily profile.
- {example}`Traffic tolling <traffic_tolling>`
  selects tolls while anticipating a congestion equilibrium.
- {example}`Stackelberg port security <stackelberg_port_security>`
  allocates limited patrol coverage against a best-responding attacker.
- {example}`Planar truss sizing <planar_truss_sizing>`
  allocates member areas while anticipating elastic equilibrium.
- {example}`DC motor MPC tuning <dc_motor_mpc_tuning>`
  learns control-cost weights while anticipating a constrained MPC response.
- {example}`Clean-equipment rebates <clean_equipment_rebate>`
  design a rebate while anticipating
  a producer's clean and fossil input choices under Cobb--Douglas production.
- {example}`Carbon-tax abatement <carbon_tax_abatement>`
  sets a carbon tax while anticipating sector-level abatement with
  exponential costs.

## Advanced examples

Advanced examples are published as a separate nested collection and may use
advanced features or take longer to run.

### Real data and literature

- {example}`Stigler food pricing <advanced/stigler_food_pricing>`
  reproduces the classical minimum-cost diet before adding a Stackelberg
  pricing model.
- {example}`Iris SVM regularization <advanced/iris_svm_regularization>`
  reproduces Fisher's discriminant scores and tunes a modern classifier on
  the corrected Iris data.
- {example}`Renewable-capacity planning with German grid data <advanced/renewable_capacity_planning_smard>`
  applies the capacity model to an observed day of German demand, wind, and
  solar generation.

### Polishing workflows

- {example}`Ridge-polishing decision <advanced/ridge_polishing>`
  retrains a validation-selected ridge model at its fixed penalty, then lets
  the user choose between lower-level feasibility and a better relaxed
  validation objective.
- {example}`Low-carbon blend polishing <advanced/low_carbon_blend_polishing>`
  designs material rebates for a constrained producer, rejects a coarse
  polished response, tightens continuation, and explicitly adopts the
  candidate that passes a quantitative deployment gate.

For live interaction, install and open the complete example workspace from a
repository checkout:

```shell
make marimo
```
