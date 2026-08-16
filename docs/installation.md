# Installation

BLVPY requires Python 3.12 or newer, CVXPY 1.9 or newer, and a native IPOPT installation.
IPOPT and its Python binding, `cyipopt`, are mandatory runtime dependencies.

1. Install IPOPT by following the official
   [IPOPT installation guide](https://coin-or.github.io/Ipopt/INSTALL.html).
2. Install BLVPY:

   ```shell
   pip install blvpy
   ```

Confirm that the native binding can be imported:

```shell
python -c "import cyipopt, blvpy; print(blvpy.__version__)"
```
