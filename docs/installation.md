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

## Development setup

BLVPY manages its development environment with [uv](https://docs.astral.sh/uv/).
Before setting up the repository, you should install uv and IPOPT on your development system.

1. Clone the repository:

   ```shell
   git clone https://github.com/dxogrp/blvpy.git
   cd blvpy
   ```

2. Create the virtual environment and install the locked development
   dependencies:

   ```shell
   make sync
   ```

   which runs the frozen uv workflow.

Run the primary contributor checks with:

```shell
make test
make lint
```

The example and documentation toolchains use separate locked dependency
groups. Install them only when needed:

```shell
make sync-examples  # Marimo and Matplotlib
make sync-docs      # Sphinx and MyST
```

For dependency updates, you should update `pyproject.toml`, run `uv lock`, and commit the resulting `uv.lock` change.
