BLUE := \033[36m
BOLD := \033[1m
RESET := \033[0m

.DEFAULT_GOAL := help

.PHONY: sync
sync: ## install the development environment
	@printf "$(BLUE)Syncing development dependencies...$(RESET)\n"
	@uv sync --frozen --group dev

.PHONY: sync-examples
sync-examples: ## install development and example dependencies
	@printf "$(BLUE)Syncing example dependencies...$(RESET)\n"
	@uv sync --frozen --group dev --group examples

.PHONY: sync-docs
sync-docs: ## install development and documentation dependencies
	@printf "$(BLUE)Syncing documentation dependencies...$(RESET)\n"
	@uv sync --frozen --group dev --group docs

.PHONY: _sync-docs-examples
_sync-docs-examples:
	@printf "$(BLUE)Syncing documentation and example dependencies...$(RESET)\n"
	@uv sync --frozen --group dev --group docs --group examples

.PHONY: marimo
marimo: sync-examples ## open the Marimo example gallery
	@printf "$(BLUE)Opening Marimo examples...$(RESET)\n"
	@uv run --frozen --group examples marimo edit examples

.PHONY: check-examples
check-examples: sync-examples ## statically check every Marimo example
	@printf "$(BLUE)Checking Marimo examples...$(RESET)\n"
	@uv run --frozen --group examples marimo check --strict examples/*.py

.PHONY: docs
docs: _sync-docs-examples ## build and serve the Sphinx documentation
	@printf "$(BLUE)Building Sphinx documentation...$(RESET)\n"
	@uv run --frozen --group docs --group examples sphinx-build -b html docs docs/_build/html
	@printf "$(BLUE)Exporting executed examples; this may take several minutes...$(RESET)\n"
	@uv run --frozen --group docs --group examples python scripts/export_examples.py \
		--source-dir examples --output-dir docs/_build/html/examples
	@printf "$(BLUE)Serving documentation at http://127.0.0.1:8000...$(RESET)\n"
	@uv run --frozen --group docs --group examples python -m http.server --directory docs/_build/html 8000

.PHONY: check-docs
check-docs: sync-docs ## strictly validate the Sphinx documentation
	@printf "$(BLUE)Checking Sphinx documentation...$(RESET)\n"
	@uv run --frozen --group docs python -m pytest tests/test_docs.py
	@uv run --frozen --group docs sphinx-build -M clean docs docs/_build
	@uv run --frozen --group docs sphinx-build -W --keep-going -n -b html docs docs/_build/html

.PHONY: test
test: sync ## run the full test suite
	@printf "$(BLUE)Running tests...$(RESET)\n"
	@uv run python -m pytest tests

.PHONY: lint
lint: sync ## check formatting and lint rules
	@printf "$(BLUE)Running Ruff checks...$(RESET)\n"
	@uv run ruff check .
	@uv run ruff format --check .

.PHONY: fmt
fmt: sync ## format and automatically fix lint violations
	@printf "$(BLUE)Formatting Python sources...$(RESET)\n"
	@uv run ruff check --fix .
	@uv run ruff format .

.PHONY: build
build: sync ## build source and wheel distributions
	@printf "$(BLUE)Building package artifacts...$(RESET)\n"
	@uv build

.PHONY: clean
clean: ## remove local build and test artifacts
	@printf "$(BLUE)Cleaning local artifacts...$(RESET)\n"
	@rm -rf .pytest_cache .ruff_cache build dist docs/_build src/*.egg-info

.PHONY: help
help: ## display this help message
	@printf "$(BOLD)Usage:$(RESET)\n"
	@printf "  make $(BLUE)<target>$(RESET)\n\n"
	@printf "$(BOLD)Targets:$(RESET)\n"
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { printf "  $(BLUE)%-15s$(RESET) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
