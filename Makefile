BLUE := \033[36m
BOLD := \033[1m
RESET := \033[0m

.DEFAULT_GOAL := help

.PHONY: sync
sync: ## install the development environment
	@printf "$(BLUE)Syncing development dependencies...$(RESET)\n"
	@uv sync --group dev

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
	@rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info

.PHONY: help
help: ## display this help message
	@printf "$(BOLD)Usage:$(RESET)\n"
	@printf "  make $(BLUE)<target>$(RESET)\n\n"
	@printf "$(BOLD)Targets:$(RESET)\n"
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { printf "  $(BLUE)%-15s$(RESET) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
