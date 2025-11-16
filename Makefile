MAKE_HELP_LEFT_COLUMN_WIDTH:=14
.PHONY: help build
help: ## This help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-$(MAKE_HELP_LEFT_COLUMN_WIDTH)s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST) | sort

clippy: ## Run Clippy linter
	cargo clippy --release --no-default-features -- -D warnings

clippy-fix: ## Auto-fix Clippy warnings
	cargo clippy --release --no-default-features --fix --allow-dirty

pyo3-develop: ## Build PyO3 package in development mode
	cd ./tweaktune-python && \
    maturin develop --release --uv

# https://github.com/PyO3/maturin/issues/2038
# You can already do that by passing --compatibility linux or --skip-auditwheel

pyo3-build: ## Build PyO3 package for release
	rm ./target/wheels/* || true && \
	cp README.md ./tweaktune-python/README.md && \
	cd ./tweaktune-python && \
    maturin build --release --compatibility manylinux2014  --skip-auditwheel


pyo3-publish: ## Publish PyO3 package to PyPI
	twine upload --verbose  --repository pypi ./target/wheels/*

fmt: ## Format Rust code
	cargo fmt --all

fmt-check: ## Check Rust formatting
	cargo fmt --all -- --check

test: ## Run Rust tests
#	RUST_LOG=debug cargo test --release common::tests::test_extract_json -- --nocapture
#	cargo test --release datasets::tests::it_works -- --nocapture
	export RUSTFLAGS="-C link-args=-lpython3.10" && \
	cargo test --release steps::tests::schema_validate2 -- --nocapture

# Python formatting
py-fmt: ## Format Python code
	cd tweaktune-python && black . && isort .

py-fmt-check: ## Check Python formatting
	cd tweaktune-python && black --check . && isort --check-only .

# Python linting
py-lint: ## Lint Python code with Ruff
	cd tweaktune-python && ruff check .

py-lint-fix: ## Auto-fix Python linting issues
	cd tweaktune-python && ruff check --fix .

# Python type checking
py-typecheck: ## Type check Python code with mypy
	cd tweaktune-python && mypy tweaktune --ignore-missing-imports

# Python tests with coverage
py-test: ## Run Python tests with coverage
	pytest --cov=tweaktune --cov-report=term --cov-report=html

py-test-simple: ## Run Python tests without coverage
	pytest

# All Python checks
py-check: py-fmt-check py-lint py-typecheck py-test ## Run all Python quality checks

# All Rust checks
rust-check: fmt-check clippy test ## Run all Rust quality checks

# Run everything
check-all: rust-check py-check ## Run all quality checks (Rust + Python)