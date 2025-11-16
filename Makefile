MAKE_HELP_LEFT_COLUMN_WIDTH:=14
.PHONY: help build
help: ## This help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-$(MAKE_HELP_LEFT_COLUMN_WIDTH)s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST) | sort

setup-dev: ## Set up development environment with all tools
	@echo "Setting up development environment..."
	@if ! command -v uv > /dev/null; then \
		echo "Error: uv is not installed. Install it with: pip install uv"; \
		exit 1; \
	fi
	uv venv .venv
	. .venv/bin/activate && \
	cd tweaktune-python && \
	uv pip install maturin pytest && \
	maturin develop --release --uv --extras "dev,db,arrow"
	@echo "✓ Development environment ready!"
	@echo "Run 'source .venv/bin/activate' to activate the virtual environment"

clippy: ## Run Clippy linter
	cargo clippy --release --no-default-features -- -D warnings -A clippy::upper_case_acronyms

clippy-fix: ## Auto-fix Clippy warnings
	cargo clippy --release --no-default-features --fix --allow-dirty -- -A clippy::upper_case_acronyms

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
	@if [ -d .venv ]; then \
		. .venv/bin/activate && cd tweaktune-python && black . && isort .; \
	else \
		echo "Error: Virtual environment not found. Run 'uv venv .venv && uv pip install -e tweaktune-python[dev]' first"; \
		exit 1; \
	fi

py-fmt-check: ## Check Python formatting
	@if [ -d .venv ]; then \
		. .venv/bin/activate && cd tweaktune-python && black --check . && isort --check-only .; \
	else \
		echo "Error: Virtual environment not found. Run 'uv venv .venv && uv pip install -e tweaktune-python[dev]' first"; \
		exit 1; \
	fi

# Python linting
py-lint: ## Lint Python code with Ruff
	@if [ -d .venv ]; then \
		. .venv/bin/activate && cd tweaktune-python && ruff check .; \
	else \
		echo "Error: Virtual environment not found. Run 'uv venv .venv && uv pip install -e tweaktune-python[dev]' first"; \
		exit 1; \
	fi

py-lint-fix: ## Auto-fix Python linting issues
	@if [ -d .venv ]; then \
		. .venv/bin/activate && cd tweaktune-python && ruff check --fix .; \
	else \
		echo "Error: Virtual environment not found. Run 'uv venv .venv && uv pip install -e tweaktune-python[dev]' first"; \
		exit 1; \
	fi

# Python type checking
py-typecheck: ## Type check Python code with mypy
	@if [ -d .venv ]; then \
		. .venv/bin/activate && cd tweaktune-python && mypy tweaktune --ignore-missing-imports; \
	else \
		echo "Error: Virtual environment not found. Run 'uv venv .venv && uv pip install -e tweaktune-python[dev]' first"; \
		exit 1; \
	fi

# Python tests with coverage
py-test: ## Run Python tests with coverage
	@if [ -d .venv ]; then \
		. .venv/bin/activate && pytest --cov=tweaktune --cov-report=term --cov-report=html; \
	else \
		echo "Error: Virtual environment not found. Run 'uv venv .venv && uv pip install -e tweaktune-python[dev]' first"; \
		exit 1; \
	fi

py-test-simple: ## Run Python tests without coverage
	@if [ -d .venv ]; then \
		. .venv/bin/activate && pytest; \
	else \
		echo "Error: Virtual environment not found. Run 'uv venv .venv && uv pip install -e tweaktune-python[dev]' first"; \
		exit 1; \
	fi

# All Python checks
py-check: py-fmt-check py-lint py-typecheck py-test ## Run all Python quality checks

# All Rust checks
rust-check: fmt-check clippy test ## Run all Rust quality checks

# Run everything
check-all: rust-check py-check ## Run all quality checks (Rust + Python)