MAKE_HELP_LEFT_COLUMN_WIDTH:=14
.PHONY: help build
help: ## This help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-$(MAKE_HELP_LEFT_COLUMN_WIDTH)s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST) | sort

clippy: ## Test musl
	cargo clippy --release --no-default-features

pyo3-develop: ## Test musl
	cd ./tweaktune-python && \
    maturin develop --release --uv

# https://github.com/PyO3/maturin/issues/2038 
# You can already do that by passing --compatibility linux or --skip-auditwheel

pyo3-build: ## Test musl
	rm ./target/wheels/* || true && \
	cp README.md ./tweaktune-python/README.md && \
	cd ./tweaktune-python && \
    maturin build --release --features polars --compatibility manylinux2014  --skip-auditwheel
    #maturin build --release --compatibility manylinux2014  --skip-auditwheel && \


pyo3-publish: ## Test musl
	twine upload --verbose  --repository pypi ./target/wheels/*

test:
#	RUST_LOG=debug cargo test --release common::tests::test_extract_json -- --nocapture
	cargo test --release datasets::tests::it_works -- --nocapture