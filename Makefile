MAKE_HELP_LEFT_COLUMN_WIDTH:=14
.PHONY: help build
help: ## This help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-$(MAKE_HELP_LEFT_COLUMN_WIDTH)s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST) | sort


pyo3-develop: ## Test musl
	cd ./crates/tweaktune-py && \
    maturin develop --release --uv

# https://github.com/PyO3/maturin/issues/2038 
# You can already do that by passing --compatibility linux or --skip-auditwheel

pyo3-build: ## Test musl
	cd ./crates/tweaktune-py && \
    maturin build --release --compatibility manylinux2014  --skip-auditwheel


pyo3-publish: ## Test musl
	twine upload --verbose  --repository pypi ./target/wheels/*