# Run the container as the invoking user so files it writes are theirs.
export UID := $(shell id -u)
export GID := $(shell id -g)

RUN = docker compose run --rm quietsky

.PHONY: build test test-all lint typecheck check fixtures shell

build:
	docker compose build

# Fast tests: synthetic data and the cropped M13 fixtures in tests/data.
test:
	$(RUN) pytest -m "not nas and not slow"

# Also the slow tests and the full-frame comparisons against PixInsight,
# which read the NAS.
test-all:
	$(RUN) pytest

lint:
	$(RUN) sh -c "ruff check . && ruff format --check ."

typecheck:
	$(RUN) mypy

check: lint typecheck test

# Regenerate tests/data/m13 from the full data set.
fixtures:
	$(RUN) python tools/make_m13_fixtures.py

shell:
	$(RUN) bash
