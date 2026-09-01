UV ?= uv

.PHONY: bootstrap format lint typecheck test check conformance

bootstrap:
	$(UV) sync

format:
	$(UV) run ruff format --check .

lint:
	$(UV) run ruff check .

typecheck:
	$(UV) run mypy

test:
	$(UV) run pytest

check: format lint typecheck test
	git diff --exit-code

conformance:
	@echo "real RiffDB MLflow conformance is not implemented yet" >&2
	@exit 1

