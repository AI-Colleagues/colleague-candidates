.PHONY: lint format test

lint:
	ruff check .
	ruff format . --check

format:
	ruff format .
	ruff check . --select I001 --fix
	ruff check . --select F401 --fix

test:
	pytest --cov --cov-report term-missing --cov-fail-under=100
