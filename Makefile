.PHONY: lint format

lint:
	ruff check .
	ruff format . --check

format:
	ruff format .
	ruff check . --select I001 --fix
	ruff check . --select F401 --fix
