.PHONY: dev-install run-tests run-linter run-linter-fix run-pylint run-mypy run-deptry

dev-install:
	sudo apt update
	poetry install
	chmod +x functions/dssp
	chmod +x functions/DAlphaBall.gcc

run-tests:
	poetry run pytest --cov=functions --cov-report=term-missing

run-linter:
	poetry run ruff check .

run-linter-fix:
	poetry run ruff check . --fix

run-pylint:
	poetry run pylint functions tests

run-mypy:
	poetry run mypy --explicit-package-bases functions

run-deptry:
	poetry run deptry .
