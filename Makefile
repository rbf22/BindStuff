.PHONY: install dev-install download-params install-system-deps run-tests run-linter run-linter-fix run-pylint run-mypy run-deptry

install: install-system-deps dev-install download-params
	@echo "Installation complete. Please activate your poetry shell with 'poetry shell'"

dev-install:
	sudo apt update
	poetry install
	chmod +x functions/dssp
	chmod +x functions/DAlphaBall.gcc

download-params:
	@echo "Downloading AlphaFold2 parameters..."
	mkdir -p params
	wget -q -O params/alphafold_params_2022-12-06.tar "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar"
	tar -xvf params/alphafold_params_2022-12-06.tar -C params
	rm params/alphafold_params_2022-12-06.tar

install-system-deps:
	@echo "Installing system dependencies..."
	sudo apt-get update
	sudo apt-get install -y libgfortran5 ffmpeg

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

Jules: install-system-deps
	chmod +x functions/dssp
	chmod +x functions/DAlphaBall.gcc
