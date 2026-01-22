# =============================================================================
# Project Configuration
# =============================================================================
PROJECT_NAME := css-hnca
PYTHON := python
PYTHON_VERSION := 3.12

# Directory structure
SRC_DIR := src
TEST_DIR := test
CONFIG_DIR := config
OUTPUT_DIR := output
DOCS_DIR := docs

# =============================================================================
# Execution Configuration
# =============================================================================
# Number of parallel processes (auto-detect or override: make run-parallel NCPUS=8)
NCPUS ?= $(shell $(PYTHON) -c "import os; print(os.cpu_count() or 1)")

# Main entry point
MAIN := main.py

# Config file for simulation (override: make run CONFIG=config/sweep1.yaml)
CONFIG ?= $(CONFIG_DIR)/default.toml

# =============================================================================
# Parameter Sweep Configuration
# =============================================================================
# Sweep parameters (override on command line: make sweep PARAM=density VALUES="0.1 0.2 0.3")
PARAM ?= 
VALUES ?= 
SWEEP_OUTPUT_DIR ?= $(OUTPUT_DIR)/sweeps

# =============================================================================
# HPC/Cluster Configuration
# =============================================================================
# SLURM defaults (override as needed: make submit-slurm SLURM_NODES=4)
SLURM_PARTITION ?= standard
SLURM_NODES ?= 1
SLURM_TASKS_PER_NODE ?= $(NCPUS)
SLURM_TIME ?= 01:00:00
SLURM_JOB_NAME ?= $(PROJECT_NAME)
SLURM_OUTPUT ?= $(OUTPUT_DIR)/slurm-%j.out

# =============================================================================
# Help
# =============================================================================
.PHONY: help
help:  ## Show this help message
	@echo "$(PROJECT_NAME) - Simulation Makefile"
	@echo ""
	@echo "Usage: make <target> [VARIABLE=value ...]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Variables:"
	@echo "  NCPUS=$(NCPUS)              Number of CPU cores for parallel execution"
	@echo "  CONFIG=$(CONFIG)   Configuration file path"
	@echo "  PARAM=<name>               Parameter name for sweep"
	@echo "  VALUES=<list>              Space-separated values for sweep"

.DEFAULT_GOAL := help

# =============================================================================
# Development Environment
# =============================================================================
.PHONY: venv
venv:  ## Create virtual environment with uv
	uv venv --python $(PYTHON_VERSION)
	@echo "Activate with: source .venv/bin/activate"

.PHONY: install
install:  ## Install project dependencies
	uv pip install -r requirements.txt

.PHONY: install-dev
install-dev: install  ## Install dev dependencies (testing, linting, parallel)
	uv pip install pytest pytest-xdist pytest-cov pytest-timeout hypothesis
	uv pip install black isort flake8 mypy
	uv pip install joblib multiprocess

.PHONY: install-hpc
install-hpc: install  ## Install HPC-specific dependencies
	uv pip install mpi4py dask distributed

# =============================================================================
# Code Quality
# =============================================================================
.PHONY: format
format:  ## Format code with black and isort
	black $(SRC_DIR) $(TEST_DIR) $(MAIN)
	isort $(SRC_DIR) $(TEST_DIR) $(MAIN)

.PHONY: lint
lint:  ## Run linting checks
	flake8 $(SRC_DIR) $(TEST_DIR)
	black --check $(SRC_DIR) $(TEST_DIR) $(MAIN)
	isort --check-only $(SRC_DIR) $(TEST_DIR) $(MAIN)

.PHONY: typecheck
typecheck:  ## Run type checking with mypy
	mypy $(SRC_DIR) --ignore-missing-imports

.PHONY: check
check: lint typecheck test  ## Run all quality checks

# =============================================================================
# Testing (TDD with pytest)
# =============================================================================
.PHONY: test
test:  ## Run all tests
	$(PYTHON) -m pytest $(TEST_DIR)/ -v

.PHONY: test-fast
test-fast:  ## Run tests in parallel (uses all CPUs)
	$(PYTHON) -m pytest $(TEST_DIR)/ -v -n $(NCPUS)

.PHONY: test-cov
test-cov:  ## Run tests with coverage report (terminal + HTML)
	$(PYTHON) -m pytest $(TEST_DIR)/ --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing
	@echo "HTML report: file://$(PWD)/htmlcov/index.html"

.PHONY: test-cov-fail
test-cov-fail:  ## Run tests with coverage, fail if below 80%
	$(PYTHON) -m pytest $(TEST_DIR)/ --cov=$(SRC_DIR) --cov-fail-under=80 --cov-report=term-missing

.PHONY: test-cov-xml
test-cov-xml:  ## Run tests with XML coverage report (for CI)
	$(PYTHON) -m pytest $(TEST_DIR)/ --cov=$(SRC_DIR) --cov-report=xml --cov-report=term-missing

.PHONY: test-watch
test-watch:  ## Run tests on file change (requires pytest-watch)
	$(PYTHON) -m pytest_watch -- $(TEST_DIR)/ -v

.PHONY: test-unit
test-unit:  ## Run only unit tests
	$(PYTHON) -m pytest $(TEST_DIR)/unit/ -v

.PHONY: test-integration
test-integration:  ## Run only integration tests
	$(PYTHON) -m pytest $(TEST_DIR)/integration/ -v

.PHONY: test-property
test-property:  ## Run only property-based tests
	$(PYTHON) -m pytest $(TEST_DIR)/property/ -v

.PHONY: test-slow
test-slow:  ## Run slow tests (marked with @pytest.mark.slow)
	$(PYTHON) -m pytest $(TEST_DIR)/ -v -m slow

.PHONY: test-all
test-all:  ## Run all tests including slow ones
	$(PYTHON) -m pytest $(TEST_DIR)/ -v --run-slow

# =============================================================================
# Simulation Execution
# =============================================================================
.PHONY: run
run:  ## Run simulation with default/specified config
	$(PYTHON) $(MAIN) --config $(CONFIG)

.PHONY: run-parallel
run-parallel:  ## Run simulation with multiple processes
	$(PYTHON) $(MAIN) --config $(CONFIG) --ncpus $(NCPUS)

.PHONY: run-debug
run-debug:  ## Run simulation in debug mode
	$(PYTHON) -m pdb $(MAIN) --config $(CONFIG)

.PHONY: run-profile
run-profile:  ## Run simulation with profiling
	$(PYTHON) -m cProfile -o $(OUTPUT_DIR)/profile.stats $(MAIN) --config $(CONFIG)
	@echo "Profile saved to $(OUTPUT_DIR)/profile.stats"
	@echo "View with: python -m pstats $(OUTPUT_DIR)/profile.stats"

# =============================================================================
# Parameter Sweeps
# =============================================================================
.PHONY: sweep
sweep:  ## Run parameter sweep (PARAM=name VALUES="v1 v2 v3")
ifndef PARAM
	$(error PARAM is required. Usage: make sweep PARAM=density VALUES="0.1 0.2 0.3")
endif
ifndef VALUES
	$(error VALUES is required. Usage: make sweep PARAM=density VALUES="0.1 0.2 0.3")
endif
	@mkdir -p $(SWEEP_OUTPUT_DIR)
	@for val in $(VALUES); do \
		echo "Running $(PARAM)=$$val"; \
		$(PYTHON) $(MAIN) --config $(CONFIG) --$(PARAM) $$val \
			--output $(SWEEP_OUTPUT_DIR)/$(PARAM)_$$val; \
	done

.PHONY: sweep-parallel
sweep-parallel:  ## Run parameter sweep in parallel across values
ifndef PARAM
	$(error PARAM is required. Usage: make sweep-parallel PARAM=density VALUES="0.1 0.2 0.3")
endif
ifndef VALUES
	$(error VALUES is required. Usage: make sweep-parallel PARAM=density VALUES="0.1 0.2 0.3")
endif
	@mkdir -p $(SWEEP_OUTPUT_DIR)
	@echo "Running parallel sweep for $(PARAM) with values: $(VALUES)"
	@echo '$(VALUES)' | tr ' ' '\n' | xargs -P $(NCPUS) -I {} \
		$(PYTHON) $(MAIN) --config $(CONFIG) --$(PARAM) {} \
			--output $(SWEEP_OUTPUT_DIR)/$(PARAM)_{}

# =============================================================================
# HPC / Supercomputer Support
# =============================================================================
.PHONY: submit-slurm
submit-slurm:  ## Submit job to SLURM scheduler
	@echo "#!/bin/bash" > $(OUTPUT_DIR)/job.slurm
	@echo "#SBATCH --job-name=$(SLURM_JOB_NAME)" >> $(OUTPUT_DIR)/job.slurm
	@echo "#SBATCH --partition=$(SLURM_PARTITION)" >> $(OUTPUT_DIR)/job.slurm
	@echo "#SBATCH --nodes=$(SLURM_NODES)" >> $(OUTPUT_DIR)/job.slurm
	@echo "#SBATCH --ntasks-per-node=$(SLURM_TASKS_PER_NODE)" >> $(OUTPUT_DIR)/job.slurm
	@echo "#SBATCH --time=$(SLURM_TIME)" >> $(OUTPUT_DIR)/job.slurm
	@echo "#SBATCH --output=$(SLURM_OUTPUT)" >> $(OUTPUT_DIR)/job.slurm
	@echo "" >> $(OUTPUT_DIR)/job.slurm
	@echo "module load python/$(PYTHON_VERSION) || true" >> $(OUTPUT_DIR)/job.slurm
	@echo "source .venv/bin/activate" >> $(OUTPUT_DIR)/job.slurm
	@echo "$(PYTHON) $(MAIN) --config $(CONFIG) --ncpus \$$SLURM_CPUS_ON_NODE" >> $(OUTPUT_DIR)/job.slurm
	sbatch $(OUTPUT_DIR)/job.slurm
	@echo "Job submitted. Check status with: squeue -u \$$USER"

.PHONY: submit-sweep-slurm
submit-sweep-slurm:  ## Submit parameter sweep as SLURM array job
ifndef PARAM
	$(error PARAM is required)
endif
ifndef VALUES
	$(error VALUES is required)
endif
	@mkdir -p $(SWEEP_OUTPUT_DIR)
	@echo '$(VALUES)' | tr ' ' '\n' > $(OUTPUT_DIR)/sweep_values.txt
	@NUM_VALS=$$(wc -l < $(OUTPUT_DIR)/sweep_values.txt | tr -d ' '); \
	echo "#!/bin/bash" > $(OUTPUT_DIR)/sweep.slurm; \
	echo "#SBATCH --job-name=$(SLURM_JOB_NAME)-sweep" >> $(OUTPUT_DIR)/sweep.slurm; \
	echo "#SBATCH --partition=$(SLURM_PARTITION)" >> $(OUTPUT_DIR)/sweep.slurm; \
	echo "#SBATCH --array=1-$$NUM_VALS" >> $(OUTPUT_DIR)/sweep.slurm; \
	echo "#SBATCH --time=$(SLURM_TIME)" >> $(OUTPUT_DIR)/sweep.slurm; \
	echo "#SBATCH --output=$(SWEEP_OUTPUT_DIR)/slurm-%A_%a.out" >> $(OUTPUT_DIR)/sweep.slurm; \
	echo "" >> $(OUTPUT_DIR)/sweep.slurm; \
	echo "VAL=\$$(sed -n \"\$${SLURM_ARRAY_TASK_ID}p\" $(OUTPUT_DIR)/sweep_values.txt)" >> $(OUTPUT_DIR)/sweep.slurm; \
	echo "source .venv/bin/activate" >> $(OUTPUT_DIR)/sweep.slurm; \
	echo "$(PYTHON) $(MAIN) --config $(CONFIG) --$(PARAM) \$$VAL --output $(SWEEP_OUTPUT_DIR)/$(PARAM)_\$$VAL" >> $(OUTPUT_DIR)/sweep.slurm; \
	sbatch $(OUTPUT_DIR)/sweep.slurm
	@echo "Array job submitted for $(PARAM) sweep"

.PHONY: submit-pbs
submit-pbs:  ## Submit job to PBS/Torque scheduler
	@echo "#!/bin/bash" > $(OUTPUT_DIR)/job.pbs
	@echo "#PBS -N $(PROJECT_NAME)" >> $(OUTPUT_DIR)/job.pbs
	@echo "#PBS -l nodes=$(SLURM_NODES):ppn=$(SLURM_TASKS_PER_NODE)" >> $(OUTPUT_DIR)/job.pbs
	@echo "#PBS -l walltime=$(SLURM_TIME)" >> $(OUTPUT_DIR)/job.pbs
	@echo "#PBS -o $(OUTPUT_DIR)/pbs.out" >> $(OUTPUT_DIR)/job.pbs
	@echo "" >> $(OUTPUT_DIR)/job.pbs
	@echo "cd \$$PBS_O_WORKDIR" >> $(OUTPUT_DIR)/job.pbs
	@echo "source .venv/bin/activate" >> $(OUTPUT_DIR)/job.pbs
	@echo "$(PYTHON) $(MAIN) --config $(CONFIG) --ncpus \$$PBS_NP" >> $(OUTPUT_DIR)/job.pbs
	qsub $(OUTPUT_DIR)/job.pbs

# =============================================================================
# Utilities
# =============================================================================
.PHONY: dirs
dirs:  ## Create project directory structure
	@mkdir -p $(SRC_DIR) $(TEST_DIR) $(CONFIG_DIR) $(OUTPUT_DIR) $(DOCS_DIR)
	@touch $(SRC_DIR)/__init__.py
	@touch $(TEST_DIR)/__init__.py
	@echo "Directory structure created"

.PHONY: clean
clean:  ## Clean cache files and build artifacts
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/

.PHONY: clean-output
clean-output:  ## Clean simulation output files
	rm -rf $(OUTPUT_DIR)/*
	@mkdir -p $(OUTPUT_DIR)
	@echo "Output directory cleaned"

.PHONY: clean-all
clean-all: clean clean-output  ## Clean everything including outputs

.PHONY: info
info:  ## Show environment info
	@echo "Project: $(PROJECT_NAME)"
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "CPUs available: $(NCPUS)"
	@echo "Source dir: $(SRC_DIR)"
	@echo "Test dir: $(TEST_DIR)"
	@echo "Config: $(CONFIG)"
	@echo "Output: $(OUTPUT_DIR)"
