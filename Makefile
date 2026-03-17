PYTHON := python3
VENV_DIR := .venv
VENV_PYTHON := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_DIR)/bin/pip

.PHONY: install run

install:
	@test -d $(VENV_DIR) || $(PYTHON) -m venv $(VENV_DIR)
	@$(VENV_PIP) install -e ".[dev]"

run:
	@$(VENV_DIR)/bin/uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000 --reload
