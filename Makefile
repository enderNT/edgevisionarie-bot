PYTHON := python3
VENV_DIR := .venv
VENV_PYTHON := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_DIR)/bin/pip
APP_HOST ?= 0.0.0.0
APP_PORT ?= 9000
NGROK ?= ngrok
NGROK_AUTHTOKEN ?=
NGROK_DOMAIN ?=
DSPY_TASK ?= route

ifneq (,$(wildcard .env))
include .env
export
endif

.PHONY: venv install run ngrok webhook-url \
	dspy-compile dspy-eval \
	dspy-route dspy-discovery-call dspy-rag dspy-conversation dspy-summary \
	dspy-eval-route dspy-eval-discovery-call dspy-eval-rag dspy-eval-conversation dspy-eval-summary

venv:
	@test -d $(VENV_DIR) || ($(PYTHON) -m venv $(VENV_DIR) && echo "✓ Virtual environment creado en $(VENV_DIR)")
	@test -d $(VENV_DIR) && echo "✓ Virtual environment ya existe en $(VENV_DIR)"

install: venv
	@$(VENV_PIP) install -e ".[dev]"

run:
	@$(VENV_DIR)/bin/uvicorn app.main:create_app --factory --host $(APP_HOST) --port $(APP_PORT) --reload

ngrok:
	@if ! command -v $(NGROK) >/dev/null 2>&1; then \
		echo "ngrok no esta instalado en PATH"; \
		exit 1; \
	fi
	@if [ -n "$(NGROK_DOMAIN)" ]; then \
		$(NGROK) http $(APP_PORT) $(if $(NGROK_AUTHTOKEN),--authtoken $(NGROK_AUTHTOKEN),) --url $(NGROK_DOMAIN); \
	else \
		$(NGROK) http $(APP_PORT) $(if $(NGROK_AUTHTOKEN),--authtoken $(NGROK_AUTHTOKEN),); \
	fi

webhook-url:
	@$(PYTHON) -c 'import json, urllib.request; \
url = "http://127.0.0.1:4040/api/tunnels"; \
data = json.load(urllib.request.urlopen(url, timeout=2)); \
tunnels = [t.get("public_url") for t in data.get("tunnels", []) if t.get("public_url", "").startswith("https://")]; \
print(f"{tunnels[0]}/webhooks/chatwoot") if tunnels else (_ for _ in ()).throw(SystemExit("No encontre un tunel HTTPS activo en ngrok."))'

dspy-compile:
	@$(VENV_PYTHON) scripts/dspy_compile.py $(DSPY_TASK)

dspy-eval:
	@$(VENV_PYTHON) scripts/dspy_eval.py $(DSPY_TASK)

dspy-route:
	@$(MAKE) dspy-compile DSPY_TASK=route

dspy-discovery-call:
	@$(MAKE) dspy-compile DSPY_TASK=discovery_call

dspy-rag:
	@$(MAKE) dspy-compile DSPY_TASK=rag

dspy-conversation:
	@$(MAKE) dspy-compile DSPY_TASK=conversation

dspy-summary:
	@$(MAKE) dspy-compile DSPY_TASK=summary

dspy-eval-route:
	@$(MAKE) dspy-eval DSPY_TASK=route

dspy-eval-discovery-call:
	@$(MAKE) dspy-eval DSPY_TASK=discovery_call

dspy-eval-rag:
	@$(MAKE) dspy-eval DSPY_TASK=rag

dspy-eval-conversation:
	@$(MAKE) dspy-eval DSPY_TASK=conversation

dspy-eval-summary:
	@$(MAKE) dspy-eval DSPY_TASK=summary
