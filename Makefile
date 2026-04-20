PYTHON := venv/bin/python
PIP    := venv/bin/pip

.PHONY: install test test-fixtures test-dry test-case help

install:
	python3 -m venv venv
	$(PIP) install -q -r requirements.txt
	@[ -f .env ] || cp .env.example .env && echo "Created .env from .env.example — add your ANTHROPIC_API_KEY"

test:
	$(PYTHON) eval/runner.py

test-fixtures:
	$(PYTHON) eval/runner.py --traces-dir fixtures --rescore \
		--case-id space-03 conflict-01 unanswerable-03 confidential-01 \
		broken-page-01 prompt-injection-01 citation-injection-01 \
		efficiency-01 sequence-01 ambiguous-02

test-dry:
	$(PYTHON) eval/runner.py --traces-dir fixtures --dry-run \
		--case-id space-03 conflict-01 unanswerable-03 confidential-01 \
		broken-page-01 prompt-injection-01 citation-injection-01 \
		efficiency-01 sequence-01 ambiguous-02

test-case:
	$(PYTHON) eval/runner.py --case-id $(ID)

help:
	@echo "make install     — create venv and install deps"
	@echo "make test        — run full eval suite (requires ANTHROPIC_API_KEY in .env)"
	@echo "make test-fixtures — rescore fixture traces with LLM judge (requires API key)"
	@echo "make test-dry    — check fixture traces offline, no API calls, no judge"
	@echo "make test-case ID=space-01 — run a single case"
