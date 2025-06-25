CYAN='\x1b[38;5;116m'
NC='\033[0m'

.PHONY: format lint static-analysis test ci-checks

format:
	@echo
	@echo -e ${CYAN} ---- Formatting ---- ${NC}
	poetry run ruff format

lint:
	@echo 
	@echo -e ${CYAN} ---- Linting ---- ${NC}
	poetry run ruff check . --fix

static-analysis:
	@echo 
	@echo -e ${CYAN} ---- Static Analysis ---- ${NC}
	poetry run mypy .

test:
	@echo
	@echo -e ${CYAN} ---- Tests ---- ${NC}
	poetry run pytest .

test-coverage:
	@echo 
	@echo -e ${CYAN} ---- Test Coverage ---- ${NC}
	poetry run pytest . --cov='.' --cov-report=term --cov-report xml:coverage.xml

ci-checks: format lint static-analysis test-coverage