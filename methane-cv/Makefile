CYAN='\x1b[38;5;116m'
NC='\033[0m'

.PHONY: format lint static-analysis test ci-checks

format:
	@echo
	@echo -e ${CYAN} ---- Formatting ---- ${NC}
	ruff format --check

format-fix:
	@echo
	@echo -e ${CYAN} ---- Formatting ---- ${NC}
	ruff format
	nbstripout --drop-empty-cells --extra-keys="metadata.kernelspec.display_name metadata.kernelspec.name" sbr_2025/notebooks/S2_SBR_exploration.ipynb
	nbstripout --drop-empty-cells --extra-keys="metadata.kernelspec.display_name metadata.kernelspec.name" sbr_2025/notebooks/Landsat_SBR_exploration.ipynb

lint:
	@echo
	@echo -e ${CYAN} ---- Linting ---- ${NC}
	ruff check src/azure_wrap/ml_client_utils.py
	ruff check src/data/
	ruff check src/inference/
	ruff check src/tests/conftest.py
	ruff check src/tests/generate_test_data.py
	ruff check src/tests/test_azure.py
	ruff check src/tests/test_dataset.py
	ruff check src/tests/test_emit.py
	ruff check src/tests/test_generate.py
	ruff check src/tests/test_geospatial_utils.py
	ruff check src/tests/test_inference.py
	ruff check src/tests/test_landsat.py
	ruff check src/tests/test_metrics.py
	ruff check src/tests/test_sentinel2.py
	ruff check src/tests/test_sim_plumes.py
	ruff check src/tests/test_transformations.py
	ruff check src/tests/utils/test_quantification_utils.py
	ruff check src/tests/utils/test_radtran_utils.py
	ruff check src/training/
	ruff check src/utils/
	ruff check src/utils/geospatial.py
	ruff check src/utils/parameters.py
	ruff check src/utils/utils.py
	ruff check src/validation/
	ruff check sbr_2025/
	ruff check scripts/

lint-fix:
	@echo
	@echo -e ${CYAN} ---- Linting ---- ${NC}
	ruff check src/azure_wrap/ml_client_utils.py --fix
	ruff check src/data/ --fix
	ruff check src/inference/ --fix
	ruff check src/tests/conftest.py --fix
	ruff check src/tests/generate_test_data.py --fix
	ruff check src/tests/test_azure.py --fix
	ruff check src/tests/test_dataset.py --fix
	ruff check src/tests/test_emit.py --fix
	ruff check src/tests/test_generate.py --fix
	ruff check src/tests/test_geospatial_utils.py --fix
	ruff check src/tests/test_inference.py --fix
	ruff check src/tests/test_landsat.py --fix
	ruff check src/tests/test_metrics.py --fix
	ruff check src/tests/test_sentinel2.py --fix
	ruff check src/tests/test_sim_plumes.py --fix
	ruff check src/tests/test_transformations.py --fix
	ruff check src/tests/utils/test_quantification_utils.py --fix
	ruff check src/tests/utils/test_radtran_utils.py --fix
	ruff check src/training/ --fix
	ruff check src/utils/ --fix
	ruff check src/utils/geospatial.py --fix
	ruff check src/utils/parameters.py --fix
	ruff check src/utils/utils.py --fix
	ruff check src/validation/ --fix
	ruff check sbr_2025/ --fix
	ruff check scripts/ --fix

static-analysis:
	@echo
	@echo -e ${CYAN} ---- Static Analysis ---- ${NC}
	mypy src/azure_wrap/ml_client_utils.py --follow-imports=skip
	# mypy does not do recursive searches so we need to find the files ourselves
	find src/data -name "*.py" -not -path "*/.ipynb_checkpoints/*" | xargs mypy --follow-imports=skip
	mypy src/inference/*.py --follow-imports=skip
	mypy src/tests/conftest.py --follow-imports=skip
	mypy src/tests/generate_test_data.py --follow-imports=skip
	mypy src/tests/test_azure.py --follow-imports=skip
	mypy src/tests/test_dataset.py --follow-imports=skip
	mypy src/tests/test_emit.py --follow-imports=skip
	mypy src/tests/test_generate.py --follow-imports=skip
	mypy src/tests/test_geospatial_utils.py --follow-imports=skip
	mypy src/tests/test_inference.py --follow-imports=skip
	mypy src/tests/test_landsat.py --follow-imports=skip
	mypy src/tests/test_metrics.py --follow-imports=skip
	mypy src/tests/test_sentinel2.py --follow-imports=skip
	mypy src/tests/test_sim_plumes.py --follow-imports=skip
	mypy src/tests/test_transformations.py --follow-imports=skip
	mypy src/tests/utils/test_quantification_utils.py --follow-imports=skip
	mypy src/tests/utils/test_radtran_utils.py --follow-imports=skip
	mypy src/training/*.py --follow-imports=skip
	mypy src/utils/*.py --follow-imports=skip
	mypy src/validation/*.py --follow-imports=skip
	mypy sbr_2025/*.py --follow-imports=skip
	mypy sbr_2025/**/*.py --follow-imports=skip
	# add fixed files to this single invocation of mypy
	mypy src/validation/fpr_dt_pipeline.py \
	src/validation/azure_run_fpr_dt_pipeline.py \
	src/utils/profiling.py \
	src/tests/test_profiling.py \
	src/data/common/data_item.py \
	src/data/common/sim_plumes.py \
	src/data/azure_run_data_generation.py \
	src/data/generate.py \
	src/data/dataset.py \
	src/data/landsat_data.py \
	src/data/generation/plumes/*.py \
	sbr_2025/*.py \
	sbr_2025/**/*.py \
	scripts/save_model_for_production_release.py

tests:
	@echo
	@echo -e ${CYAN} ---- Tests ---- ${NC}
	pytest . --benchmark-skip

test-coverage:
	@echo
	@echo -e ${CYAN} ---- Test Coverage ---- ${NC}
	pytest . --benchmark-skip --cov='.' --cov-report=term --cov-report xml:coverage.xml

benchmarks:
	@echo
	@echo -e ${CYAN} ---- Benchmarks ---- ${NC}
	pytest . --benchmark-only

# Fix formatting and linting where possible
fix: format-fix lint-fix

# Run the same checks as in CI
ci-checks: format lint static-analysis tests

strip-sbr-notebooks:
	nbstripout --drop-empty-cells --extra-keys="metadata.kernelspec.display_name metadata.kernelspec.name" sbr_2025/notebooks/*.ipynb
