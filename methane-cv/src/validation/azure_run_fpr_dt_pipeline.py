"""Dispatch the FPR and Detection Threshold pipeline to Azure.

```bash
python src/validation/azure_run_fpr_dt_pipeline.py \
    model_id=1054 \
    satellite=s2 \
    experiment_name=<experiment name>
```
"""

import shutil
import tempfile

import hydra
from azure.ai.ml import command
from omegaconf import DictConfig

from src.azure_wrap.ml_client_utils import ensure_compute_target_exists, initialize_ml_client
from src.utils import IGNORE_PATTERNS, METHANE_CV_ENV, REPO_ROOT
from src.utils.parameters import SatelliteID


@hydra.main(version_base=None, config_path="config", config_name="fpr_dt_config")
def main(config: DictConfig) -> None:
    """Dispatch the FPR to DT pipeline job to Azure."""
    assert isinstance(config.model_id, int)
    if config.satellite_id != SatelliteID.S2:
        raise ValueError(
            f"Currently only Sentinel-2 is supported to run this script. Got: {config.satellite_id}. "
            "We will need to update the script to download the correct bands and use the "
            "correct SWIR16 and SWIR22 band names."
        )

    ml_client = initialize_ml_client()

    ensure_compute_target_exists(ml_client, config.compute)
    custom_env = ml_client.environments.get(name=METHANE_CV_ENV, version=1)

    command_str = (
        "cd methane-cv; "
        "conda run -n methane-cv pip install --no-deps -r requirements.txt; "
        "conda run -n methane-cv --no-capture-output python -m src.validation.fpr_dt_pipeline "
        f"val_parquet_folder_path={config.val_parquet_folder_path} "
        f"experiment_name={config.experiment_name} "
        f"model_id={config.model_id} "
        f"crop_size={config.crop_size} "
        f"ncrops={config.ncrops} "
        f"hapi_data_path={config.hapi_data_path} "
        "azure_cluster=true"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        shutil.copytree(
            REPO_ROOT,
            tmpdir,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(*IGNORE_PATTERNS),
        )

        command_job = command(
            code=tmpdir,
            command=command_str,
            environment=custom_env,
            compute=config.compute,
            experiment_name=f"kpi-pipeline-{config.satellite_id}-{config.experiment_name}",
            display_name=None,
            description=f"model_id: {config.model_id}",
        )

        returned_job = ml_client.jobs.create_or_update(command_job)
        print(f"Job name: {returned_job.name}")
        print(f"Job status URL: {returned_job.studio_url}")


if __name__ == "__main__":
    main()
