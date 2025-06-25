"""Shared fixtures between test_dataset.py and test_generate.py are in this module to avoid overloading conftest.py."""

import datetime
import logging
from pathlib import Path

from azure.core.exceptions import HttpResponseError

from src.azure_wrap.azure_path import AzureBlobPath
from src.azure_wrap.ml_client_utils import initialize_ml_client
from src.data.common.sim_plumes import PlumeType
from src.data.emit_data import EMITL2AMaskLabel
from src.data.generate import SATELLITE_CLASSES, DataGenerationConfig
from src.data.generation.base import BaseDataGeneration
from src.utils.parameters import (
    CROP_SIZE,
    EMIT_HAPI_DATA_PATH,
    EMIT_PSF_DEFAULT,
    LANDSAT_HAPI_DATA_PATH,
    LANDSAT_PSF_DEFAULT,
    S2_B12_DEFAULT,
    S2_HAPI_DATA_PATH,
    SatelliteID,
)
from src.utils.utils import initialize_s3_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s (%(filename)s:%(lineno)d)",
    datefmt="%H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)


def build_satellite_generation_params(
    sat_key: SatelliteID, plume_type: PlumeType, out_dir: Path, in_dir: Path, storage_options: dict | None
) -> dict:
    """Specify the test parameters for a specific satellite generation."""
    # Create base config
    # FIXME: either make satellite specific config pydantic models, or remove quality_thresholds from base config
    quality_thresholds = {
        SatelliteID.S2: {
            "main_crop": (0.0, 1.0),
            "main_tile_cloud": (0.0, 0.98),
            "main_tile_nodata": (0.0, 0.95),
            "reference_tile_cloud_shadow": (0.0, 0.8),
            "reference_crop_cloud_shadow": (0.0, 0.05),
        },
        SatelliteID.EMIT: {"main_crop": (0.0, 0.3)},
        SatelliteID.LANDSAT: {
            "main_crop": (0.0, 1.0),
            "main_tile_cloud": (0.0, 0.98),
            "main_tile_nodata": (0.0, 0.95),
            "reference_tile_cloud_shadow": (0.0, 0.8),
            "reference_crop_cloud_shadow": (0.0, 0.05),
        },
    }

    plume_catalog = str(in_dir / "plumes" / "catalog_condensed.json")

    sat_psf_sigmas = {
        SatelliteID.S2: S2_B12_DEFAULT,
        SatelliteID.EMIT: EMIT_PSF_DEFAULT,
        SatelliteID.LANDSAT: LANDSAT_PSF_DEFAULT,
    }
    sat_spatial_resolutions = {
        SatelliteID.S2: 20,
        SatelliteID.EMIT: 60,
        SatelliteID.LANDSAT: 30,
    }
    # Concentration rescale factors will tend to vary by both satellite (given different instrument sensitivities)
    # and plume type (how strong do plumes tend to be in our catalog). The below reflects this, although the
    # values are somewhat arbitrary at time of creation.
    sat_concentration_rescale_vals: dict[SatelliteID, dict[PlumeType, float | None]] = {
        SatelliteID.S2: {
            PlumeType.RECYCLED: 1,
            PlumeType.CARBONMAPPER: 10,
            PlumeType.GAUSSIAN: None,
        },
        SatelliteID.EMIT: {
            PlumeType.RECYCLED: 1,
            PlumeType.CARBONMAPPER: 2.5,
            PlumeType.GAUSSIAN: None,
        },
        SatelliteID.LANDSAT: {
            PlumeType.RECYCLED: 1,
            PlumeType.CARBONMAPPER: 10,
            PlumeType.GAUSSIAN: None,
        },
    }
    sat_min_emission_rates = {
        SatelliteID.S2: 100,
        SatelliteID.EMIT: 50,
        SatelliteID.LANDSAT: 100,
    }
    sat_hapi_data_paths = {
        SatelliteID.S2: S2_HAPI_DATA_PATH,
        SatelliteID.EMIT: EMIT_HAPI_DATA_PATH,
        SatelliteID.LANDSAT: LANDSAT_HAPI_DATA_PATH,
    }

    base_config = DataGenerationConfig(
        plume_catalog=plume_catalog,
        plume_type=plume_type,
        crop_size=CROP_SIZE,
        random_seed=42,
        transformation_params={"modulate": 1.0, "resize": 1.0},  # TODO: remove hard conding
        plume_proba_dict={0: 0.334, 1: 0.333, 2: 0.333},
        storage_options=storage_options,
        azure_cluster=False,
        git_revision_hash="",
        test=True,
        out_dir=str(out_dir),
        ml_client=None,  # Will be set later in generate_test_data_for_satellite
        s3_client=None,  # Will be set later in generate_test_data_for_satellite
        quality_thresholds=quality_thresholds[sat_key],
        psf_sigma=sat_psf_sigmas[sat_key],
        target_spatial_resolution=sat_spatial_resolutions[sat_key],
        concentration_rescale_value=sat_concentration_rescale_vals[sat_key][plume_type],
        min_emission_rate=sat_min_emission_rates[sat_key],
        simulated_emission_rate=1000.0,
        hapi_data_path=sat_hapi_data_paths[sat_key],
    )
    if sat_key == SatelliteID.S2:
        return {
            **base_config.model_dump(),  # Use model_dump() to convert to dict
            "sentinel_MGRS": "13REQ",
            "sentinel_date": datetime.datetime(2024, 11, 28),
            "bands": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B11", "B12", "B8A", "SCL"],
            "time_delta_days": 100,
            "nb_reference_ids": 3,  # 10
            "omnicloud_cloud_t": 35,
            "omnicloud_shadow_t": 30,
        }
    elif sat_key == SatelliteID.EMIT:
        return {
            **base_config.model_dump(),  # Use model_dump() to convert to dict
            "emit_id": "EMIT_L1B_RAD_001_20240127T195840_2402713_006",
            "emit_mask_labels": [
                EMITL2AMaskLabel.CLOUD,
                EMITL2AMaskLabel.CIRRUS_CLOUD,
                EMITL2AMaskLabel.DILATED_CLOUD,
            ],
        }
    elif sat_key == SatelliteID.LANDSAT:
        return {
            **base_config.model_dump(),  # Use model_dump() to convert to dict
            "landsat_tile_id": "LC09_L1TP_155029_20240111_20240111_02_T1",
            "bands": ["swir16", "swir22", "qa_pixel"],
            "time_delta_days": 50,
            "nb_reference_ids": 3,
        }
    else:
        raise ValueError(f"Unknown satellite type: {sat_key}")


def initialize_satellite_generator(
    sat_key: SatelliteID,
    plume_type: PlumeType,
    out_dir: Path | AzureBlobPath,
    in_dir: Path,
    storage_options: dict | None,
) -> tuple[BaseDataGeneration, type[BaseDataGeneration]]:
    """Initialize a satellite data generator with common setup logic."""
    if sat_key not in SATELLITE_CLASSES:
        raise ValueError(f"Invalid satellite key: {sat_key}")

    # Initialise S3 Client
    try:
        # Try without MSI first
        ml_client = initialize_ml_client(force_msi=False)
        s3_client = initialize_s3_client(ml_client)
    except HttpResponseError:
        # Retry with MSI if first attempt fails
        ml_client = initialize_ml_client(force_msi=True)
        s3_client = initialize_s3_client(ml_client)

    sat_class = SATELLITE_CLASSES[sat_key]
    sat_params = build_satellite_generation_params(sat_key, plume_type, out_dir, in_dir, storage_options)
    sat_params.update(
        {
            "ml_client": ml_client,
            "s3_client": s3_client,
        }
    )

    data_gen = sat_class(**sat_params)
    return data_gen, sat_class


def generate_test_data_for_satellite(
    out_dir: Path | AzureBlobPath,
    in_dir: Path,
    sat_key: SatelliteID,
    plume_type: PlumeType,
    storage_options: dict | None,
) -> None:
    """
    Generate training data for a specific satellite and save to out_dir.

    `out_dir` can be either a local directory or on Azure Blob Storage.
    """
    data_gen_pipeline, _ = initialize_satellite_generator(sat_key, plume_type, out_dir, in_dir, storage_options)
    logger.info(f"Generating testing data for {sat_key} with {plume_type}...")
    data_gen_pipeline()
