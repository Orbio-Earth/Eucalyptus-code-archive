"""Satellite agnostic data generation script for synthetic methane plume datasets."""

import argparse
import datetime
import json

from src.data.common.sim_plumes import PlumeType
from src.data.generation.base import DataGenerationConfig
from src.data.generation.emit_generation import EMITDataGeneration, EMITL2AMaskLabel
from src.data.generation.landsat import LandsatDataGeneration
from src.data.generation.sentinel2 import S2DataGeneration
from src.data.sentinel2 import (
    BAND_RESOLUTIONS,
)
from src.utils.parameters import SATELLITE_SPATIAL_RESOLUTIONS, SatelliteID
from src.utils.utils import setup_logging

logger = setup_logging()

# Registry of available satellite data generation classes
SATELLITE_CLASSES = {
    SatelliteID.S2: S2DataGeneration,
    SatelliteID.EMIT: EMITDataGeneration,
    SatelliteID.LANDSAT: LandsatDataGeneration,
}


def parse_quality(quality_threshold_str: str) -> float:
    """Parse quality threshold (0-1) from a string."""
    quality_threshold = float(quality_threshold_str)
    if not 0 <= quality_threshold <= 1:
        raise ValueError("quality_threshold should be between 0 (inclusive) and 1 (inclusive)")
    return quality_threshold


def parse_quality_thresholds(quality_dict: dict) -> dict:
    """Parse quality thresholds from a dictionary."""
    # Validate each value using parse_quality
    for key, (min_val, max_val) in quality_dict.items():
        quality_dict[key] = (
            parse_quality(min_val),
            parse_quality(max_val),
        )
    return quality_dict


def parse_date(s: str) -> datetime.datetime:
    """Parse date from a string."""
    return datetime.datetime.strptime(s, "%Y-%m-%d")


def parse_s2_bands(bands_str: str) -> list[str]:
    """Parse bands from a string."""
    bands = bands_str.split(",")
    for band in bands:
        if band not in BAND_RESOLUTIONS:
            raise ValueError(f"Invalid band {band}")
    if "SCL" not in bands:
        raise ValueError("SCL should be among the bands")
    if "B11" not in bands:
        raise ValueError("B11 should be among the bands")
    if "B12" not in bands:
        raise ValueError("B12 should be among the bands")
    return bands


def parse_landsat_bands(bands_str: str) -> list[str]:
    """Parse bands from a string."""
    bands = bands_str.split(",")
    if "qa_pixel" not in bands:
        raise ValueError("qa_pixel should be among the bands")
    if "swir16" not in bands:
        raise ValueError("swir16 should be among the bands")
    if "swir22" not in bands:
        raise ValueError("swir22 should be among the bands")
    if "pan" in bands:
        raise ValueError("pan should NOT be in bands, it has a different resolution (15m) than the other bands (30m)")
    return bands


def float_or_none(s: str) -> float | None:
    """Parse float from a string, or None if the string is empty."""
    if s in ("", None, "None", "none", "null"):
        return None
    return float(s)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic methane plume datasets for different satellite sources"
    )

    # General arguments
    parser.add_argument(
        "--satellite",
        choices=SatelliteID.list(),
        type=SatelliteID,
        required=True,
        help="Satellite data source to use",
    )
    parser.add_argument(
        "--plume_catalog",
        required=True,
        help="Path to JSON catalog containing plume file paths",
    )
    parser.add_argument(
        "--plume_type",
        choices=PlumeType.list(),
        required=True,
        type=PlumeType,
        help="Type of plume dataset to use",
    )
    parser.add_argument(
        "--plume_proba_dict",
        type=json.loads,
        help=(
            "JSON string containing probabilities to insert X number of plumes into chips. "
            """Example: '{"0": 0.1667, "1": 0.1667, "2": 0.1667, "3": 0.1667, "4": 0.1666, "5": 0.1666}'"""
        ),
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory in Azure Blob Storage",
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        required=True,
        help="Size of crops taken from satellite images",
    )
    parser.add_argument(
        "--quality_thresholds",
        type=json.loads,
        required=True,
        help=(
            "JSON string containing quality thresholds (min, max) for different regions. "
            """Example: '{"main_tile_cloud": [0.0, 0.98], "main_tile_nodata": [0.0, 0.95], "reference_tile_cloud_shadow": [0.0, 0.8]}'"""  # noqa: E501 (line-too-long)
        ),
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        required=True,
        help="Seed for random number generation",
    )
    parser.add_argument(
        "--transformation_params",
        type=json.loads,
        required=True,
        help="JSON-formatted transformation parameters for plume modification",
    )
    parser.add_argument(
        "--git_revision_hash",
        type=str,
        required=True,
        help="Git revision hash for tracking code version",
    )
    parser.add_argument(
        "--azure_cluster",
        action="store_true",
        help="Whether running on Azure cluster",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with limited data",
    )

    parser.add_argument(
        "--psf_sigma",
        type=float,
        help="Sigma of gaussian filter used to rescale AVIRIS plume resolution to that of the target sensor",
        required=True,
    )
    parser.add_argument(
        "--concentration_rescale_value",
        type=float_or_none,
        help=(
            "The factor by which to scale plume concentration values when inserting into imagery. "
            "This allows us to use e.g. weaker plumes in datasets for less sensitive instruments."
        ),
        required=True,
    )
    parser.add_argument(
        "--simulated_emission_rate",
        type=float,
        help=("The emission rate of the simulated plume. "),
        required=True,
    )
    parser.add_argument(
        "--min_emission_rate",
        type=float,
        help=("The minimum emission rate to rescale the plume to. "),
        required=True,
    )
    parser.add_argument(
        "--hapi_data_path",
        type=str,
        help="Path to HAPI spectral data",
        required=True,
    )

    # Common arguments for MultiTemporal Satellites (S2, LANDSAT)
    multitemporal_satellites = parser.add_argument_group("Time series arguments (S2 and Landsat)")
    multitemporal_satellites.add_argument(
        "--time_delta_days",
        type=int,
        help="Days to search before/after the target date",
    )
    multitemporal_satellites.add_argument(
        "--nb_reference_ids",
        type=int,
        help="Number of reference Images to consider for chipping",
    )

    # Sentinel-2 specific arguments
    s2_group = parser.add_argument_group("Sentinel-2 arguments")
    s2_group.add_argument(
        "--sentinel_MGRS",
        help="Sentinel-2 MGRS tile identifier",
    )
    s2_group.add_argument(
        "--sentinel_date",
        type=parse_date,
        help="Date of Sentinel-2 tile (YYYY-MM-DD)",
    )
    s2_group.add_argument(
        "--s2_bands",
        type=parse_s2_bands,
        help="Comma-separated list of bands to process",
    )
    s2_group.add_argument(
        "--omnicloud_cloud_t",
        type=float,
        help="Threshold for OmniCloud clouds",
    )
    s2_group.add_argument(
        "--omnicloud_shadow_t",
        type=float,
        help="Threshold for OmniCloud cloud shadows",
    )

    # EMIT specific arguments
    emit_group = parser.add_argument_group("EMIT arguments")
    emit_group.add_argument(
        "--emit_id",
        help="EMIT tile identifier",
    )

    # Landsat specific arguments
    landsat_group = parser.add_argument_group("Landsat arguments")
    landsat_group.add_argument(
        "--landsat_tile_id",
        help="Landsat tile identifier (e.g., LC08_L1TP_193038_20191014_20200825_02_T1)",
    )
    landsat_group.add_argument(
        "--landsat_bands",
        type=parse_landsat_bands,
        help="Comma-separated list of Landsat bands to process",
    )

    args = parser.parse_args()

    # Validate satellite-specific required arguments
    if args.satellite == SatelliteID.S2:
        for arg in [
            "sentinel_MGRS",
            "sentinel_date",
            "s2_bands",
            "time_delta_days",
            "nb_reference_ids",
            "omnicloud_cloud_t",
            "omnicloud_shadow_t",
        ]:
            if getattr(args, arg) is None:
                parser.error(f"--{arg} is required for Sentinel-2 processing")
    elif args.satellite == SatelliteID.EMIT:
        for arg in ["emit_id"]:
            if getattr(args, arg) is None:
                parser.error(f"--{arg} is required for EMIT processing")
    elif args.satellite == SatelliteID.LANDSAT:
        for arg in [
            "landsat_tile_id",
            "landsat_bands",
            "time_delta_days",
            "nb_reference_ids",
        ]:
            if getattr(args, arg) is None:
                parser.error(f"--{arg} is required for Landsat processing")
    else:
        raise ValueError(f"Unhandled satellite {args.satellite}")

    return args


def main() -> None:
    """Run synthetic dataset generation for specified satellite."""
    # Parse arguments
    args = parse_args()

    # Parse quality thresholds
    # TODO: should we have any error handling here? I.e check for expected quality keys in each satellite class?
    quality_thresholds = parse_quality_thresholds(args.quality_thresholds)

    # Validate transformation params
    assert isinstance(args.transformation_params, dict) and all(
        isinstance(k, str) and isinstance(v, float) for k, v in args.transformation_params.items()
    ), "transformation_params must be a dictionary with each string key having one float value"

    # Create base config
    base_config = DataGenerationConfig(
        plume_catalog=args.plume_catalog,
        plume_type=args.plume_type,
        out_dir=args.out_dir,
        crop_size=args.crop_size,
        quality_thresholds=quality_thresholds,
        random_seed=args.random_seed,
        transformation_params=args.transformation_params,
        azure_cluster=args.azure_cluster,
        git_revision_hash=args.git_revision_hash,
        test=args.test,
        ml_client=None,
        s3_client=None,
        storage_options=None,
        psf_sigma=args.psf_sigma,
        target_spatial_resolution=SATELLITE_SPATIAL_RESOLUTIONS[
            args.satellite
        ],  # TODO: should this be taken from the satellite hydra configs?
        concentration_rescale_value=args.concentration_rescale_value,
        simulated_emission_rate=args.simulated_emission_rate,
        min_emission_rate=args.min_emission_rate,
        plume_proba_dict=args.plume_proba_dict,
        hapi_data_path=args.hapi_data_path,
    )
    # Get satellite-specific class and parameters
    SatelliteClass = SATELLITE_CLASSES[args.satellite]
    if (
        args.satellite == SatelliteID.S2
    ):  # NOTE: this can be error prone if we don't check how the key is specified in SATELLITE_CLASSES
        satellite_params = {
            "sentinel_MGRS": args.sentinel_MGRS,
            "sentinel_date": args.sentinel_date,
            "bands": args.s2_bands,
            "time_delta_days": args.time_delta_days,
            "nb_reference_ids": args.nb_reference_ids,
            "omnicloud_cloud_t": args.omnicloud_cloud_t,
            "omnicloud_shadow_t": args.omnicloud_shadow_t,
        }
    elif args.satellite == SatelliteID.EMIT:
        satellite_params = {
            "emit_id": args.emit_id,
            "emit_mask_labels": [
                EMITL2AMaskLabel.CLOUD,
                EMITL2AMaskLabel.CIRRUS_CLOUD,
                EMITL2AMaskLabel.DILATED_CLOUD,
                EMITL2AMaskLabel.WATER,
                EMITL2AMaskLabel.SPACECRAFT,
            ],
        }
    elif args.satellite == SatelliteID.LANDSAT:
        satellite_params = {
            "landsat_tile_id": args.landsat_tile_id,
            "bands": args.landsat_bands,
            "time_delta_days": args.time_delta_days,
            "nb_reference_ids": args.nb_reference_ids,
        }
    else:
        raise ValueError(f"Unhandled satellite {args.satellite}")

    # Create and run pipeline
    pipeline = SatelliteClass(**satellite_params, **base_config.model_dump())

    pipeline()

    if (
        args.satellite in [SatelliteID.S2, SatelliteID.LANDSAT]
        and pipeline.reference_success_perc < 90  # noqa: PLR2004 (magic number)
        and pipeline.failed_5perc_count > 5  # noqa: PLR2004 (magic number)
    ):
        # If we found not enough good reference chips, restart the pipeline with more reference IDs to choose from
        satellite_params["nb_reference_ids"] = 20 if args.satellite == SatelliteID.LANDSAT else 25
        logger.info(
            f"Success is only {pipeline.reference_success_perc:.1f}% --> REDO with "
            f"nb_reference_ids={satellite_params['nb_reference_ids']}"
        )
        pipeline = SatelliteClass(**satellite_params, **base_config.model_dump())
        pipeline(log_params=False)
    logger.info("Data generation completed successfully")


if __name__ == "__main__":
    main()
