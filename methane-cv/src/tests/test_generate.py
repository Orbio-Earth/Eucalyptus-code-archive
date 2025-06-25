"""Test the data generation `src/data/generate.py` script."""

from pathlib import Path

import pytest

from src.data.common.sim_plumes import PlumeType
from src.tests.generate_test_data import initialize_satellite_generator
from src.utils.parameters import SatelliteID


def test_parameter_serialization(
    sat_key: SatelliteID, plume_type: PlumeType, local_sat_dir: Path, local_in_dir: Path
) -> None:
    """
    Test that parameter serialization works for all satellite classes.

    This ensures that if any new attribute is added to a satellite class, it is added to NON_SERIALIZABLE_ATTRS if it
    can't be serialized. This way the `_log_params_to_mlflow` will not fail when trying to log the parameters.
    """
    if sat_key == SatelliteID.EMIT and plume_type == PlumeType.CARBONMAPPER:
        pytest.skip("Unimplemented configuration")

    data_gen, sat_class = initialize_satellite_generator(
        sat_key, plume_type, local_sat_dir, local_in_dir, storage_options=None
    )

    try:
        data_gen._get_serializable_params()
    except Exception as e:
        pytest.fail(
            f"Failed to retrieve serializable params for satellite {sat_key} "
            f"({sat_class.__name__}): {e!s}. This likely means a non-serializable parameter "
            "was introduced without being added to NON_SERIALIZABLE_ATTRS."
        )


# FIXME: We deleted test/unzipped_data/plumes/catalog_condensed.json which makes this test fail for Recycled plumes
# As this was the only test case we used, we are skipping this test for now.
# @pytest.mark.usefixtures("azure_test_data")
# def test_generate_to_azure(azure_sat_dir: Path, storage_options: dict, ml_client: MLClient) -> None:
#     """Verifies that the Parquet files have been successfully generated and saved to the test Azure Blob Storage."""
#     azure_out_dir = get_abfs_output_directory(ml_client, azure_sat_dir)
#     essential_columns = ["crop_main", "target", "plume_files", "bands", "size", "crop_x", "crop_y"]
#     ddf = dd.read_parquet(
#         azure_out_dir,
#         dtype_backend="pyarrow",
#         split_row_groups=1,
#         storage_options=storage_options,
#         columns=essential_columns,
#     )
#     assert len(ddf) >= 1


# FIXME: Add a data generation test that compares the test generated azure file with a file that
# already exists on Azure/in repo. See https://git.orbio.earth/orbio/orbio/-/issues/1321
# @pytest.mark.usefixtures("local_test_data")
# def test_generate_to_local(local_sat_dir: Path) -> None:
#     """Verifies that the Parquet files have been successfully saved and visualizations were not run."""
#     # Mock visualization-related functions
#     with patch.object(plt, "plot") as mock_plot, patch.object(plt, "show") as mock_show:
#         assert len(list(local_sat_dir.glob("*.parquet"))) >= 1

#         # Ensure that visualization functions were NOT called
#         mock_plot.assert_not_called()
#         mock_show.assert_not_called()

# @pytest.mark.usefixtures("local_test_data")
# @pytest.mark.usefixtures("azure_test_data")
# def test_reproducibility(  # PLR0915 (too-many-statements)
#     sat_key: SatelliteID,
#     plume_type: PlumeType,
#     local_sat_dir: Path,
#     azure_sat_dir: Path,
#     storage_options: dict,
#     ml_client: MLClient,
# ) -> None:
#     """Confirms parquet file generated on ABS matches the parquet file saved locally to ensure data consistency."""
#     # dtype_backend="pyarrow" is needed so the plume_files column gets read in as a list rather than a string
#     if sat_key == SatelliteID.EMIT and plume_type == PlumeType.CARBONMAPPER:
#         pytest.skip("Unimplemented configuration")

#     # Base columns common to all satellites
#     base_essential_columns = [
#         "crop_main",
#         "target",
#         "plume_files",
#         "bands",
#         "size",
#         "crop_x",
#         "crop_y",
#         "modulate",
#         "resize",
#     ]

#     # Additional columns specific to S2/Landsat
#     multitemporal_columns = [
#         "crop_earlier",
#         "crop_before",
#         "orig_swir16",
#         "orig_swir22",
#         "cloud_earlier",
#         "cloud_before",
#         "cloud_main",
#         "main_and_reference_ids",
#         "cloudshadow_main",
#         "cloudshadow_before",
#         "cloudshadow_earlier",
#         "exclusion_mask_plumes",
#         "how_many_plumes_we_wanted",
#         "how_many_plumes_we_inserted",
#         "plumes_inserted_idxs",
#         "plume_sizes",
#         "frac_abs_sum",
#         "min_frac",
#         "plume_category",
#         "exclusion_perc",
#         "region_overlap",
#     ]

#     # EMIT specific columns
#     emit_columns = ["mask_main", "granule_item_id", "orig_bands"]

#     # Select columns based on satellite type
#     essential_columns = base_essential_columns + (
#         multitemporal_columns
#         if sat_key in [SatelliteID.S2, SatelliteID.LANDSAT]
#         else emit_columns
#         if sat_key == SatelliteID.EMIT
#         else []
#     )

#     # Read the data from the local directory
#     if sat_key == SatelliteID.S2:
#         local_path = local_sat_dir / "13REQ_2024-11-28.parquet"
#     elif sat_key == SatelliteID.LANDSAT:
#         local_path = local_sat_dir / "LC09_L1TP_155029_20240111_20240111_02_T1.parquet"
#     else:
#         local_path = local_sat_dir

#     ddf1 = dd.read_parquet(
#         local_path,
#         storage_options={},
#         dtype_backend="pyarrow",
#         split_row_groups=1,
#         columns=essential_columns,
#     )
#     assert len(ddf1) > 0

#     # Read the data from the ABS directory
#     azure_out_dir = get_abfs_output_directory(ml_client, azure_sat_dir)
#     if sat_key == SatelliteID.S2:
#         abs_path = azure_out_dir / "13REQ_2024-11-28.parquet"
#     elif sat_key == SatelliteID.LANDSAT:
#         abs_path = azure_out_dir / "LC09_L1TP_155029_20240111_20240111_02_T1.parquet"
#     else:
#         abs_path = azure_out_dir

#     ddf2 = dd.read_parquet(
#         abs_path,
#         storage_options=storage_options,
#         dtype_backend="pyarrow",
#         split_row_groups=1,
#         columns=essential_columns,
#     )
#     assert len(ddf2) > 0

#     assert len(ddf1) == len(ddf2)
#     # Compare sample rows
#     sample_row1 = ddf1.get_partition(0).compute().iloc[0, :]
#     sample_row2 = ddf2.get_partition(0).compute().iloc[0, :]

#     # Compare common columns
#     assert sample_row1.crop_main == sample_row2.crop_main
#     assert sample_row1.target == sample_row2.target
#     # Compare just plume filenames since local/ABS prefixes will differ
#     assert Counter([p.split("/")[-1] for p in sample_row1.plume_files]) == Counter(
#         [p.split("/")[-1] for p in sample_row2.plume_files]
#     )
#     assert Counter(sample_row1.bands) == Counter(sample_row2.bands)
#     assert sample_row1.size == sample_row2.size
#     assert sample_row1.crop_x == sample_row2.crop_x
#     assert sample_row1.crop_y == sample_row2.crop_y
#     assert sample_row1.modulate == sample_row2.modulate
#     assert sample_row1.resize == sample_row2.resize

#     # Compare satellite specific columns
#     if sat_key in [SatelliteID.S2, SatelliteID.LANDSAT]:
#         assert sample_row1.crop_earlier == sample_row2.crop_earlier
#         assert sample_row1.crop_before == sample_row2.crop_before
#         assert sample_row1.orig_swir16 == sample_row2.orig_swir16
#         assert sample_row1.orig_swir22 == sample_row2.orig_swir22
#         assert sample_row1.cloud_earlier == sample_row2.cloud_earlier
#         assert sample_row1.cloud_before == sample_row2.cloud_before
#         assert sample_row1.cloud_main == sample_row2.cloud_main
#         assert Counter(sample_row1.main_and_reference_ids) == Counter(sample_row2.main_and_reference_ids)
#         assert sample_row1.cloudshadow_main == sample_row2.cloudshadow_main
#         assert sample_row1.cloudshadow_before == sample_row2.cloudshadow_before
#         assert sample_row1.cloudshadow_earlier == sample_row2.cloudshadow_earlier
#         assert sample_row1.exclusion_mask_plumes == sample_row2.exclusion_mask_plumes
#         assert sample_row1.how_many_plumes_we_wanted == sample_row2.how_many_plumes_we_wanted
#         assert sample_row1.how_many_plumes_we_inserted == sample_row2.how_many_plumes_we_inserted
#         assert Counter(sample_row1.plumes_inserted_idxs) == Counter(sample_row2.plumes_inserted_idxs)
#         assert Counter(sample_row1.plume_sizes) == Counter(sample_row2.plume_sizes)
#         assert sample_row1.frac_abs_sum == sample_row2.frac_abs_sum
#         assert sample_row1.min_frac == sample_row2.min_frac
#         assert sample_row1.plume_category == sample_row2.plume_category
#         assert sample_row1.exclusion_perc == sample_row2.exclusion_perc
#         assert sample_row1.region_overlap == sample_row2.region_overlap
#         # Theres so much more we can add ... but just matched the fields in MultiTemporalPlumesDataItem
#     elif sat_key == SatelliteID.EMIT:
#         assert sample_row1.mask_main == sample_row2.mask_main
#         assert sample_row1.granule_item_id == sample_row2.granule_item_id
#         assert sample_row1.orig_bands == sample_row2.orig_bands
#     else:
#         raise ValueError(f"Unknown satellite type: {sat_key}")
