"""EMIT data generation pipeline."""

import itertools
from collections import namedtuple
from collections.abc import Iterator
from typing import Any

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import radtran
import xarray as xr
from pydantic import AnyUrl
from pyproj import CRS, Transformer
from rasterio.transform import Affine
from shapely.geometry import Polygon
from shapely.ops import transform, unary_union

from src.data.common.data_item import BasePlumesDataItem, MonoTemporalPlumesDataItem
from src.data.common.sim_plumes import PlumeType, transform_and_position_plumes
from src.data.emit_data import EmitGranuleAccess, EMITL2AMaskLabel
from src.data.generation.base import BaseDataGeneration, DataGenerationConfig
from src.utils import PROJECT_ROOT
from src.utils.radtran_utils import (
    compute_normalized_brightness,
    precompute_log_norm_brightness,
)
from src.utils.utils import earthaccess_login

PlumeSim = namedtuple("PlumeSim", ["radiance", "enhancement", "gamma"])


class EMITDataGeneration(BaseDataGeneration):
    """EMIT data generation pipeline."""

    def __init__(
        self,
        emit_id: str,
        emit_mask_labels: list[EMITL2AMaskLabel],
        **kwargs: Any,
    ) -> None:
        # Set instance attributes first
        self.emit_id = emit_id
        self.emit_mask_labels = emit_mask_labels

        # Initialize target producing area polygons to assign region labels to training/validation set chips
        self.hassi_poly = (
            gpd.read_file(f"{PROJECT_ROOT}/src/data/ancillary/hassi.geojson").geometry.iloc[0].buffer(1.5).simplify(0.1)
        )
        self.permian_poly = (
            gpd.read_file(f"{PROJECT_ROOT}/src/data/ancillary/permian.geojson").geometry.iloc[0].simplify(0.01)
        )
        self.colorado_poly = (
            gpd.read_file(f"{PROJECT_ROOT}/src/data/ancillary/colorado.geojson").geometry.iloc[0].simplify(0.01)
        )

        # self.producing_union = Union of all producing areas, TODO: Insert as we can't share the one we were using

        # Create config object for base class
        config = DataGenerationConfig(**kwargs)
        super().__init__(config=config)

        earthaccess_login(self.ml_client)

    @property
    def hash_id(self) -> str:
        """Get the unique scene identifier for hash computation."""
        return self.emit_id

    @property
    def scene_id(self) -> str:
        """Get the unique scene identifier for file naming."""
        return self.emit_id

    def get_producing_union_polygon_transformed(self, chip_crs: CRS) -> Polygon:
        """Get the producing union polygon transformed to the chip CRS."""
        transformer = Transformer.from_crs("EPSG:4326", chip_crs, always_xy=True)
        return transform(transformer.transform, self.producing_union)

    def get_target_producing_regions_transformed(self, chip_crs: CRS) -> dict[str, Polygon]:
        """Get the target producing regions."""
        transformer = Transformer.from_crs("EPSG:4326", chip_crs, always_xy=True)
        return {
            "Hassi": transform(transformer.transform, self.hassi_poly),
            "Permian": transform(transformer.transform, self.permian_poly),
            "Colorado": transform(transformer.transform, self.colorado_poly),
        }

    def download_data(self) -> dict[str, xr.DataArray | xr.Dataset]:
        """
        Download EMIT data for a given tile ID.

        Returns
        -------
            Dictionary containing:
            - rad_ds: Radiance xarray DataArray
            - obs_ds: Observation xarray DataArray
            - mask_ds: Mask xarray DataArray
            - sensor_band_parameters: Sensor band parameters xarray Dataset
        """
        emit_granule = EmitGranuleAccess(self.emit_id)

        # Get radiance data with all bands
        rad_ds = emit_granule.get_radiance()

        # Get observation data
        obs_ds = emit_granule.get_obs()

        # Get mask data
        mask_ds = emit_granule.get_mask()

        # Get sensor band parameters
        sensor_band_parameters = emit_granule.get_sensor_band_parameters()

        return {
            "rad_ds": rad_ds,
            "obs_ds": obs_ds,
            "mask_ds": mask_ds,
            "sensor_band_parameters": sensor_band_parameters,
        }

    def prepare_tile_level_data(self, emit_data: dict[str, xr.DataArray | xr.Dataset]) -> dict[str, Any]:
        """Prepare data that is shared across all crops from a single EMIT scene.

        This method extracts the sensor band parameters from the EMIT radiance dataset,
        which are constant across the entire scene and needed for plume simulation.

        Parameters
        ----------
        emit_data : dict[str, xr.Dataset]
            Dictionary containing EMIT datasets:
            - rad_ds: Radiance dataset
            - obs_ds: Observation dataset
            - mask_ds: Mask dataset
            - sensor_band_parameters: Sensor band parameters dataset

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - sensor_band_parameters: xarray Dataset with EMIT sensor band parameters
        """
        return {"sensor_band_parameters": emit_data["sensor_band_parameters"]}

    def generate_crops(self, emit_data: dict[str, xr.DataArray | xr.Dataset]) -> Iterator[dict[str, Any]]:
        """Generate crops from EMIT data."""
        rad_ds = emit_data["rad_ds"]
        obs_ds = emit_data["obs_ds"]
        mask_ds = emit_data["mask_ds"]

        # check all the crop heights and widths are the same
        shapes = {
            "radiance": rad_ds.shape[:2],
            "observation": obs_ds.shape[:2],
            "mask": mask_ds.shape[:2],
        }
        assert len(set(shapes.values())) == 1, f"Dataset shapes don't match: {shapes}"

        label_values = [label.value for label in self.emit_mask_labels]

        # Get dimensions
        height, width = rad_ds.shape[0:2]

        for i, j in itertools.product(range(height // self.crop_size), range(width // self.crop_size)):
            # Calculate crop coordinates
            h_start = i * self.crop_size
            h_end = h_start + self.crop_size
            w_start = j * self.crop_size
            w_end = w_start + self.crop_size

            # Extract crops
            rad_crop = rad_ds.isel(downtrack=slice(h_start, h_end), crosstrack=slice(w_start, w_end))
            obs_crop = obs_ds.isel(downtrack=slice(h_start, h_end), crosstrack=slice(w_start, w_end))
            mask_crop = mask_ds.isel(downtrack=slice(h_start, h_end), crosstrack=slice(w_start, w_end))

            # Filter for valid crops
            # Create single boolean mask from emit_mask_labels
            mask_crop_bool = mask_crop.isel(bands=label_values).astype(bool)
            cloud_percentage = mask_crop_bool.any(dim="bands").mean().astype(np.float32).values
            lower_limit, upper_limit = self.quality_thresholds["main_crop"]
            if not (lower_limit <= cloud_percentage < upper_limit):
                continue

            # TODO: think about whether it is better to return a data class/xarray instead
            yield {
                "rad_crop": rad_crop,
                "obs_crop": obs_crop,
                "mask_crop": mask_crop_bool.max(dim="bands"),
                "crop_x": w_start,
                "crop_y": h_start,
                "cloud_ratio": cloud_percentage,
            }

    def generate_synthetic_data_items(
        self,
        plume_files: npt.NDArray,
        crops: Iterator[dict],
        position_by_source: bool,
        sensor_band_parameters: xr.Dataset,
    ) -> Iterator[MonoTemporalPlumesDataItem]:
        """Generate synthetic EMIT data items with plumes.

        For EMIT data, the target variable represents the product of:
        - enhancement: The plume concentration in mol/m²
        - gamma: Path length factor computed from observation angles as:
            1/cos(θ_sensor) + 1/cos(θ_sun)
          where θ_sensor is the sensor zenith angle and θ_sun is the solar zenith angle

        This product (gamma * enhancement) represents the effective methane column density
        accounting for the actual path length that light travels through the plume based
        on the viewing geometry.

        Args:
            plume_files: Array of plume file paths to sample from
            crops: Iterator of crop data dictionaries containing radiance, observation
                  and mask data for each spatial crop

        Returns
        -------
            Iterator of MonoTemporalPlumesDataItem containing the synthetic data
        """
        # Load constant data
        CH4_absorption_da = self._prepare_absorption_data(self.hapi_data_path)

        # Precompute brightness lookup
        log_normalized_brightness = precompute_log_norm_brightness(
            sensor_band_parameters,
            CH4_absorption_da,
            # For out gamma concentrations, we set one very low concentration (1e-5) so we can interpolate
            # for very small enhancements, but then have a bigger jump to the next concentration given low
            # sensitivity at low values. We more densely interpolate at higher concentrations.
            gamma_concentration=2 * np.array([0.00001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]),
        )

        # Process each crop
        for i, crop_data in enumerate(crops):
            # For testing, only process a small subset of crops
            if self.test and i > self.NUM_SAMPLES_FOR_TESTING:
                break
            rad_crop = crop_data["rad_crop"]
            obs_crop = crop_data["obs_crop"]
            mask_crop = crop_data["mask_crop"]
            crop_x = crop_data["crop_x"]
            crop_y = crop_data["crop_y"]
            cloud_ratio = crop_data["cloud_ratio"]

            # Get spatial metadata for region overlap check
            crs = CRS.from_wkt(rad_crop.attrs["spatial_ref"])
            gt = rad_crop.attrs["geotransform"]
            transform = Affine(gt[1], gt[2], gt[0], gt[4], gt[5], gt[3])

            _, overlap_region_label = self._overlaps_with_producing_area(
                crop_x=crop_x,
                crop_y=crop_y,
                chip_transform=transform,
                producing_area_polygon=self.get_producing_union_polygon_transformed(crs),
                target_producing_regions=self.get_target_producing_regions_transformed(crs),
                check_overlap_ratio=False,
            )

            # Select plumes for this crop
            chosen_plume_files, plume_arrays, plume_emissions = self._select_plumes(plume_files)

            sim = self._add_plumes(
                rad_crop,
                obs_crop,
                mask_crop,
                plume_arrays,
                self.rng,
                self.transformation_params,
                log_normalized_brightness,
                position_by_source=position_by_source,
            )

            data_item = MonoTemporalPlumesDataItem.create_data_item(
                # EMIT granules can contain strips of nans that will cause model training
                # to fail. Here we take a simple approach of 0-filling but in future we
                # should consider: whether we want a dedicated nan value that is not also a
                # plausible radiance value (-1); and how we sample these nan strips to ensure
                # the model can learn to ignore them.
                modified_crop=sim.radiance.fillna(0),
                target=sim.enhancement * sim.gamma,  # the target variable is the plume concentration times gamma
                mask=mask_crop,
                granule_id=self.emit_id,
                plume_files=chosen_plume_files,
                plume_emissions=plume_emissions,
                bands=list(rad_crop.bands.values),
                size=self.crop_size,
                crop_x=crop_x,
                crop_y=crop_y,
                main_cloud_ratio=cloud_ratio,
                transformation_params=self.transformation_params,
                region_overlap=overlap_region_label,
            )
            yield data_item

    def save_parquet(self, data_items: Iterator[BasePlumesDataItem]) -> None:
        """Save data items to parquet file."""
        parquet_path = self._get_parquet_path()

        # Convert data items to DataFrame
        data_df = pd.DataFrame([item.to_dict() for item in data_items])

        if data_df.empty:
            return

        if self.plume_type == PlumeType.CARBONMAPPER:
            # Merge emission rates for AVIRIS plumes
            aviris_plumes = pd.concat(
                (
                    pd.read_csv(f"{PROJECT_ROOT}/src/data/ancillary/aviris_plumes_training.csv"),
                    pd.read_csv(f"{PROJECT_ROOT}/src/data/ancillary/aviris_plumes_validation.csv"),
                ),
                ignore_index=True,
            )
            aviris_plumes["con_tif"] = aviris_plumes["con_tif"].apply(lambda x: f"azureml://{x}")
            # 100 plumes do not have an emission rate
            aviris_plumes["emission_auto"] = aviris_plumes["emission_auto"].fillna(
                aviris_plumes["emission_auto"].median()
            )
            aviris_plumes["emission_auto"] = aviris_plumes["emission_auto"].round(1)
            plume_to_emission = (
                aviris_plumes[["con_tif", "emission_auto"]].set_index("con_tif")["emission_auto"].to_dict()
            )

            data_df["plume_emissions"] = data_df["plume_files"].apply(
                lambda x: [plume_to_emission.get(x.decode("utf-32le").strip())]
                if isinstance(x, bytes) and len(x) > 0
                else np.nan
            )

        if "validation_" in str(self.out_dir):
            # Set all chips to one of Hassi, Colorado or Permian as we want to validate only in these groups
            val_region = [k for k in data_df["region_overlap"].unique().tolist() if k != "Other"]
            if len(val_region) > 0:
                data_df["region_overlap"] = val_region[0]

        # Save with standard compression settings
        data_df.to_parquet(
            str(parquet_path),
            compression="zstd",
            compression_level=9,
            row_group_size=1,
            write_statistics=False,  # row-group statistics -> meaningless when row groups are 1 row.
            write_page_index=False,  # groups the page statistics into one place for more efficient IO
            store_schema=False,  # False - do not write Arrow schema to file.  Will effect recreation of some types.
            use_dictionary=False,  # don't use dictionary encoding.  Doesn't reduce our file size, uses more memory.
            storage_options=self.storage_options,
        )

    @staticmethod
    def _prepare_absorption_data(hapi_data_prefix: AnyUrl, T: float = 300, p: float = 1.013) -> xr.DataArray:
        """Prepare CH4 absorption data."""
        CH4_absorption = radtran.filter_functions.get_absorption_cross_section_vector(
            temperature=T,
            pressure=p,
            instrument="EMIT",
            band="VSWIR",
            species="CH4",
            hapi_data_prefix=hapi_data_prefix,
        )

        return xr.DataArray(CH4_absorption.response, coords={"wavelength": CH4_absorption.wavelength})

    @staticmethod
    def _add_plumes(
        rad_crop: xr.DataArray,
        obs_crop: xr.DataArray,
        mask_crop: npt.NDArray,
        plume_arrays: list[np.ndarray],
        rng: np.random.Generator,
        transformation_params: dict[str, float],
        log_nB_table: xr.DataArray,
        position_by_source: bool,
    ) -> PlumeSim:
        """Add synthetic plumes to EMIT crop.

        Returns
        -------
        PlumeSim
            A named tuple containing:
            - radiance: The modified radiance data with simulated plumes applied
            - enhancement: The plume concentration in mol/m²
            - gamma: Path length factor computed from observation angles. When multiplied
              with enhancement, gives the target variable used for training.
        """
        # Calculate gamma from observation angles
        gamma = radtran.get_gamma(obs_crop.sel(bands="solar_zenith"), obs_crop.sel(bands="sensor_zenith"))

        # Get reference band without bands dimension
        crop_spatial_grid = rad_crop.isel(bands=0)

        # Generate enhancement and modify radiance
        enhancement, mask, plumes_inserted_idxs = transform_and_position_plumes(
            plume_arrs=plume_arrays,
            tile_band=crop_spatial_grid.values,
            exclusion_mask_plumes=mask_crop,
            rng=rng,
            transformation_params=transformation_params,
            position_by_source=position_by_source,
        )

        enhancement_da = xr.DataArray(enhancement, coords=crop_spatial_grid.coords)

        # If enhancement is all zeros, return original radiance. This mimics the behavior of the S2 pipeline where we
        # return the original bands if the plume was not able to be placed in the tile. See comment in sim_plumes.py
        if np.all(enhancement == 0):
            modified_rad_crop = rad_crop.copy()
        else:
            normalized_brightness = compute_normalized_brightness(
                gamma=gamma,
                concentration=enhancement_da,
                log_nB_table=log_nB_table,
            )

            modified_radiance = rad_crop * normalized_brightness
            modified_rad_crop = rad_crop.copy()
            modified_rad_crop = modified_radiance

        return PlumeSim(
            radiance=modified_rad_crop.astype(np.float32),
            enhancement=enhancement_da.astype(np.float32).values,
            gamma=gamma.astype(np.float32).values,
        )
