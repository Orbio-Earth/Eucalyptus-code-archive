import numpy as np
import pytest
import xarray as xr

from src.plotting.plotting_functions import get_rgb_from_xarray, get_swir_ratio_from_xarray
from src.utils.parameters import SatelliteID


@pytest.mark.parametrize(
    "x,satellite_id,rgb_target",
    [
        pytest.param(
            xr.DataArray(
                np.stack([np.full((128, 128), i, dtype=np.float32) for i in range(1_000, 6_000, 1_000)]),
                dims=["bands", "y", "x"],
                coords={"bands": ["B01", "B02", "B03", "B04", "B05"], "y": np.arange(128), "x": np.arange(128)},
            ),
            SatelliteID.S2,
            xr.DataArray(
                np.stack(
                    [
                        np.full((128, 128), 0.4, dtype=np.float32),
                        np.full((128, 128), 0.3, dtype=np.float32),
                        np.full((128, 128), 0.2, dtype=np.float32),
                    ],
                    axis=-1,
                ),
                dims=["y", "x", "bands"],
                coords={"y": np.arange(128), "x": np.arange(128), "bands": ["B04", "B03", "B02"]},
            ),
            id="Correct Sentinel-2 RGB image",
        ),
        pytest.param(
            xr.DataArray(
                np.stack([np.full((128, 128), i, dtype=np.float32) for i in range(9)]),
                dims=["bands", "y", "x"],
                coords={
                    "bands": [
                        # 2 blue bands
                        10,
                        12,
                        # 1 in-between band
                        15,
                        # 2 green bands
                        18,
                        24,
                        # 1 in-between band
                        30,
                        # 3 red bands
                        35,
                        40,
                        45,
                    ],
                    "y": np.arange(128),
                    "x": np.arange(128),
                },
            ),
            SatelliteID.EMIT,
            xr.DataArray(
                np.stack(
                    [
                        np.full((128, 128), 7, dtype=np.float32),
                        np.full((128, 128), 3.5, dtype=np.float32),
                        np.full((128, 128), 0.5, dtype=np.float32),
                    ],
                    axis=-1,
                ),
                dims=["y", "x", "bands"],
                coords={"y": np.arange(128), "x": np.arange(128), "bands": ["red", "green", "blue"]},
            ),
            id="Correct EMIT RGB image",
        ),
        pytest.param(
            xr.DataArray(
                np.stack([np.full((128, 128), i, dtype=np.float32) for i in range(1_000, 8_000, 1_000)]),
                dims=["bands", "y", "x"],
                coords={
                    "bands": ["blue", "green", "red", "nir08", "swir16", "swir22", "qa_pixel"],
                    "y": np.arange(128),
                    "x": np.arange(128),
                },
            ),
            SatelliteID.LANDSAT,
            xr.DataArray(
                np.stack(
                    [
                        np.full((128, 128), 0.3, dtype=np.float32),
                        np.full((128, 128), 0.2, dtype=np.float32),
                        np.full((128, 128), 0.1, dtype=np.float32),
                    ],
                    axis=-1,
                ),
                dims=["y", "x", "bands"],
                coords={"y": np.arange(128), "x": np.arange(128), "bands": ["red", "green", "blue"]},
            ),
            id="Correct Landsat RGB image",
        ),
    ],
)
def test_get_rgb(
    x: xr.DataArray,
    satellite_id: SatelliteID,
    rgb_target: xr.DataArray,
) -> None:
    """Ensure we are getting expected RGB values."""
    rgb_result = get_rgb_from_xarray(x, satellite_id)
    assert rgb_result.equals(rgb_target)


@pytest.mark.parametrize(
    "x,satellite_id",
    [
        pytest.param(
            xr.DataArray(
                np.stack([np.full((128, 128), i, dtype=np.float32) for i in range(1_000, 6_000, 1_000)]),
                dims=["bands", "y", "x"],
                coords={"bands": ["B01", "B02", "B03", "B05", "B06"], "y": np.arange(128), "x": np.arange(128)},
            ),
            SatelliteID.S2,
            id="Incorrect bands for Sentinel-2 RGB image",
        ),
        pytest.param(
            xr.DataArray(
                np.stack([np.full((128, 128), i, dtype=np.float32) for i in range(6)]),
                dims=["bands", "y", "x"],
                coords={
                    "bands": [
                        # NO blue bands
                        # 2 green bands
                        18,
                        24,
                        # 1 in-between band
                        30,
                        # 3 red bands
                        35,
                        40,
                        45,
                    ],
                    "y": np.arange(128),
                    "x": np.arange(128),
                },
            ),
            SatelliteID.EMIT,
            id="Inorrect bands for EMIT RGB image",
        ),
        pytest.param(
            xr.DataArray(
                np.stack([np.full((128, 128), i, dtype=np.float32) for i in range(1_000, 5_000, 1_000)]),
                dims=["bands", "y", "x"],
                coords={"bands": ["nir08", "swir16", "swir22", "qa_pixel"], "y": np.arange(128), "x": np.arange(128)},
            ),
            SatelliteID.LANDSAT,
            id="Incorrect bands for Landsat RGB image",
        ),
    ],
)
def test_get_rgb_incorrect_bands(
    x: xr.DataArray,
    satellite_id: SatelliteID,
) -> None:
    """Check that our RGB function fails if required bands not present."""
    with pytest.raises(KeyError):
        _ = get_rgb_from_xarray(x, satellite_id)


@pytest.mark.parametrize(
    "x,satellite_id,swir_target",
    [
        pytest.param(
            xr.DataArray(
                np.stack([np.full((128, 128), i, dtype=np.float32) for i in [1, 5, 10]]),
                dims=["bands", "y", "x"],
                coords={"bands": ["B10", "B11", "B12"], "y": np.arange(128), "x": np.arange(128)},
            ),
            SatelliteID.S2,
            xr.DataArray(
                np.full((128, 128), 2, dtype=np.float32),
                dims=["y", "x"],
                coords={"y": np.arange(128), "x": np.arange(128)},
            ),
            id="Correct Sentinel-2 SWIR image",
        ),
        pytest.param(
            xr.DataArray(
                np.stack([np.full((128, 128), i, dtype=np.float32) for i in range(6)]),
                dims=["bands", "y", "x"],
                coords={
                    "bands": [
                        # 2 swir1 bands
                        220,
                        230,
                        # 1 in-between band
                        240,
                        # 2 swir2 bands
                        250,
                        260,
                        # 1 additional band
                        270,
                    ],
                    "y": np.arange(128),
                    "x": np.arange(128),
                },
            ),
            SatelliteID.EMIT,
            xr.DataArray(
                np.full((128, 128), 7, dtype=np.float32),
                dims=["y", "x"],
                coords={"y": np.arange(128), "x": np.arange(128)},
            ),
            id="Correct EMIT SWIR image",
        ),
        pytest.param(
            xr.DataArray(
                np.stack([np.full((128, 128), i, dtype=np.float32) for i in [1, 2, 3]]),
                dims=["bands", "y", "x"],
                coords={"bands": ["nir08", "swir16", "swir22"], "y": np.arange(128), "x": np.arange(128)},
            ),
            SatelliteID.LANDSAT,
            xr.DataArray(
                np.full((128, 128), 1.5, dtype=np.float32),
                dims=["y", "x"],
                coords={"y": np.arange(128), "x": np.arange(128)},
            ),
            id="Correct Landsat SWIR image",
        ),
    ],
)
def test_get_swir_ratio(
    x: xr.DataArray,
    satellite_id: SatelliteID,
    swir_target: xr.DataArray,
) -> None:
    """Ensure we are getting expected RGB values."""
    swir_result = get_swir_ratio_from_xarray(x, satellite_id)
    assert swir_result.equals(swir_target)


@pytest.mark.parametrize(
    "x,satellite_id",
    [
        pytest.param(
            xr.DataArray(
                np.stack([np.full((128, 128), i, dtype=np.float32) for i in range(3)]),
                dims=["bands", "y", "x"],
                coords={"bands": ["B08", "B09", "B10"], "y": np.arange(128), "x": np.arange(128)},
            ),
            SatelliteID.S2,
            id="Incorrect bands for Sentinel-2 RGB image",
        ),
        pytest.param(
            xr.DataArray(
                np.stack([np.full((128, 128), i, dtype=np.float32) for i in range(5)]),
                dims=["bands", "y", "x"],
                coords={
                    "bands": [
                        # 2 swir1 bands
                        220,
                        230,
                        # 1 in-between band
                        240,
                        # NO swir2 bands
                        # additional bands
                        270,
                        280,
                    ],
                    "y": np.arange(128),
                    "x": np.arange(128),
                },
            ),
            SatelliteID.EMIT,
            id="Incorrect bands for EMIT RGB image",
        ),
        pytest.param(
            xr.DataArray(
                np.stack([np.full((128, 128), i, dtype=np.float32) for i in range(3)]),
                dims=["bands", "y", "x"],
                coords={"bands": ["blue", "green", "red"], "y": np.arange(128), "x": np.arange(128)},
            ),
            SatelliteID.LANDSAT,
            id="Incorrect bands for Landsat SWIR image",
        ),
    ],
)
def test_get_swir_ratio_incorrect_bands(
    x: xr.DataArray,
    satellite_id: SatelliteID,
) -> None:
    """Check that our RGB function fails if required bands not present."""
    with pytest.raises(KeyError):
        _ = get_swir_ratio_from_xarray(x, satellite_id)
