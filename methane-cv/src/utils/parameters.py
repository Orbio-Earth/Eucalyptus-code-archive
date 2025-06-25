"""Central location for key model parameters that are used across modules (i.e data, training, inference)."""

from enum import Enum

import numpy as np
import torch
from pydantic import AnyUrl

REQUIRED_NUM_SNAPSHOTS = 3  # Number of snapshots in the time series including the main scene
REQUIRED_NUM_PREVIOUS_SNAPSHOTS = 2

S2_B12_DEFAULT = 10.4
EMIT_PSF_DEFAULT = 12.61
LANDSAT_PSF_DEFAULT = 4.28

S2_HAPI_DATA_PATH = AnyUrl(
    "azureml://subscriptions/6e71ce37-b9fe-4c43-942b-cf0f7e78c8ab/resourcegroups/orbio-ml-rg/workspaces/orbio-ml-ml-workspace/datastores/workspaceblobstore/paths/data/radtran/hapi/2025_02_26"
)
EMIT_HAPI_DATA_PATH = AnyUrl(
    "azureml://subscriptions/6e71ce37-b9fe-4c43-942b-cf0f7e78c8ab/resourcegroups/orbio-ml-rg/workspaces/orbio-ml-ml-workspace/datastores/workspaceblobstore/paths/data/radtran/hapi/v1.1.1.0"
)
LANDSAT_HAPI_DATA_PATH = AnyUrl(
    "azureml://subscriptions/6e71ce37-b9fe-4c43-942b-cf0f7e78c8ab/resourcegroups/orbio-ml-rg/workspaces/orbio-ml-ml-workspace/datastores/workspaceblobstore/paths/data/radtran/hapi/2025_03_17_v4"
)


class SatelliteID(str, Enum):
    """Identifiers for satellites."""

    S2 = "S2"
    EMIT = "EMIT"
    ENMAP = "ENMAP"
    LANDSAT = "LANDSAT"

    def __str__(self) -> str:
        """Get string representation."""
        return self.value

    @classmethod
    def list(cls) -> list[str]:
        """Return a list of the valid values for PlumeType."""
        return list(map(lambda c: c.value, cls))  # type: ignore[attr-defined]


CROP_SIZE = 128
S2_BANDS = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B09",
    "B11",
    "B12",
    "B8A",
    "SCL",
]
LANDSAT_BANDS = [  # NOTE: this should match the data gen config
    "coastal",
    "blue",
    "green",
    "red",
    "nir08",
    "swir16",
    "swir22",
    "cirrus",
    "lwir11",
    "lwir12",
    "qa_pixel",
]

NUM_S2_BANDS = len(S2_BANDS)
NUM_LANDSAT_BANDS = len(LANDSAT_BANDS)

# Satellite ground sampling distance (GSD) in meters
# For satellites with different GSD per band (e.g. S2) we select the GSD that we resample all bands to.
SATELLITE_SPATIAL_RESOLUTIONS = {
    SatelliteID.S2: 20,
    SatelliteID.EMIT: 60,
    SatelliteID.LANDSAT: 30,
}


class CropSnapshots(int, Enum):
    """Identifiers for crop snapshots."""

    CROP_EARLIER = 2
    CROP_BEFORE = 1
    CROP_MAIN = 0

    @classmethod
    def list(cls) -> list[int]:
        """Return a list of the valid values for CropSnapshots."""
        return list(map(lambda c: c.value, cls))  # type: ignore[attr-defined]

    def as_str(self) -> str:
        """Return the string representation of the CropSnapshots."""
        return self.name


TARGET_COLUMN = "target"

NUM_EMIT_BANDS = 285  # get this from the parquet file
EMIT_BANDS = np.arange(NUM_EMIT_BANDS).tolist()

# TODO: number of bands and crop_size should be read from the parquet file instead of hard-coding
# NOTE: Not used for S2/LS training anymore, there we construct it based on the dataset
SATELLITE_COLUMN_CONFIGS = {
    SatelliteID.S2: {
        "crop_earlier": {"shape": (NUM_S2_BANDS, CROP_SIZE, CROP_SIZE), "dtype": torch.int16},
        "crop_main": {"shape": (NUM_S2_BANDS, CROP_SIZE, CROP_SIZE), "dtype": torch.int16},
        "crop_before": {"shape": (NUM_S2_BANDS, CROP_SIZE, CROP_SIZE), "dtype": torch.int16},
        "orig_swir16": {"shape": (1, CROP_SIZE, CROP_SIZE), "dtype": torch.int16},
        "orig_swir22": {"shape": (1, CROP_SIZE, CROP_SIZE), "dtype": torch.int16},
        TARGET_COLUMN: {"shape": (1, CROP_SIZE, CROP_SIZE), "dtype": torch.float32},
    },
    SatelliteID.EMIT: {
        "crop_main": {"shape": (NUM_EMIT_BANDS, CROP_SIZE, CROP_SIZE), "dtype": torch.float32},
        TARGET_COLUMN: {"shape": (1, CROP_SIZE, CROP_SIZE), "dtype": torch.float32},
    },
    SatelliteID.LANDSAT: {
        "crop_earlier": {"shape": (NUM_LANDSAT_BANDS, CROP_SIZE, CROP_SIZE), "dtype": torch.int16},
        "crop_main": {"shape": (NUM_LANDSAT_BANDS, CROP_SIZE, CROP_SIZE), "dtype": torch.int16},
        "crop_before": {"shape": (NUM_LANDSAT_BANDS, CROP_SIZE, CROP_SIZE), "dtype": torch.int16},
        "orig_swir16": {"shape": (1, CROP_SIZE, CROP_SIZE), "dtype": torch.int16},
        "orig_swir22": {"shape": (1, CROP_SIZE, CROP_SIZE), "dtype": torch.int16},
        TARGET_COLUMN: {"shape": (1, CROP_SIZE, CROP_SIZE), "dtype": torch.float32},
    },
}

# Currently selected to be roughly the 33.3% and 66.67% cutoff of AVIRIS plume frac distributions
PLUME_STRENGTH_STRONG_MIN_FRAC_CUTOFF = -0.04
PLUME_STRENGTH_MEDIUM_MIN_FRAC_CUTOFF = -0.025
PLUME_STRENGTH_WEAK_MIN_FRAC_CUTOFF = -0.001
