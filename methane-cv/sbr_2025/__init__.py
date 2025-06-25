"""Constants and parameters."""

from src.utils.parameters import SatelliteID
from src.utils.utils import get_satellite_concatenator_defaults

WIND_DATA_BASE_URL_TPL = "https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das/Y{:04}/M{:02}/D{:02}"
WIND_ENTRY_NAME_TPL = "GEOS.fp.asm.{}.{:04}{:02}{:02}_{:02}00.V01.nc4"
WIND_PRODUCT = "inst3_2d_asm_Nx"

BACKGROUND_METHANE1 = 0.01

MODEL_IDENTIFIERS = [
    "models:/torchgeo_pwr_unet/1226",
    "models:/torchgeo_pwr_unet/1340",
    "models:/torchgeo_pwr_unet/1395",
]
DEFAULT_PARAMS = get_satellite_concatenator_defaults(SatelliteID.S2)
BANDS = DEFAULT_PARAMS["all_available_bands"]

DEFAULT_PARAMS_LANDSAT = get_satellite_concatenator_defaults(SatelliteID.LANDSAT)
BANDS_LANDSAT = DEFAULT_PARAMS_LANDSAT["all_available_bands"]
