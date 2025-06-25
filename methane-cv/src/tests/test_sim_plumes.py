"""Tests for logic in sim_plumes.py."""

import math
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from azure.ai.ml import MLClient
from azure.storage.blob import BlobServiceClient
from PIL import Image
from rasterio.warp import Resampling, reproject

from src.azure_wrap.ml_client_utils import initialize_blob_service_client
from src.data.common.sim_plumes import (
    DENSITY_AIR_GROUND_LEVEL,
    M_AIR,
    UNIT_MULTIPLIER,
    AVIRISPlumeTransformParams,
    GaussianPlumeTransformParams,
    PlumeTransformParams,
    PlumeType,
    RecycledPlumeTransformParams,
    convert_ppmm_to_mol_m2,
    create_simulated_bands,
    load_and_transform_plume_arr,
    randomly_position_sim_plume_by_source,
    resize,
    trim_zero_padding_from_array,
    upscale_rotate_plume,
)
from src.data.sentinel2 import SceneClassificationLabel as SCLabel
from src.data.sentinel2 import Sentinel2Item
from src.tests.generate_test_data import S2_B12_DEFAULT, S2_HAPI_DATA_PATH


class TestPlumeTransformParams(PlumeTransformParams):
    """A customizable PlumeTransformParams for testing."""

    __test__ = False
    target_spatial_resolution: int = 20
    plumes_in_ppm_m: bool = False
    psf_sigma: float = S2_B12_DEFAULT
    upscale: bool = False
    transform: bool = False
    plume_type: PlumeType = PlumeType.RECYCLED

    def __init__(
        self,
        target_spatial_resolution: int = 20,
        plumes_in_ppm_m: bool = False,
        psf_sigma: float = S2_B12_DEFAULT,
        upscale: bool = False,
        transform: bool = False,
    ):
        self.target_spatial_resolution = target_spatial_resolution
        self.plumes_in_ppm_m = plumes_in_ppm_m
        self.psf_sigma = psf_sigma
        self.upscale = upscale
        self.transform = transform


#######################
### SETUP FUNCTIONS ###
#######################


@pytest.fixture(scope="module")
def plumes_dir(local_in_dir: Path) -> Path:
    """Directory to store local plume test outputs."""
    plumes_dir = local_in_dir / "plumes"
    return plumes_dir


@pytest.fixture(scope="module")
def recycled_plume_files(plumes_dir: Path) -> list[str]:
    """Manuallly created list of recycled plume files stored in local and azure test directory."""
    return [
        (plumes_dir / "0002bca8-1a15-44f7-a9b7-c31829b3dba3.tiff").as_posix(),
        (plumes_dir / "00025f2b-2e48-4aef-a8b3-d73c822c02d9.tiff").as_posix(),
        (plumes_dir / "00187d1a-3596-46fb-90c9-c6f0bd1fac01.tiff").as_posix(),
        (plumes_dir / "000938ff-824e-4525-a05a-d035463d0410.tiff").as_posix(),
        "azureml://0000010a-75a5-49ed-9c55-63094f1471b6.tiff",
        "azureml://000006e7-18a1-468a-ac0a-b8e568c178ae.tiff",
        "azureml://00000ddc-1b3f-481a-93cc-f8986d954090.tiff",
        "azureml://00001213-f7d3-4816-b384-1e50a7801e5a.tiff",
    ]


@pytest.fixture(scope="module")
def aviris_plume_files(plumes_dir: Path) -> list[str]:
    """Manuallly created list of AVIRIS plume files stored in local and azure test directory."""
    return [
        "azureml://subscriptions/6e71ce37-b9fe-4c43-942b-cf0f7e78c8ab/resourcegroups/orbio-ml-rg/workspaces/orbio-ml-ml-workspace/datastores/workspaceblobstore/paths/data/aviris/training/ang20230319t181713-A_l3a-ch4-mf-v0_concentrations.tif",
        "azureml://subscriptions/6e71ce37-b9fe-4c43-942b-cf0f7e78c8ab/resourcegroups/orbio-ml-rg/workspaces/orbio-ml-ml-workspace/datastores/workspaceblobstore/paths/data/aviris/training/ang20230319t181713-B_l3a-ch4-mfa-v0_concentrations.tif",
        "azureml://subscriptions/6e71ce37-b9fe-4c43-942b-cf0f7e78c8ab/resourcegroups/orbio-ml-rg/workspaces/orbio-ml-ml-workspace/datastores/workspaceblobstore/paths/data/aviris/training/ang20230319t181244-A_l3a-ch4-mf-v0_concentrations.tif",
        "azureml://subscriptions/6e71ce37-b9fe-4c43-942b-cf0f7e78c8ab/resourcegroups/orbio-ml-rg/workspaces/orbio-ml-ml-workspace/datastores/workspaceblobstore/paths/data/aviris/training/ang20230319t181244-B_l3a-ch4-mfa-v0_concentrations.tif",
    ]


# TODO: replace with Gaussian plume files here
@pytest.fixture(scope="module")
def gaussian_plume_files(plumes_dir: Path) -> list[str]:
    """Manuallly created list of Gaussian plume files stored in local and azure test directory."""
    return [
        "azureml://subscriptions/6e71ce37-b9fe-4c43-942b-cf0f7e78c8ab/resourcegroups/orbio-ml-rg/workspaces/orbio-ml-ml-workspace/datastores/workspaceblobstore/paths/gaussian_plumes/plumes/gaussian_plume_0000001.tif",
        "azureml://subscriptions/6e71ce37-b9fe-4c43-942b-cf0f7e78c8ab/resourcegroups/orbio-ml-rg/workspaces/orbio-ml-ml-workspace/datastores/workspaceblobstore/paths/gaussian_plumes/plumes/gaussian_plume_0000002.tif",
        "azureml://subscriptions/6e71ce37-b9fe-4c43-942b-cf0f7e78c8ab/resourcegroups/orbio-ml-rg/workspaces/orbio-ml-ml-workspace/datastores/workspaceblobstore/paths/gaussian_plumes/plumes/gaussian_plume_0000003.tif",
        "azureml://subscriptions/6e71ce37-b9fe-4c43-942b-cf0f7e78c8ab/resourcegroups/orbio-ml-rg/workspaces/orbio-ml-ml-workspace/datastores/workspaceblobstore/paths/gaussian_plumes/plumes/gaussian_plume_0000004.tif",
    ]


@pytest.fixture(scope="module")
def local_out_dir_plume_outputs(local_out_dir: Path) -> Path:
    """Local directory to store the plume outputs."""
    out_dir_repro = local_out_dir / "recycled_plumes" / "test_sim_plumes"
    out_dir_repro.mkdir(parents=True, exist_ok=True)
    return out_dir_repro


@pytest.fixture(scope="module")
def blob_service_client(ml_client: MLClient) -> BlobServiceClient:
    """Initialize and return an MLClient instance."""
    return initialize_blob_service_client(ml_client)


# TODO: parameterizes with other CRSs
# TODO: ask @zani if it's OK to reproject from/to the same CRS
@pytest.fixture(scope="module")
def crs() -> rasterio.crs.CRS:
    """Return a CRS object."""
    return rasterio.crs.CRS.from_wkt(
        'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]'
    )


# TODO: make a PseudoArray class because this type signature is beefy
@pytest.fixture
def pseudo_arrays(
    request: pytest.FixtureRequest,
) -> tuple[tuple[np.ndarray, np.ndarray], float, np.ndarray, np.ndarray]:
    """Generate pseudo arrays to test resize function."""
    simple_arr = np.array([[1, 2], [3, 4]])
    small_arr = np.array([[1]])
    edge_case_arr = np.array([[1.0, 0.0], [0.0, 0.0]])  # sparse array

    simple_arr_mask = simple_arr > 0
    small_arr_mask = small_arr > 0
    edge_case_arr_mask = edge_case_arr > 0

    test_cases = {
        "identity": (
            (simple_arr, simple_arr_mask),
            1.0,
            simple_arr,
            simple_arr_mask,  # resize factor 1.0 should produce same array
        ),
        "downscale": ((simple_arr, simple_arr_mask), 0.5, np.array([[2.5]]), np.array([[True]])),
        "upscale": (
            (simple_arr, simple_arr_mask),
            2.0,
            np.array([[1, 1.25, 1.75, 2], [1.5, 1.75, 2.25, 2.5], [2.5, 2.75, 3.25, 3.5], [3.0, 3.25, 3.75, 4.0]]),
            np.ones((4, 4), dtype=bool),
        ),
        "small": (
            (small_arr, small_arr_mask),
            0.5,
            small_arr,
            small_arr_mask,  # small array is not resized down to 0
        ),
        "edge_case": (
            (edge_case_arr, edge_case_arr_mask),
            0.5,
            np.array([[0.25]]),
            np.array([[True]]),  # sparse array is not resized down to 0
        ),
    }
    return test_cases[request.param]


######################
### TEST FUNCTIONS ###
######################


def test_create_simulated_bands_recycled(
    local_out_dir_plume_outputs: Path,
    recycled_plume_files: list[str],
    blob_service_client: BlobServiceClient,
    blob_container: str,
) -> None:
    """Test if simulation logic is correct."""
    tile_id = "S2B_MSIL2A_20220228T102849_R108_T32TMT_20220303T082201"
    sentinel2_item = Sentinel2Item.from_id(tile_id)
    cropped_band11 = sentinel2_item.get_band_crop("B11", 0, 0, 128, 128)
    cropped_band12 = sentinel2_item.get_band_crop("B12", 0, 0, 128, 128)
    rng = np.random.default_rng(42)
    rotation_degrees = rng.integers(0, 360)
    plume_transform_params = RecycledPlumeTransformParams(
        psf_sigma=S2_B12_DEFAULT,
        target_spatial_resolution=20,
    )

    plumes = [
        load_and_transform_plume_arr(
            str(filename),
            blob_service_client,
            blob_container,
            plume_transform_params,
            rotation_degrees,
            concentration_rescale_value=0.3,
        )
        for filename in recycled_plume_files
    ]
    cloud_mask = sentinel2_item.get_mask_crop(
        [SCLabel.CLOUD_HIGH_PROBABILITY, SCLabel.CLOUD_MEDIUM_PROBABILITY],
        0,
        0,
        128,
        128,
    )

    sim_swir16, sim_swir22, plumes_inserted_idxs = create_simulated_bands(
        sentinel2_item,
        cropped_band11,
        cropped_band12,
        S2_HAPI_DATA_PATH,
        plumes,
        exclusion_mask_plumes=cloud_mask,
        rng=np.random.default_rng(42),
    )
    print(f"sim band 11 shape: {sim_swir16.shape}, dtype: {sim_swir16.dtype}")
    # test that the simulated bands are not the same as the original bands
    assert not np.allclose(sim_swir16, cropped_band11)
    assert not np.allclose(sim_swir22, cropped_band12)

    img11 = Image.fromarray((sim_swir16 * (255 / 5000)).astype(np.uint8))
    img11.save(str(local_out_dir_plume_outputs / "test_create_simulated_bands__band11.png"))
    img12 = Image.fromarray((sim_swir22 * (255 / 5000)).astype(np.uint8))
    img12.save(str(local_out_dir_plume_outputs / "test_create_simulated_bands__band12.png"))


def test_create_simulated_bands_aviris(
    local_out_dir_plume_outputs: Path,
    aviris_plume_files: list[str],
    blob_service_client: BlobServiceClient,
    blob_container: str,
) -> None:
    """Test if simulation logic is correct."""
    tile_id = "S2B_MSIL2A_20220228T102849_R108_T32TMT_20220303T082201"
    sentinel2_item = Sentinel2Item.from_id(tile_id)
    cropped_band11 = sentinel2_item.get_band_crop("B11", 0, 0, 128, 128)
    cropped_band12 = sentinel2_item.get_band_crop("B12", 0, 0, 128, 128)
    rng = np.random.default_rng(42)
    rotation_degrees = rng.integers(0, 360)
    plume_transform_params = AVIRISPlumeTransformParams(
        psf_sigma=S2_B12_DEFAULT,
        target_spatial_resolution=20,
    )

    plumes = [
        load_and_transform_plume_arr(
            str(filename),
            blob_service_client,
            blob_container,
            plume_transform_params,
            rotation_degrees,
            concentration_rescale_value=0.3,
        )
        for filename in aviris_plume_files
    ]
    cloud_mask = sentinel2_item.get_mask_crop(
        [SCLabel.CLOUD_HIGH_PROBABILITY, SCLabel.CLOUD_MEDIUM_PROBABILITY],
        0,
        0,
        128,
        128,
    )

    sim_swir16, sim_swir22, plumes_inserted_idxs = create_simulated_bands(
        sentinel2_item,
        cropped_band11,
        cropped_band12,
        S2_HAPI_DATA_PATH,
        plumes,
        exclusion_mask_plumes=cloud_mask,
        rng=np.random.default_rng(42),
    )
    print(f"sim band 11 shape: {sim_swir16.shape}, dtype: {sim_swir16.dtype}")
    # test that the simulated bands are not the same as the original bands
    assert not np.allclose(sim_swir16, cropped_band11)
    assert not np.allclose(sim_swir22, cropped_band12)

    img11 = Image.fromarray((sim_swir16 * (255 / 5000)).astype(np.uint8))
    img11.save(str(local_out_dir_plume_outputs / "test_create_simulated_bands__band11.png"))
    img12 = Image.fromarray((sim_swir22 * (255 / 5000)).astype(np.uint8))
    img12.save(str(local_out_dir_plume_outputs / "test_create_simulated_bands__band12.png"))


def test_create_simulated_bands_gaussian_plume(
    local_out_dir_plume_outputs: Path,
    gaussian_plume_files: list[str],
    blob_service_client: BlobServiceClient,
    blob_container: str,
) -> None:
    """Test if simulation logic is correct."""
    tile_id = "S2B_MSIL2A_20220228T102849_R108_T32TMT_20220303T082201"
    sentinel2_item = Sentinel2Item.from_id(tile_id)
    cropped_band11 = sentinel2_item.get_band_crop("B11", 0, 0, 128, 128)
    cropped_band12 = sentinel2_item.get_band_crop("B12", 0, 0, 128, 128)
    rng = np.random.default_rng(42)
    rotation_degrees = rng.integers(0, 360)
    concentration_rescale_value = 0.5
    plume_transform_params = GaussianPlumeTransformParams(
        psf_sigma=S2_B12_DEFAULT,
        target_spatial_resolution=20,
    )

    plumes = [
        load_and_transform_plume_arr(
            str(filename),
            blob_service_client,
            blob_container,
            plume_transform_params,
            rotation_degrees,
            concentration_rescale_value=concentration_rescale_value,
        )
        for filename in gaussian_plume_files
    ]
    cloud_mask = sentinel2_item.get_mask_crop(
        [SCLabel.CLOUD_HIGH_PROBABILITY, SCLabel.CLOUD_MEDIUM_PROBABILITY],
        0,
        0,
        128,
        128,
    )

    sim_swir16, sim_swir22, plumes_inserted_idxs = create_simulated_bands(
        sentinel2_item,
        cropped_band11,
        cropped_band12,
        S2_HAPI_DATA_PATH,
        plumes,
        exclusion_mask_plumes=cloud_mask,
        rng=np.random.default_rng(42),
        position_by_source=True,
    )
    print(f"sim band 11 shape: {sim_swir16.shape}, dtype: {sim_swir16.dtype}")
    # test that the simulated bands are not the same as the original bands
    assert not np.allclose(sim_swir16, cropped_band11)
    assert not np.allclose(sim_swir22, cropped_band12)

    img11 = Image.fromarray((sim_swir16 * (255 / 5000)).astype(np.uint8))
    img11.save(str(local_out_dir_plume_outputs / "test_create_simulated_bands__band11.png"))
    img12 = Image.fromarray((sim_swir22 * (255 / 5000)).astype(np.uint8))
    img12.save(str(local_out_dir_plume_outputs / "test_create_simulated_bands__band12.png"))


def test_randomly_position_sim_plume_by_source_difficult() -> None:
    """Test that the function raises an appropriate error when plume placement fails after max attempts."""
    # Create a plume that's too large for the valid area
    plume_arr = np.array(
        [
            [0.0, 0.0, 0.2, 0.0, 0.0],
            [0.0, 5.0, 5.0, 0.1, 0.0],
            [0.0, 5.0, 5.0, 0.1, 0.0],
            [0.0, 0.0, 0.1, 0.0, 0.0],
            [0.0, 0.0, 0.1, 0.0, 0.0],
        ]
    )

    # Create an exclusion mask that leaves only a tiny valid area
    exclusion_mask = np.array([[1, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]], dtype=bool)

    tile_band = np.ones((4, 4))

    # Use fixed random seed for reproducibility
    rng = np.random.default_rng(42)

    # The function should raise ValueError after 1000 attempts
    methane_enhancement_molperm2, methane_enhancement_mask, plumes_inserted_idxs = (
        randomly_position_sim_plume_by_source(
            sim_plumes=[
                (plume_arr, plume_arr > 0),
            ],
            tile_band=tile_band,
            exclusion_mask_plumes=exclusion_mask,
            rng=rng,
        )
    )
    assert methane_enhancement_molperm2.sum() >= 0.8 * plume_arr.sum()


@pytest.mark.parametrize("pseudo_arrays", ["identity", "downscale", "upscale", "small", "edge_case"], indirect=True)
def test_resize(pseudo_arrays: tuple[tuple[np.ndarray, np.ndarray], float, np.ndarray, np.ndarray]) -> None:
    """Test resizing of pseudo arrays."""
    input_data, zoom, expected_resized, expected_mask = pseudo_arrays
    resized, mask = resize(input_data, zoom)
    assert np.array_equal(resized, expected_resized), "Resized array does not match expected."
    assert np.array_equal(mask, expected_mask), "Mask array does not match expected."


@pytest.mark.parametrize(
    "plume_ppm,plume_mol_m2",
    [
        pytest.param(
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            id="0.0",
        ),
        pytest.param(
            np.ones((2, 2)),
            np.array([[4.229329797958874e-05, 4.229329797958874e-05], [4.229329797958874e-05, 4.229329797958874e-05]]),
            id="1.0",
        ),
        pytest.param(
            np.ones((2, 2)),
            np.array([[8.458659595917748e-05, 8.458659595917748e-05], [8.458659595917748e-05, 8.458659595917748e-05]]),
            id="2.0",
        ),
        pytest.param(
            np.array([[1.0, 0.0], [0.0, 2.0]]),
            np.array([[4.229329797958874e-05, 0.0], [0.0, 8.458659595917748e-05]]),
            id="diagonal",
        ),
    ],
)
def test_convert_ppmm_to_mol_m2(plume_ppm: np.ndarray, plume_mol_m2: np.ndarray) -> None:
    """Test conversion from parts per million (ppm) to mol/m² (mol_m2)."""
    plume_mol_m2 = convert_ppmm_to_mol_m2(plume_ppm)
    assert (plume_mol_m2 == plume_ppm * (UNIT_MULTIPLIER * DENSITY_AIR_GROUND_LEVEL / M_AIR)).all()


@pytest.mark.parametrize(
    "array,expected_shape",
    [
        pytest.param(
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]
            ),
            (3, 1),
            id="1 full columns",
        ),
        pytest.param(
            np.array(
                [
                    [1.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                ]
            ),
            (3, 2),
            id="2 full columns",
        ),
        pytest.param(
            np.array(
                [
                    [1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ),
            (1, 3),
            id="1 full rows",
        ),
        pytest.param(
            np.array(
                [
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0],
                ]
            ),
            (2, 3),
            id="2 full rows",
        ),
        pytest.param(
            np.array(
                [
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                ]
            ),
            (3, 3),
            id="3 full columns & rows",
        ),
        pytest.param(
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            (3, 3),
            id="diagonal",
        ),
        pytest.param(
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ),
            (1, 1),
            id="single 1",
        ),
        pytest.param(
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ),
            (1, 1),
            id="single 2",
        ),
        pytest.param(
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            (1, 1),
            id="single 3",
        ),
        pytest.param(
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            (3, 3),
            id="separated",
        ),
    ],
)
def test_trim_zero_padding_from_array(array: np.ndarray, expected_shape: tuple[int, ...]) -> None:
    """Test removal of whitespace around a plume (i.e. rows and columns without plume information)."""
    plume_reborn = trim_zero_padding_from_array(array)

    assert plume_reborn.shape == expected_shape


@pytest.mark.skip(reason="unimplemented")
def test_randomly_position_sim_plume() -> None:
    """Test random positioning of a plume."""
    pass


def test_identity_transform_rescaling(crs: rasterio.crs.CRS) -> None:
    """Test that the affine transforms result in a no op and the original transform is preserved."""
    transform = Affine.identity()
    scaling_transform = Affine.scale(1.0)
    rotation_transform = Affine.rotation(0.0)
    translation_transform = Affine.translation(0.0, 0.0)

    dst_transform = transform * rotation_transform * translation_transform * scaling_transform

    plume = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    dst_plume = np.zeros((3, 3))
    reproject(
        source=plume,
        destination=dst_plume,
        src_transform=transform,
        src_crs=crs,
        dst_transform=dst_transform,
        dst_crs=crs,
        resampling=Resampling.bilinear,
    )
    assert transform == dst_transform
    assert (dst_plume == plume).all()


@pytest.mark.parametrize(
    "plume,transform,transform_params,rotation_deg,concentration_rescale_value",
    [
        pytest.param(
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            Affine.identity(),
            TestPlumeTransformParams(
                target_spatial_resolution=1,
                plumes_in_ppm_m=False,
                psf_sigma=0.0,  # no op the gaussian filter
                upscale=False,
                transform=False,
            ),
            0,
            1.0,
            id="identity",
        ),
        pytest.param(
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            Affine.identity(),
            TestPlumeTransformParams(
                target_spatial_resolution=1,
                plumes_in_ppm_m=False,
                psf_sigma=0.0,  # no op the gaussian filter
                upscale=False,
                transform=True,
            ),
            90,
            1.0,
            id="rotate 90",
        ),
        pytest.param(
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            Affine.identity(),
            TestPlumeTransformParams(
                target_spatial_resolution=1,
                plumes_in_ppm_m=False,
                psf_sigma=0.0,  # no op the gaussian filter
                upscale=False,
                transform=True,
            ),
            180,
            1.0,
            id="rotate 180",
        ),
        pytest.param(
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            Affine.identity(),
            TestPlumeTransformParams(
                target_spatial_resolution=1,
                plumes_in_ppm_m=False,
                psf_sigma=0.0,  # no op the gaussian filter
                upscale=False,
                transform=False,  # do not rotate
            ),
            -15,
            1.0,
            id="rotate -15",
        ),
        pytest.param(
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            Affine(a=20.0, b=0.0, c=0.0, d=0.0, e=-20.0, f=0.0),
            TestPlumeTransformParams(
                target_spatial_resolution=20,
                plumes_in_ppm_m=False,
                psf_sigma=0.0,  # no op the gaussian filter
                upscale=False,
                transform=True,
            ),
            0,
            1.0,
            id="rescale spatial resolution [20->20]",
        ),
        pytest.param(
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            Affine(a=10.0, b=0.0, c=0.0, d=0.0, e=-10.0, f=0.0),
            TestPlumeTransformParams(
                target_spatial_resolution=20,  # 10 -> 20 - reducing the resolution by a factor of 2*2
                plumes_in_ppm_m=False,
                psf_sigma=0.0,  # no op the gaussian filter
                upscale=False,
                transform=True,
            ),
            0,
            # we're reducing the resolution by a factor of 2*2 so we need to rescale by 4
            4.0,
            id="rescale spatial resolution [10->20]",
        ),
        pytest.param(
            np.array(
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                ]
            ),
            Affine(a=10.0, b=0.0, c=0.0, d=0.0, e=-10.0, f=0.0),
            TestPlumeTransformParams(
                target_spatial_resolution=40,  # 10 -> 40 - reducing the resolution by a factor of 4*4
                plumes_in_ppm_m=False,
                psf_sigma=0.0,  # no op the gaussian filter
                upscale=False,
                transform=True,
            ),
            0,
            # we're reducing the resolution by a factor of 4*4 so we need to rescale by 16
            16.0,
            id="rescale spatial resolution [10->40]",
        ),
    ],
)
def test_upscale_rotate_plume_no_gaussian_filter(
    plume: np.ndarray,
    crs: rasterio.crs.CRS,
    transform: Affine,
    transform_params: PlumeTransformParams,
    rotation_deg: int,
    concentration_rescale_value: float,
) -> None:
    """Test that the plume transform params (except for gaussian filter).

    - test conservation of mass - tranformed plume should have ~the same amount of methane
        sum(pixel values) * pixel area (resolution squared)
    - test that the plume is all within the destination band after transforming
    """
    upscaled_plume = upscale_rotate_plume(plume, crs, transform, transform_params, rotation_deg)

    plume_concentration_recaled = plume * concentration_rescale_value

    spatial_resolution = transform.a
    original_methane_IME = plume_concentration_recaled.sum() * spatial_resolution**2
    upscaled_methane_IME = upscaled_plume.sum() * transform_params.target_spatial_resolution**2

    math.isclose(upscaled_methane_IME, original_methane_IME * concentration_rescale_value, rel_tol=0.05)


@pytest.mark.parametrize(
    "plume,transform,transform_params,rotation_deg,concentration_rescale_value",
    [
        pytest.param(
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            Affine.identity(),
            TestPlumeTransformParams(
                target_spatial_resolution=1,
                plumes_in_ppm_m=True,
                psf_sigma=0.0,  # no op the gaussian filter
                upscale=False,
                transform=False,
            ),
            0,
            1.0,
            id="plume in ppm_m",
        ),
    ],
)
def test_upscale_rotate_plume_no_gaussian_filter_ppmm_to_mol_m2(
    plume: np.ndarray,
    crs: rasterio.crs.CRS,
    transform: Affine,
    transform_params: PlumeTransformParams,
    rotation_deg: int,
    concentration_rescale_value: float,
) -> None:
    """Test that the plume transform params (except for gaussian filter).

    - test conservation of mass - tranformed plume should have ~the same amount of methane
        sum(pixel values) * pixel area (resolution squared)
    - test that the plume is all within the destination band after transforming
    """
    upscaled_plume = upscale_rotate_plume(plume, crs, transform, transform_params, rotation_deg)

    plume_concentration_recaled = plume * concentration_rescale_value

    spatial_resolution = transform.a
    original_methane = plume_concentration_recaled.sum() * spatial_resolution**2
    upscaled_methane = upscaled_plume.sum() * transform_params.target_spatial_resolution**2

    math.isclose(upscaled_methane, original_methane, rel_tol=0.05)


@pytest.mark.parametrize(
    "plume,transform,transform_params,rotation_deg",
    [
        pytest.param(
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            Affine.identity(),
            TestPlumeTransformParams(
                target_spatial_resolution=1,
                plumes_in_ppm_m=False,
                psf_sigma=0.0,  # no op the gaussian filter
                upscale=False,
                transform=False,
            ),
            0,
            id="identity",
        ),
        pytest.param(
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            Affine.identity(),
            TestPlumeTransformParams(
                target_spatial_resolution=1,
                plumes_in_ppm_m=False,
                psf_sigma=0.0,  # no op the gaussian filter
                upscale=True,
                transform=True,
            ),
            90,
            id="rotate 90",
        ),
        pytest.param(
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            Affine.identity(),
            TestPlumeTransformParams(
                target_spatial_resolution=1,
                plumes_in_ppm_m=False,
                psf_sigma=0.0,  # no op the gaussian filter
                upscale=True,
                transform=True,
            ),
            180,
            id="rotate 180",
        ),
        pytest.param(
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            Affine.identity(),
            TestPlumeTransformParams(
                target_spatial_resolution=1,
                plumes_in_ppm_m=False,
                psf_sigma=0.0,  # no op the gaussian filter
                upscale=True,
                transform=True,  # do not rotate
            ),
            5,
            id="rotate 45",
        ),
        pytest.param(
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            Affine(a=20.0, b=0.0, c=0.0, d=0.0, e=-20.0, f=0.0),
            TestPlumeTransformParams(
                target_spatial_resolution=20,
                plumes_in_ppm_m=False,
                psf_sigma=0.0,  # no op the gaussian filter
                upscale=False,
                transform=True,
            ),
            0,
            id="rescale spatial resolution [20->20]",
        ),
        pytest.param(
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            Affine(a=10.0, b=0.0, c=0.0, d=0.0, e=-10.0, f=0.0),
            TestPlumeTransformParams(
                target_spatial_resolution=20,  # 10 -> 20 - reducing the resolution by a factor of 2*2
                plumes_in_ppm_m=False,
                psf_sigma=0.0,  # no op the gaussian filter
                upscale=False,
                transform=True,
            ),
            0,
            id="rescale spatial resolution [10->20]",
        ),
        pytest.param(
            np.array(
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                ]
            ),
            Affine(a=10.0, b=0.0, c=0.0, d=0.0, e=-10.0, f=0.0),
            TestPlumeTransformParams(
                target_spatial_resolution=40,  # 10 -> 40 - reducing the resolution by a factor of 4*4
                plumes_in_ppm_m=False,
                psf_sigma=1.0,
                upscale=False,
                transform=True,
            ),
            0,
            id="rescale spatial resolution [10->40]",
        ),
        pytest.param(
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            Affine.identity(),
            TestPlumeTransformParams(
                target_spatial_resolution=1,
                plumes_in_ppm_m=False,
                psf_sigma=1.0,
                upscale=False,
                transform=False,
            ),
            0,
            id="gaussian filter",
        ),
    ],
)
def test_upscale_rotate_plume_no_ppmm_to_mol_m2_conversion(
    plume: np.ndarray,
    crs: rasterio.crs.CRS,
    transform: Affine,
    transform_params: PlumeTransformParams,
    rotation_deg: int,
) -> None:
    """Test that the plume transform params (except for converting from ppmm to mol/m2).

    - test conservation of mass - tranformed plume should have ~the same amount of methane
        sum(pixel values) * pixel area (resolution squared)
    - test that the plume is all within the destination band after transforming
    """
    upscaled_plume = upscale_rotate_plume(plume, crs, transform, transform_params, rotation_deg)

    spatial_resolution = transform.a
    original_methane = plume.sum() * spatial_resolution**2
    upscaled_methane = upscaled_plume.sum() * transform_params.target_spatial_resolution**2

    math.isclose(upscaled_methane, original_methane, rel_tol=0.05)


@pytest.mark.parametrize(
    "plume,transform,transform_params,concentration_rescale_value",
    [
        pytest.param(
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            Affine.identity(),
            TestPlumeTransformParams(
                target_spatial_resolution=1,
                plumes_in_ppm_m=False,
                psf_sigma=1.0,
                upscale=False,
                transform=False,
            ),
            1.0,
            id="no upscaling",
        ),
        pytest.param(
            np.array(
                [
                    [1.0, np.nan, np.nan],
                    [np.nan, 1.0, np.nan],
                    [np.nan, np.nan, 1.0],
                ]
            ),
            Affine.identity(),
            TestPlumeTransformParams(
                target_spatial_resolution=1,
                plumes_in_ppm_m=False,
                psf_sigma=1.0,
                upscale=False,
                transform=False,
            ),
            1.0,
            id="convert NaN values",
        ),
        pytest.param(
            np.array(
                [
                    [1.0, np.nan, np.nan],
                    [np.nan, 1.0, np.nan],
                    [np.nan, np.nan, 1.0],
                ]
            ),
            Affine.identity(),
            TestPlumeTransformParams(
                target_spatial_resolution=1,
                plumes_in_ppm_m=False,
                psf_sigma=1.0,
                upscale=True,
                transform=True,
            ),
            1.0,
            id="convert NaN values",
        ),
        pytest.param(
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            Affine.identity(),
            TestPlumeTransformParams(
                target_spatial_resolution=1,
                plumes_in_ppm_m=False,
                psf_sigma=1.0,
                upscale=True,
                transform=False,  # no rotation
            ),
            1.0,
            id="psf_sigma=1.0",
        ),
        pytest.param(
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            Affine.identity(),
            TestPlumeTransformParams(
                target_spatial_resolution=1,
                plumes_in_ppm_m=False,
                psf_sigma=0.0,
                upscale=True,
                transform=True,
            ),
            1.0,
            id="psf_sigma=0.0",
        ),
    ],
)
def test_load_transform_plume_array(
    plume: np.ndarray,
    transform: Affine,
    transform_params: PlumeTransformParams,
    blob_service_client: BlobServiceClient,
    blob_container: str,
    tmp_path: Path,
    concentration_rescale_value: float,
) -> None:
    """Test the loading and transformation of a plume array.

    - test conservation of mass - tranformed plume should have ~the same amount of methane
        sum(pixel values) * pixel area (resolution squared)
    - test that the plume is all within the destination band after transforming
    """
    # Create a test plume geotiff file
    crs = rasterio.crs.CRS.from_wkt(
        'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]'
    )

    tmp_path.mkdir(parents=True, exist_ok=True)
    filename = tmp_path / "test_plume.tif"

    with rasterio.open(
        filename,
        "w",
        driver="GTiff",
        height=plume.shape[0],
        width=plume.shape[1],
        count=1,
        dtype=str(plume.dtype),
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(plume, 1)

    # Test Code
    rotation_degrees = 90
    upscaled_plume = load_and_transform_plume_arr(
        filename, blob_service_client, blob_container, transform_params, rotation_degrees, concentration_rescale_value
    )

    plume = np.nan_to_num(plume, nan=0.0)

    spatial_resolution = transform.a
    original_methane_IME = plume.sum() * spatial_resolution**2
    upscaled_methane_IME = upscaled_plume.sum() * transform_params.target_spatial_resolution**2
    math.isclose(upscaled_methane_IME, original_methane_IME * concentration_rescale_value, rel_tol=0.05)


@pytest.mark.parametrize(
    "plume,transform,transform_params, rotation_degrees,concentration_rescale_value",
    [
        pytest.param(
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            Affine(a=20.0, b=0.0, c=0.0, d=0.0, e=-20.0, f=0.0),
            RecycledPlumeTransformParams(psf_sigma=S2_B12_DEFAULT, target_spatial_resolution=20),
            0,
            1.0,
            id="rotation 0 (should skip rotation)",
        ),
        pytest.param(
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            Affine(a=20.0, b=0.0, c=0.0, d=0.0, e=-20.0, f=0.0),
            RecycledPlumeTransformParams(psf_sigma=S2_B12_DEFAULT, target_spatial_resolution=20),
            90,
            1.0,
            id="rotation 90 (should skip rotation)",
        ),
    ],
)
def test_load_transform_recycled_plume(
    plume: np.ndarray,
    transform: Affine,
    transform_params: PlumeTransformParams,
    rotation_degrees: int,
    concentration_rescale_value: float,
    blob_service_client: BlobServiceClient,
    blob_container: str,
    tmp_path: Path,
) -> None:
    """Test the loading and transformation of a plume array.

    - test conservation of mass - tranformed plume should have ~the same amount of methane
        sum(pixel values) * pixel area (resolution squared)
    - test that the plume is all within the destination band after transforming
    """
    # Create a test plume geotiff file
    crs = rasterio.crs.CRS.from_wkt(
        'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]'
    )

    (tmp_path / "recycled").mkdir(parents=True, exist_ok=True)
    filename = tmp_path / "test_plume.tif"

    with rasterio.open(
        filename,
        "w",
        driver="GTiff",
        height=plume.shape[0],
        width=plume.shape[1],
        count=1,
        dtype=str(plume.dtype),
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(plume, 1)

    # Test Code
    upscaled_plume = load_and_transform_plume_arr(
        filename, blob_service_client, blob_container, transform_params, rotation_degrees, concentration_rescale_value
    )

    spatial_resolution = transform.a
    original_methane = plume.sum() * spatial_resolution**2
    upscaled_methane = upscaled_plume.sum() * transform_params.target_spatial_resolution**2

    math.isclose(upscaled_methane, original_methane, rel_tol=0.05)


@pytest.mark.parametrize(
    "plume,transform,transform_params, rotation_degrees,concentration_rescale_value",
    [
        pytest.param(
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            Affine(a=10.0, b=0.0, c=0.0, d=0.0, e=-10.0, f=0.0),
            AVIRISPlumeTransformParams(
                psf_sigma=S2_B12_DEFAULT,
                target_spatial_resolution=20,
            ),
            0,
            10.0,
            id="rotation 0",
        ),
        pytest.param(
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            Affine(a=10.0, b=0.0, c=0.0, d=0.0, e=-10.0, f=0.0),
            AVIRISPlumeTransformParams(
                psf_sigma=S2_B12_DEFAULT,
                target_spatial_resolution=20,
            ),
            90,
            10.0,
            id="rotation 90",
        ),
    ],
)
def test_load_transform_aviris_plume(
    plume: np.ndarray,
    transform: Affine,
    transform_params: PlumeTransformParams,
    rotation_degrees: int,
    concentration_rescale_value: float,
    blob_service_client: BlobServiceClient,
    blob_container: str,
    tmp_path: Path,
) -> None:
    """Test the loading and transformation of a plume array.

    - test conservation of mass - tranformed plume should have ~the same amount of methane
        sum(pixel values) * pixel area (resolution squared)
    - test that the plume is all within the destination band after transforming
    """
    # Create a test plume geotiff file
    crs = rasterio.crs.CRS.from_wkt(
        'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]'
    )

    (tmp_path / "recycled").mkdir(parents=True, exist_ok=True)
    filename = tmp_path / "test_plume.tif"

    with rasterio.open(
        filename,
        "w",
        driver="GTiff",
        height=plume.shape[0],
        width=plume.shape[1],
        count=1,
        dtype=str(plume.dtype),
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(plume, 1)

    # Test Code
    upscaled_plume = load_and_transform_plume_arr(
        filename, blob_service_client, blob_container, transform_params, rotation_degrees, concentration_rescale_value
    )

    plume = convert_ppmm_to_mol_m2(plume)
    plume = plume * concentration_rescale_value

    spatial_resolution = transform.a
    original_methane_IME = plume.sum() * spatial_resolution**2
    upscaled_methane_IME = upscaled_plume.sum() * transform_params.target_spatial_resolution**2

    math.isclose(upscaled_methane_IME, original_methane_IME * concentration_rescale_value, rel_tol=0.05)
