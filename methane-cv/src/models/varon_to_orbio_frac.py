import argparse
import itertools
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

# Constants
SUBSCRIPTION_ID = "6e71ce37-b9fe-4c43-942b-cf0f7e78c8ab"
RESOURCE_GROUP = "orbio-ml-rg"
WORKSPACE_NAME = "orbio-ml-ml-workspace"
LOOKUP_TABLE_PATH = "src/models/varon_to_orbio_frac_lookup.csv"


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a Parquet file in Azure Blob Storage.
    """
    from azureml.core import Dataset, Datastore, Workspace

    # Initialize Azure Workspace
    workspace = Workspace(SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME)

    datastore = Datastore.get(workspace, "workspaceblobstore")
    dataset = Dataset.Tabular.from_parquet_files(path=(datastore, file_path))
    return dataset.to_pandas_dataframe()


def read_bytes_into_xarray(input_array: bytes, dtype, bands: list[str], size: int) -> xr.DataArray:
    """
    Convert a bytes object into an xarray DataArray.
    """
    input_arr_shape = (len(bands), size, size)
    arr = np.frombuffer(input_array, dtype=dtype).reshape(input_arr_shape)
    da = xr.DataArray(arr, coords={"band": bands}, dims=["band", "x", "y"])
    return da


def read_bytes_into_array(input_array: bytes, dtype, size: int | None = None) -> np.ndarray:
    """
    Convert a bytes object into a numpy array.
    """
    if size:
        input_arr_shape = (size, size)
        arr = np.frombuffer(input_array, dtype=dtype).reshape(input_arr_shape)
    else:
        arr = np.frombuffer(input_array, dtype=dtype)
    return arr


def preprocess_data(df: pd.DataFrame):
    """
    Preprocess each row in the input DataFrame by converting binary data into xarray and numpy arrays.
    """
    df["crop_before"] = [
        read_bytes_into_xarray(row.crop_before, dtype=np.float32, bands=row.bands, size=row.size)
        for row in df.itertuples(index=False)
    ]
    df["crop_main"] = [
        read_bytes_into_xarray(row.crop_main, dtype=np.float32, bands=row.bands, size=row.size)
        for row in df.itertuples(index=False)
    ]
    df["target_frac"] = [
        read_bytes_into_array(row.target_frac, dtype=np.float32, size=row.size) for row in df.itertuples(index=False)
    ]
    return df


# TODO: how to apply mask to cropped images?
def apply_mask(df: pd.DataFrame):
    """
    Apply cloud and data quality amsk to cropped images.
    """
    df["mask_before"] = [
        read_bytes_into_array(row.mask_before, dtype=np.bool8, size=row.size) for row in df.itertuples(index=False)
    ]
    df["mask_main"] = [
        read_bytes_into_array(row.mask_main, dtype=np.bool8, size=row.size) for row in df.itertuples(index=False)
    ]
    # Apply the mask to each band in crop_before
    df["crop_before"] = [apply_mask_to_bands(row.crop_before, row.mask_before) for row in df.itertuples(index=False)]
    # Apply the mask to each band in crop_main
    df["crop_main"] = [apply_mask_to_bands(row.crop_main, row.mask_main) for row in df.itertuples(index=False)]

    return df


def apply_mask_to_bands(crop_image: xr.DataArray, mask: np.ndarray) -> xr.DataArray:
    """
    Apply the mask to each band in the input image.
    """
    masked_image = crop_image.where(
        ~mask, np.nan
    )  # where keeps elements when condition is true, replaces with np.nan when false
    return masked_image


def apply_orbio_light_diff_weight(target_arr: np.ndarray, reference_arr: np.ndarray) -> np.ndarray:
    """
    Apply a scaling factor to the reference satellite image based on the target image's brightness.
    """
    # Avoid division by zero by replacing zeros with NaN
    reference_arr = np.where(reference_arr == 0, np.nan, reference_arr)
    pixel_weights = target_arr / reference_arr
    img_weight = np.nanmedian(pixel_weights)
    img_weight = np.where(np.isnan(img_weight), 0, img_weight)
    return reference_arr * img_weight


def calculate_frac(target_arr: np.ndarray, reference_arr: np.ndarray) -> np.ndarray:
    """
    Compute the fractional difference (FRAC) between the target and reference band data.
    """
    # Avoid division by zero by replacing zeros in reference_arr with NaN
    reference_arr_safe = np.where(reference_arr == 0, np.nan, reference_arr)
    frac = (target_arr - reference_arr) / reference_arr_safe
    frac = np.where(np.isnan(frac), 0, frac)
    return frac


def calculate_mb_ratio(band_data: xr.DataArray) -> np.ndarray:
    """
    Calculate the normalised B12/B11 ratio.
    """
    b12 = band_data.sel(band="B12")
    b11 = band_data.sel(band="B11")
    # Avoid division by zero by replacing zeros in B11 with NaN
    b11_safe = np.where(b11 == 0, np.nan, b11)
    ratio = (b12 - b11) / b11_safe
    ratio = np.where(np.isnan(ratio), 0, ratio)
    return ratio


def caclculate_orbio_band_ratio(band_data: xr.DataArray) -> np.ndarray:
    """
    Calculate orbio band ratio B12/B11
    """
    b12 = band_data.sel(band="B12")
    b11 = band_data.sel(band="B11")
    # Avoid division by zero by replacing zeros in B11 with NaN
    b11_safe = np.where(b11 == 0, np.nan, b11)
    return b12 / b11_safe


def get_orbio_frac_from_varon(varon_frac: np.ndarray, lookup_table_path: str = LOOKUP_TABLE_PATH) -> np.ndarray:
    """
    Get corresponding Orbio Frac values for a given array of Varon Frac values using the lookup table.
    """
    lookup_table = pd.read_csv(lookup_table_path)
    sorted_lookup_table = lookup_table.sort_values(by="varon_frac")

    orbio_frac = np.interp(varon_frac, sorted_lookup_table["varon_frac"], sorted_lookup_table["orbio_frac"])

    return orbio_frac


def regularise_varon_frac_to_zero_model(varon_frac: np.ndarray, lookup: bool) -> np.ndarray:
    """
    Regularizes Varon frac values to align with physical limits and realistic methane measurements.

    Frac values are theoretically expected to range between -1 and 0, assuming the reference image
    does not contain methane. Positive frac values are therefore clipped to 0. When the 'lookup'
    parameter is True, values are further adjusted based on a Varon to Orbio frac lookup table.
    According to this table, 1000 mol/m2 of methane (an extremely large amount) corresponds to
    -0.7715 Varon Frac and -0.6149 Orbio Frac values. Therefore, Varon Frac values less than
    -0.7715 are adjusted to -0.6149 when 'lookup' is True, or clipped to -0.7715 otherwise.
    """
    varon_frac = np.where(varon_frac > 0, 0, varon_frac)
    if lookup:
        varon_frac = np.where(varon_frac < -0.7715, -0.6149, varon_frac)  # values taken from lookup table
    else:
        varon_frac = np.where(varon_frac < -0.7715, -0.7715, varon_frac)
    return varon_frac


def calculate_model_frac(
    model_type: str,
    lookup: bool,
    target_data: xr.DataArray,
    reference_data: xr.DataArray | None = None,
) -> np.ndarray:
    """
    Calculate the FRAC for Varon's Single Band Multi Pass (SBMP), Multi Band
    Single Pass (MBSP), and Multi Band Multi Pass (MBMP) methods.
    """
    if model_type == "sbmp":
        b12_target = target_data.sel(band="B12")
        b12_reference = reference_data.sel(band="B12")
        b12_reference = apply_orbio_light_diff_weight(b12_target, b12_reference)
        varon_frac = calculate_frac(b12_target, b12_reference)
        varon_frac = regularise_varon_frac_to_zero_model(varon_frac, lookup)
        if lookup:
            return get_orbio_frac_from_varon(varon_frac)
        else:
            return varon_frac
    elif model_type == "mbsp":
        varon_frac = calculate_mb_ratio(target_data)
        varon_frac = regularise_varon_frac_to_zero_model(varon_frac, lookup)
        if lookup:
            return get_orbio_frac_from_varon(varon_frac)
        else:
            return varon_frac
    elif model_type == "mbmp":
        target_mb_ratio = calculate_mb_ratio(target_data)
        reference_mb_ratio = calculate_mb_ratio(reference_data)
        reference_mb_ratio = apply_orbio_light_diff_weight(target_mb_ratio, reference_mb_ratio)
        varon_frac = calculate_frac(target_mb_ratio, reference_mb_ratio)
        varon_frac = regularise_varon_frac_to_zero_model(varon_frac, lookup)
        if lookup:
            return get_orbio_frac_from_varon(varon_frac)
        else:
            return varon_frac
    elif model_type == "zero":
        target_mb_ratio = calculate_mb_ratio(target_data)
        return np.zeros(target_mb_ratio.shape)
    else:
        raise ValueError("Invalid model type")


def mean_squared_error(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """
    Calculate the mean squared error between observed and predicted arrays.
    """
    if observed.ndim == 1:
        observed = np.stack(observed)
    if predicted.ndim == 1:
        predicted = np.stack(predicted)

    if observed.shape != predicted.shape:
        raise ValueError("Arrays must have the same shape")

    error = np.subtract(observed, predicted)
    squared_error = np.square(error)
    mse = np.nanmean(squared_error, axis=(1, 2))
    return mse


def process_file(file_path: str, model_type: str, mask: bool, lookup: bool) -> np.ndarray:
    """
    Process a single Parquet file and compute the mean squared error for the specified model type.
    """
    df = load_data(file_path)
    df_preprocessed = preprocess_data(df)
    if mask:
        my_logger.info("Applying cloud and data quality masks")
        df_preprocessed = apply_mask(df_preprocessed)
    tqdm.pandas(desc="    Processing rows")
    df_preprocessed["model_output"] = df_preprocessed.progress_apply(
        lambda row: calculate_model_frac(model_type, lookup, row.crop_main, row.crop_before),
        axis=1,
    )
    target_fracs = np.stack(df_preprocessed["target_frac"].to_numpy())
    model_outputs = np.stack(df_preprocessed["model_output"].to_numpy())
    mse = mean_squared_error(target_fracs, model_outputs)
    return mse


def process_all_files_for_model(file_paths: list[str], model_type: str, mask: bool, lookup: bool) -> list[np.ndarray]:
    """
    Process multiple files and compute the mean squared error for each file for the specified model type.
    """
    results = []
    for file_path in tqdm(file_paths, desc="Processing files"):
        mse = process_file(file_path, model_type, mask, lookup)
        results.append(mse)
    return results


def save_results_to_csv(results: list[np.ndarray], model_type: str, mask: bool, csv_filename: str):
    """
    Save the mean squared error results to a CSV file for the specified model type.
    """
    # Flatten the list of lists into a single list
    flattened_results = list(itertools.chain(*results))

    # Create a DataFrame with the flattened results
    df = pd.DataFrame(flattened_results, columns=[f"{model_type}_mse"])

    mask_description = "with mask" if mask else "without mask"
    my_logger.info(f"Saving {model_type} {mask_description} MSE results to {csv_filename}")

    Path(csv_filename).parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(csv_filename, index=False)


# Set up logging
logging.basicConfig(level=logging.WARNING)
my_logger = logging.getLogger("baseline_models")
my_logger.setLevel(logging.INFO)


def main():
    from azureml.core import Dataset, Datastore, Workspace

    # Set up argument parser
    parser = argparse.ArgumentParser(
        prog="varon_models",
        description=(
            """
            Compute Mean Squared Error (MSE) values for Varon's baseline methods 
            (SBMP, MBSP, MBMP) by comparing Varon's fractional difference (FRAC) 
            with target FRAC across training data. Optionally, mask invalid pixels 
            due to cloud and data quality issues.
            """
        ),
    )

    parser.add_argument(
        "-training_dataset",
        "--training_dataset",
        type=str,
        required=True,
        help="Specify training dataset to compute MSE metrics for. Should be prefixed with 'data/'",
    )

    parser.add_argument(
        "-method",
        "--method",
        nargs="+",
        required=True,
        choices=["sbmp", "mbsp", "mbmp", "zero"],
        help="List of methods to run (sbmp, mbsp, mbmp, zero)",
    )

    parser.add_argument(
        "-mask",
        "--mask",
        action="store_true",
        help="Apply cloud and data quality mask on input images (default: False)",
    )

    parser.add_argument(
        "-lookup",
        "--lookup",
        action="store_true",
        help="Translate Varon Fracs to Orbio Frac using the lookup table (default: False)",
    )

    # Parse arguments
    args = parser.parse_args()
    methods = args.method
    FOLDER_PATH = args.training_dataset

    # Initialize Azure Workspace
    workspace = Workspace(SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME)

    # Get the list of Parquet files in the folder
    datastore = Datastore.get(workspace, "workspaceblobstore")
    dataset = Dataset.File.from_files(path=(datastore, FOLDER_PATH))
    parquet_files = dataset.to_path()
    parquet_files = [fp for fp in parquet_files if fp != "/metadata.json"]
    file_paths = [FOLDER_PATH + path for path in parquet_files]

    # Define the root directory and training data version for saving results
    root_dir = os.path.dirname(os.path.abspath(__file__))
    training_data_version = "/".join(FOLDER_PATH.split("/")[1:])

    # Process files for each selected model and save results
    for method in methods:
        mask = args.mask
        lookup = args.lookup
        my_logger.info(
            f"Running {method.upper()} with {'--mask' if mask else ''} {'--lookup' if lookup else ''} on training data in {FOLDER_PATH}"
        )
        results = process_all_files_for_model(file_paths, method, mask, lookup)
        results_path = os.path.join(
            root_dir,
            "results",
            training_data_version,
            f"{method}_{'with_mask' if mask else 'without_mask'}_{'lookup' if lookup else ''}_mse_results.csv",
        )
        save_results_to_csv(results, method, mask, results_path)


if __name__ == "__main__":
    main()
