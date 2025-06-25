"""
Convert the CSVs in `src/data/recycled_plumes/no_methane_tiles/csv_files` into a unified parquet schema.

We use the parquet files in the `run_data_generation_job.py` script to avoid any data preprocessing logic.
"""

from pathlib import Path

import pandas as pd


def extract_mgrs_from_tile_name(tile_name) -> str:
    """tile_name is in format {mgrs}_{index}."""
    return tile_name.split("_")[0]


def parse_dates(date_series) -> pd.Series:
    """Attempt to parse dates into correct format."""
    # Define possible date formats in any csv
    date_formats = ["%Y/%m/%d", "%Y-%m-%d"]

    # Try parsing with each format
    for fmt in date_formats:
        try:
            return pd.to_datetime(date_series, format=fmt).dt.strftime("%Y-%m-%d")
        except Exception:
            # If this format fails, try the next one
            continue
    # If no format was successful, raise an exception
    raise ValueError(f"Could not parse dates into '%Y-%m-%d' format. Tried these formats: {date_formats}")


def convert_csv_to_parquet(file_path: Path, output_dir: Path, is_validation: bool) -> None:
    """Convert csv into parquet file."""
    # Read the CSV file
    df = pd.read_csv(file_path)

    if "mgrs" not in df.columns:
        df["mgrs"] = df["tile_name"].apply(extract_mgrs_from_tile_name)

    # Standardize date format
    df["date"] = parse_dates(df["date"])

    # Check if columns are present
    validation_columns = ["mgrs", "date", "cloud_cover_min", "cloud_cover_max", "season", "four_items_dt_50"]
    regular_columns = ["mgrs", "date"]
    required_columns = validation_columns if is_validation else regular_columns
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Explicit type conversions
    if is_validation:
        type_dict = {
            col: "string" for col in df.columns if col not in ["cloud_cover_min", "cloud_cover_max", "four_items_dt_50"]
        }
        type_dict.update({"cloud_cover_min": "float32", "cloud_cover_max": "float32", "four_items_dt_50": "bool"})
        df = df.astype(type_dict)
    else:
        df = df.astype({col: "string" for col in df.columns})

    # Save to Parquet
    output_file = output_dir / Path(file_path).name.replace(".csv", ".parquet")
    df.to_parquet(output_file, index=False)
    print(f"Converted {file_path} to {output_file}")


def main():
    """
    Convert all csvs in `no_methane_tiles/csv_files` into parquet file.

    Parquet files are saved in `no_methane_tiles/parquet_files`. Specify which csv is the validation master file so the
    parquet file schema is adjusted accordingly.
    """
    csv_dir = Path("src/data/recycled_plumes/no_methane_tiles/csv_files")
    output_dir = Path("src/data/recycled_plumes/no_methane_tiles/parquet_files")
    validation_master_file = "validation_master_set_mgrs_dates_4items.csv"

    # Convert all CSV files in the directory
    for file_path in csv_dir.iterdir():
        if file_path.suffix == ".csv":
            is_validation = file_path.name == validation_master_file
            convert_csv_to_parquet(file_path, output_dir, is_validation)


if __name__ == "__main__":
    main()
