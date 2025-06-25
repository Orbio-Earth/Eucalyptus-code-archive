"""Create the Recycled Plumes dataset from Orbio retrievals."""

import json
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile

from tqdm import tqdm

from src.azure_wrap.blob_storage_sdk_v2 import download_from_blob, upload_dir

prefix_on_datastore = "orbio-data-exports-dev/orbio-data-exports/usa_v2.2.0"
local_dir = Path.home() / "localfiles" / "data" / "orbio-data-exports" / "usa_v2.2.0"
upload_prefix = "orbio-data-exports-dev/unzipped_complete2/usa_v2.2.0"


def process_zip_file(args: tuple[str, str]) -> list[str]:
    """Extract zip contents to temp dir, upload to Azure, return list of extracted Azure paths."""
    zip_file, upload_prefix = args
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        shutil.unpack_archive(zip_file, temp_path)
        upload_dir(temp_path, upload_prefix)

        with ZipFile(zip_file, "r") as zip:
            extracted_files = zip.namelist()

        # Prefix each file path with azureml://
        prefixed_files = list(map(lambda file: "azureml://" + file, extracted_files))
        return prefixed_files


def download_and_extract() -> None:
    """Download Orbio retrievals from Azure Blob Storage to create the Recycled Plumes dataset."""
    # Clear the local_dir if it exists
    if local_dir.exists():
        shutil.rmtree(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    download_from_blob(prefix_on_datastore, local_dir.expanduser())

    zip_files = sorted(local_dir.glob("**/*.zip"))
    results = []
    for zip_file in tqdm(zip_files, desc="Processing zips"):
        result = process_zip_file((zip_file, upload_prefix))
        results.append(result)

    extracted_files = []
    for files in results:
        extracted_files.extend(files)

    # Create and upload catalog.json
    catalog_path = local_dir / "catalog_condensed.json"
    with catalog_path.open("w") as catalog_file:
        json.dump(extracted_files, catalog_file, indent=2)

    # Upload catalog.json to Azure Blob Storage
    # NOTE: this uploads with a recursive structure, i.e.
    # catalog_condensed.json/catalog_condensed.json. generate.py
    # is set to work with the recursive structure
    upload_dir(catalog_path, f"{upload_prefix}/catalog_condensed.json")


if __name__ == "__main__":
    download_and_extract()
