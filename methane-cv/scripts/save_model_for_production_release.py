"""Use this script within the `methane-cv` conda environment to move a trained model to a production ready datastore.

Example:
python save_model_for_production_release.py --model-name models:/torchgeo_pwr_unet/1475 --satellite-id S2

--> Now follow the remainder of our release process documented in
https://git.orbio.earth/orbio/orbio/-/blob/main/docs/computer-vision/releasing_a_model.md?ref_type=heads
"""

import argparse
import logging
import tempfile
from pathlib import Path

import fsspec
import mlflow
from azure.storage.blob import ContainerClient
from torch import nn

from src.utils.utils import initialize_clients, load_model_and_concatenator

logger = logging.getLogger(__name__)


def _save_model_to_blob_storage(
    model: nn.Module,
    registered_model_name: str,
    storage_account_name: str,
    storage_account_container_name: str,
    container_client: ContainerClient,
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        mlflow.pytorch.save_model(pytorch_model=model, path=tmpdir)

        blob_uri_prefix = (
            f"az://{storage_account_name}.blob.core.windows.net/"
            f"{storage_account_container_name}/"
            f"{registered_model_name}"
        )

        for file in Path(tmpdir).glob("**/*"):
            if file.is_file():
                with fsspec.open(file, "rb") as local_fs:
                    blob_uri = blob_uri_prefix + "/" + file.name
                    print(f"Writing {blob_uri}")
                    container_client.upload_blob(
                        name=f"{registered_model_name}/{file.name}",
                        data=local_fs.read(),
                        validate_content=True,
                        overwrite=False,
                    )

        print(
            "Saved model file and metadata to output azure blob storage. "
            "You may find the file URIs in the azure portal with the prefix "
            f"{blob_uri_prefix}"
        )


def main() -> None:
    """Invoke the script to download the model and save to permanent storage on Azure."""
    parser = argparse.ArgumentParser(
        prog="Save model for production release",
        description="Copies model outputs including metadata to easily retrievable Azure Blob Storage location",
    )

    parser.add_argument(
        "--model-name", "-n", required=True, help="Name of model to release, eg: models:/torchgeo_pwr_unet/548"
    )
    parser.add_argument("--satellite-id", "-s", required=True, help="Name of Satellite ID, eg: S2, LANDAST, EMIT")
    parser.add_argument(
        "--azure-blob-storage-container",
        "-c",
        required=False,
        default="saved-models",
        help="Name of container in the azure blob storage account in which to output saved models.",
    )

    args = parser.parse_args()

    ml_client, _, _, azure_blob_client, _ = initialize_clients(force_msi=False)
    default_blob_storage_account_name = ml_client.datastores.get_default().account_name
    azure_container_client = azure_blob_client.get_container_client(args.azure_blob_storage_container)

    model, _, _ = load_model_and_concatenator(args.model_name, device="cpu", satellite_id=args.satellite_id)

    _save_model_to_blob_storage(
        model=model,
        registered_model_name=args.model_name,
        storage_account_name=default_blob_storage_account_name,
        storage_account_container_name=args.azure_blob_storage_container,
        container_client=azure_container_client,
    )


if __name__ == "__main__":
    main()
