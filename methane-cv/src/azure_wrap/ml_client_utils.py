"""Interact with Azure ML filesystem."""

import re
from pathlib import Path, PurePosixPath

from azure.ai.ml import MLClient
from azure.ai.ml.entities import AzureBlobDatastore, Compute
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
from azure.storage.blob import BlobServiceClient
from azureml.fsspec import AzureMachineLearningFileSystem

from src.azure_wrap.azure_path import AzureBlobPath, AzureMLBlobPath


def initialize_ml_client(force_msi: bool = False) -> MLClient:
    """Initialize and return the MLClient."""
    if force_msi:
        # FIXME: client_id is found on the azure console by looking up the `cv-job-runner` Managed Identity
        # This should really not be hardcoded and will have to be replaced upon recreation of the MI.
        return MLClient.from_config(ManagedIdentityCredential(client_id=""))

    return MLClient.from_config(DefaultAzureCredential())


def ensure_compute_target_exists(ml_client: MLClient, compute_target: str) -> Compute:
    """Ensure the compute target exists."""
    return ml_client.compute.get(compute_target)


def get_default_blob_storage(ml_client: MLClient) -> AzureBlobDatastore:
    """Retrieve the default blob storage for the given MLClient."""
    return ml_client.datastores.get_default(include_secrets=True)


def get_azureml_uri(ml_client: MLClient, data_prefix: str) -> PurePosixPath:
    """
    Get the full azureml:// URI for the files contained in the `data_prefix` directory.

    This is specifically for the default AzureML datastore for our workspace.
    """
    def_blob_storage = get_default_blob_storage(ml_client)
    path_root = AzureMLBlobPath(def_blob_storage.id)
    folder_uri = path_root / "paths" / data_prefix
    return folder_uri


def make_acceptable_uri(uri: str) -> str:
    r"""
    Modify URI to fit the required pattern.

    There is a bug in AzureML for parsing folder URIs. It expects a URI matching the regular expression
    azureml://subscriptions/[a-zA-Z0-9\-_]+/resourcegroups/[a-zA-Z0-9._\-()]+/workspaces/[a-zA-Z0-9\-_]+/datastores/[a-zA-Z0-9_]+/paths/.*
    but the URI returned by their own SDK does not fit that pattern (thanks guys!).
    So we need to modify it a bit so it fits their regular expression
    """
    acceptable_uri = re.sub("resourceGroups", "resourcegroups", uri)
    acceptable_uri = re.sub(r"providers/[a-zA-Z0-9.\-_]+/", "", acceptable_uri)
    return acceptable_uri


def get_storage_options(ml_client: MLClient) -> dict:
    """
    Retrieve and return storage options for the default AzureML datastore.

    These are needed when uploading/writing files directly to Azure Blob Storage.
    """
    def_blob_storage = get_default_blob_storage(ml_client)
    storage_options = {
        "account_name": def_blob_storage.account_name,
        # "account_key": def_blob_storage.credentials.account_key,
        "sas_token": def_blob_storage.credentials.sas_token,
    }
    return storage_options


def get_azure_ml_file_system(ml_client: MLClient) -> AzureMachineLearningFileSystem:
    """
    Return an initialized Azure Machine Learning File System instance.

    This function initializes an MLClient from the default configuration and
    retrieves the file system associated with the default blob storage for use
    in data operations within the Azure ML workspace.
    """
    def_blob_storage = get_default_blob_storage(ml_client)
    if def_blob_storage.id is None:
        raise ValueError("The default blob storage does not have a valid ID.")
    return AzureMachineLearningFileSystem("azureml:/" + def_blob_storage.id)


def get_abfs_output_directory(ml_client: MLClient, out_dir: Path) -> AzureBlobPath:
    """
    Return the Azure Blob File System (ABFS) directory path for output.

    This function retrieves the default blob storage information from the Azure ML workspace,
    and constructs the ABFS path based on the provided output directory.
    """
    def_blob_storage = get_default_blob_storage(ml_client)
    blob_container = def_blob_storage.container_name
    return AzureBlobPath(f"abfs://{blob_container}/{out_dir}")


# TODO: Figure out how to just use one download method between blob_storage_sdk and this one
def download_blob_directly(
    blob_name: str, local_download_filepath: Path, blob_service_client: BlobServiceClient, container_name: str
) -> None:
    """Download a blob directly to a specified local path."""
    if not local_download_filepath.parent.exists():
        local_download_filepath.parent.mkdir(parents=True, exist_ok=True)

    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    try:
        with open(local_download_filepath, "wb") as download_file:
            download_stream = blob_client.download_blob()
            download_file.write(download_stream.readall())
    except ResourceNotFoundError as e:
        print(f"Error downloading blob [{blob_name}]")
        raise e


def initialize_blob_service_client(ml_client: MLClient) -> BlobServiceClient:
    """Initialise and return a BlobServiceClient."""
    storage_options = get_storage_options(ml_client)
    connection_string = f"BlobEndpoint=https://{storage_options['account_name']}.blob.core.windows.net;SharedAccessSignature={storage_options['sas_token']}"
    return BlobServiceClient.from_connection_string(connection_string)


def create_ml_client_config() -> None:
    """
    Create and write Azure ML client configuration.

    Necessary when interacting with Azure file systems on a compute instance.
    """
    with open("/config.json", "w") as file:
        file.write(
            """
            {
                "subscription_id": "6e71ce37-b9fe-4c43-942b-cf0f7e78c8ab",
                "resource_group": "orbio-ml-rg",
                "workspace_name": "orbio-ml-ml-workspace"
            }            
            """
        )
