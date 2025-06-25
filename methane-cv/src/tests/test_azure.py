"""Tests for Azure functionality."""

import pandas as pd
import pytest

from src.azure_wrap.azure_path import AzureBlobPath


@pytest.fixture(scope="module")
def parquet_file_uri(azure_data_dir: AzureBlobPath) -> AzureBlobPath:
    """Specific Parquet file URI based on the Azure base test directory."""
    return azure_data_dir / "test.parquet"


def test_uri_to_string() -> None:
    """Test conversion from AzureBlobPath to string."""
    uri = AzureBlobPath("abfs://a/b")
    assert str(uri) == "abfs://a/b"
    uri = AzureBlobPath("/a/b")  # also works
    assert str(uri) == "abfs://a/b"
    uri = AzureBlobPath("abfs://a/b/")  # trailing / should make no difference
    assert str(uri) == "abfs://a/b"


def test_append() -> None:
    """Test appending paths to AzureBlobPath."""
    prefix = AzureBlobPath("abfs://a/b")
    uri = prefix / "file.doc"
    assert str(uri) == "abfs://a/b/file.doc"


def test_round_trip(parquet_file_uri: AzureBlobPath, storage_options: dict) -> None:
    """Test data integrity when saving to Azure."""
    dummy_df = pd.DataFrame({"column_a": [1, 2, 3], "column_b": [42.0, 16.5, 143.8]})
    dummy_df.to_parquet(
        parquet_file_uri,
        storage_options=storage_options,
    )
    df_read = pd.read_parquet(
        parquet_file_uri,
        storage_options=storage_options,
    )
    pd.testing.assert_frame_equal(dummy_df, df_read)
