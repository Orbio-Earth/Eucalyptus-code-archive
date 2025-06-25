"""Silence useless warnings triggered by poorly maintained Azure packages."""

import warnings

from marshmallow.warnings import ChangedInMarshmallow4Warning, RemovedInMarshmallow4Warning
from pydantic.warnings import PydanticDeprecatedSince20

warnings.filterwarnings("ignore", category=ChangedInMarshmallow4Warning)
warnings.filterwarnings("ignore", category=RemovedInMarshmallow4Warning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="azureml.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=PydanticDeprecatedSince20)
    import mlflow  # noqa: F401 (imported but unused)
    import planetary_computer  # noqa: F401 (imported but unused)
