# mypy: disable-error-code="no-untyped-def"
from pathlib import PurePosixPath


class AzureBlobPath(PurePosixPath):
    """A class that behaves like a pathlib Path
    for azure blob storage URIs
    """

    azure_schema = "abfs://"

    def __new__(cls, *args, **kwargs):
        # Override __new__ to handle the Azure URI scheme
        uri = args[0]
        if uri.startswith(cls.azure_schema):
            # Remove the custom scheme before passing to the parent class
            posix_path = uri.replace(cls.azure_schema, "/", 1)
            args = (posix_path,) + args[1:]
        return super().__new__(cls, *args, **kwargs)

    def __str__(self):
        # pathlib paths have a method to turn them
        # into a URI starting with file://
        # which is very convenient for us,
        # so we'll use it as a starting point
        posix_path = PurePosixPath(self)
        file_uri = posix_path.as_uri()
        # now we just replace 'file://' with 'abfs://'
        azure_uri = file_uri.replace("file:///", self.azure_schema)
        return azure_uri


class AzureMLBlobPath(AzureBlobPath):
    azure_schema = "azureml://"
