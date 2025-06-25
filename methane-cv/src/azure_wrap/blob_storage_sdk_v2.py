import argparse
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from azureml.fsspec import AzureMachineLearningFileSystem

SUBSCRIPTION_ID = ""
RESOURCE_GROUP = ""
WORKSPACE_NAME = ""
DATASTORE_NAME = ""
DATASTORE_URI = f""


def upload_dir(source: str, target: str, recursive: bool = True) -> None:
    fs = AzureMachineLearningFileSystem(DATASTORE_URI)
    fs.put(source, target, recursive=recursive)


def download_from_blob(source: str, target_local_dir: Path, recursive: bool = True) -> None:
    fs = AzureMachineLearningFileSystem(DATASTORE_URI)
    with TemporaryDirectory() as temp_dir:
        # Download to a temporary directory
        temp_path = Path(temp_dir) / Path(source).name
        fs.get(str(source), str(temp_path), recursive=recursive)

        # Ensure the target directory exists
        target_local_dir.mkdir(parents=True, exist_ok=True)

        # Move or merge contents from the temporary path to the target directory
        if temp_path.is_dir():
            # If the downloaded content is a directory, merge it
            for item in temp_path.iterdir():
                dest_path = target_local_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, dest_path, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest_path)
        else:
            # If it's a single file, move it directly to the target directory
            shutil.move(str(temp_path), str(target_local_dir / temp_path.name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Upload to or download (recursively) from the default blob storage " "in the current Azure ML workspace"
        )
    )

    parser.add_argument("method", help="upload or download", choices=["upload", "download"])
    parser.add_argument("source", help="source directory or blob prefix")
    parser.add_argument("target", help="target blob prefix or directory")

    args = parser.parse_args()
    if args.method == "upload":
        print(f"uploading directory {args.source} to {args.target}" "in blob storage...")
        source = Path(args.source).expanduser()
        upload_dir(str(source), args.target)
    elif args.method == "download":
        print(f"downloading blobs from {args.source} to {args.target}")
        target = Path(args.target).expanduser()
        download_from_blob(args.source, target)
