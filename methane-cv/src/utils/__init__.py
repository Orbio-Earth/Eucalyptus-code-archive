"""Module for general project utilities."""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # orbio/methane-cv/src
PROJECT_ROOT = PACKAGE_ROOT.parent  # orbio/methane-cv
REPO_ROOT = PROJECT_ROOT.parent  # orbio
METHANE_CV_ENV = "methane-cv-env-7-6"

# Since we are copying the orbio repo up to AML and we don't need to copy everything,
# we supply a list of patterns to ignore so we only upload what's needed.
IGNORE_PATTERNS = [
    # gitlab stuff
    ".ci-cd",
    ".gitlab",
    # unecessary projects for methane-cv
    "dags",
    "aws",
    "docs",
    "infrastructure",
    "nomad_jobs",
    "POC",
    "test",
    # unecessary directories
    "*emit",
    "*tests",
    "*notebooks",
    "*sbr_2025",
    "*outputs",
    # unecessary file types
    "*.pt",
    "*.pyc",
    "*.ipynb",
    # misc stuff
    "*.git*",
    "*.benchmarks",
    "*.cache",
    ".cache",
    "*.mypy_cache",
    "*.pytest_cache",
    "*.ruff_cache",
    "*.dockerignore",
    "*.vscode",
    "*venv",
    "*__pycache__",
    "operator_config",
    "qa_tool",
    "cv_data",
    "s2grid.pickle",
    "EMIT",  # EMIT hapi_data in radtran library can be large
    "*.png",
]
