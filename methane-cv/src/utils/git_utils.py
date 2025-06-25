"""Utility functions for interacting with git."""

from pathlib import Path

import git

GIT_REPO = git.Repo(__file__, search_parent_directories=True)
GIT_REPO_ROOT = Path(GIT_REPO.git.rev_parse("--show-toplevel"))


def get_git_revision_hash() -> str:
    """Get the full git revision hash of the current repository."""
    return GIT_REPO.head.object.hexsha


def get_git_revision_short_hash() -> str:
    """Get the short git revision hash of the current repository."""
    return GIT_REPO.head.object.hexsha[:7]
