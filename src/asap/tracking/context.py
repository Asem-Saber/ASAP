import hashlib
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from asap.config import Settings, get_settings

logger = logging.getLogger(__name__)

_CHUNK = 1024 * 1024


@dataclass(frozen=True)
class RunContext:
    variant: str
    git_sha: str
    git_dirty: bool
    params: dict[str, Any] = field(default_factory=dict)


def sha256_file(path: Path) -> str:
    """Compute the SHA256 hash of a file.

    Args:
        path (Path): The path to the file to hash.

    Returns:
        str: The SHA256 hash of the file.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def build_run_context(
    variant: str,
    params: dict[str, Any],
    *,
    settings: Settings | None = None,
) -> RunContext:
    cfg = settings or get_settings()
    del cfg

    try:
        git_sha = _git("rev-parse", "HEAD")
        git_dirty = bool(_git("status", "--porcelain"))
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        logger.warning("not a git checkout; recording an empty SHA")
        git_sha, git_dirty = "", False

    return RunContext(
        variant=variant,
        git_sha=git_sha,
        git_dirty=git_dirty,
        params=dict(params),
    )
