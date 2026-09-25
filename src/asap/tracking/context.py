import hashlib
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from asap.config import Settings, get_settings

logger = logging.getLogger(__name__)

_CHUNK = 1024 * 1024
LOCK_FILE = Path("dvc.lock")
PREPARE_OUTPUT = "data/processed"


@dataclass(frozen=True)
class RunContext:
    variant: str
    git_sha: str
    git_dirty: bool
    data_hash: str | None
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


def read_data_hash(lock_path: Path) -> str | None:
    """The md5 DVC recorded for the prepare stage's data/processed output.

    Returns None when DVC is not in use yet, when the stage or output is
    absent, or when the lock file is unreadable. A broken lock file must not
    fail a run that is otherwise fine — provenance is recorded, not required.

    DVC stores one hash for a tracked directory, which is the granularity we
    want: the three splits are only meaningful together.
    """
    if not lock_path.is_file():
        return None

    try:
        lock = yaml.safe_load(lock_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        logger.warning("could not parse %s; recording no data hash", lock_path)
        return None

    if not isinstance(lock, dict):
        return None

    stages = lock.get("stages")
    if not isinstance(stages, dict):
        return None

    outs = (stages.get("prepare") or {}).get("outs") or []
    for out in outs:
        if isinstance(out, dict) and out.get("path") == PREPARE_OUTPUT:
            md5 = out.get("md5")
            return str(md5) if md5 else None
    return None


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
        data_hash=read_data_hash(LOCK_FILE),
        params=dict(params),
    )
