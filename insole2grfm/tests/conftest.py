import sys
from pathlib import Path

import pytest

# The training scripts are flat modules at the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synthetic_prepared import SyntheticPrepared, run_training_cli, write_synthetic_prepared  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_prepared(tmp_path_factory) -> SyntheticPrepared:
    """Read-only synthetic prepared dataset shared by the whole session."""
    return write_synthetic_prepared(tmp_path_factory.mktemp("synthetic_prepared"))


@pytest.fixture(scope="session")
def trained_once(tmp_path_factory):
    """``trained_once(data_dir, *args) -> output dir``: one engine CLI run per distinct argument list.

    Tests that need the same run (same data, same flags) share it; the output
    directory is read-only for them.
    """
    runs: dict[tuple, Path] = {}

    def train(data_dir: Path, *args: str) -> Path:
        key = (str(data_dir), *args)
        if key not in runs:
            out = tmp_path_factory.mktemp("cli_run") / "run"
            run_training_cli(data_dir, out, *args)
            runs[key] = out
        return runs[key]

    return train
