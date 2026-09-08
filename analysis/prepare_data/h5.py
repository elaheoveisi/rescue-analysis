"""HDF5 and CSV I/O helpers."""

import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import pandas as pd


@contextmanager
def open_store(h5_path: Path) -> Generator[pd.HDFStore, None, None]:
    """Open an HDF5 store for writing, suppressing PyTables warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.HDFStore(str(h5_path), mode="w") as store:
            yield store
