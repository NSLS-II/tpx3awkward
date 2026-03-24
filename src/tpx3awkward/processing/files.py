import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

f_type = SimpleNamespace(HDF=".h5", PARQUET=".parquet")


def trim_corr_file(
    mask_fpath: str = "/nsls2/users/jgoodrich/proposals/2025-1/qmicroscope/jgoodrich/new_clustering/bool_mask_total.csv",
):
    """
    Load a boolean mask from a file, supporting .npy and .csv formats.

    Parameters:
    -----------
    mask_fpath : str, optional
        Path to the mask file. Supports `.npy` (NumPy binary) and `.csv` (comma-separated values).
        Defaults to a predefined file path.

    Returns:
    --------
    np.ndarray or None
        A NumPy boolean array if the file is successfully loaded.
        Returns `None` if the file format is unsupported.

    Notes:
    ------
    - If the file is `.npy`, it is loaded using `np.load()` and converted to `bool`.
    - If the file is `.csv`, it is loaded using `np.loadtxt()` with `delimiter=','` and converted to `bool`.
    - Prints a message and returns `None` for unsupported file formats.
    """
    if mask_fpath is None:
        return None

    mask_fpath = Path(mask_fpath)

    if mask_fpath.suffix == ".npy":
        return np.load(mask_fpath).astype(bool)
    if mask_fpath.suffix == ".csv":
        return np.loadtxt(mask_fpath, delimiter=",").astype(bool)
    print("Unsupported file format. Use .npy or .csv. Returning None.")
    return None


def find_unmatched_tpx3_files(directory_list, reprocess=False):

    # Finds .tpx3 files in the given directories that do not have corresponding _cent{extension} files.
    # Returns a list of Path objects.

    unmatched_files = []
    all_tpx3_files = []

    for tpx3_dir in directory_list:
        # Get all .tpx3 files
        tpx3_files = list(Path(tpx3_dir).glob("*.tpx3"))

        if reprocess:
            all_tpx3_files.extend(tpx3_files)
            continue

        # Generate corresponding _cent.h5 file paths
        h5_cent_files = [converted_path(tpx3_file, cent=True) for tpx3_file in tpx3_files]

        if not h5_cent_files:
            continue

        # Find h5_dir from the first _cent.h5 file
        h5_dir = h5_cent_files[0].parent

        # Get all existing _cent.h5 files in that directory
        existing_h5_files = [p for p in h5_dir.iterdir() if p.suffix in vars(f_type).values()]

        # Check which _cent.h5 files are missing
        unmatched_files.extend(
            tpx3_file
            for tpx3_file, h5_cent_file in zip(tpx3_files, h5_cent_files, strict=True)
            if h5_cent_file not in existing_h5_files
        )

    if reprocess:
        return all_tpx3_files
    return unmatched_files


def converted_path(filepath: str | Path, extension: str = f_type.PARQUET, cent: bool = False):
    """
    Converts .tpx3 file path(s) to corresponding output file path(s).
    Handles individual strings, Path objects, lists, or numpy arrays.

    This is specific to CHX beamline pre and post data security. Is there a better way or place to store this?

    Returns Path objects.
    """
    if isinstance(filepath, (list, np.ndarray)):
        return [converted_path(fp, extension=extension, cent=cent) for fp in filepath]

    filepath = Path(str(filepath).replace("file:", ""))

    if extension not in vars(f_type).values():
        raise TypeError(f"path conversion to unknown file type {extension}")

    if "/nsls2/data/chx/proposals/" in str(filepath):
        out_path = str(filepath).replace("/assets/", "/Compressed_Data/")
    else:
        if "/nsls2/data/chx/legacy/" not in str(filepath):
            warnings.warn(
                "unexpected file path used, operation will proceed but it is suggested to confirm correct target directory",
                stacklevel=2,
            )
        out_path = str(filepath)
    # else:
    #     raise ValueError(f"Unknown path format: {filepath}")

    return Path(out_path.replace(".tpx3", f"{'_cent' if cent else ''}{extension}"))


def save_df(df: pd.DataFrame, fpath: str | Path):
    """
    Save a Pandas DataFrame to a parquet file, ensuring that all necessary directories exist.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be saved.
    fpath : Union[str, Path]
        The full path to the output .parquet file.
    """
    fpath = Path(fpath)  # Ensure fpath is a Path object

    # Create parent directories if they do not exist
    fpath.parent.mkdir(parents=True, exist_ok=True)

    # Save DataFrame
    match fpath.suffix:
        case f_type.HDF:
            df.to_hdf(fpath, key="df", format="table", mode="w")
        case f_type.PARQUET:
            df.to_parquet(
                fpath,
                engine="pyarrow",
                index=False,  # important: do not rely on pandas index
                compression="snappy",
            )
        case _:
            raise TypeError(f"unknown/unimplemented file type: {fpath.suffix}")

    # df.to_hdf(fpath, key="df", format="table", mode="w")
