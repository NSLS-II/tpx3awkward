from pathlib import Path
import pandas as pd
import numpy as np

def find_unmatched_tpx3_files(directory_list, reprocess=False):

    # Finds .tpx3 files in the given directories that do not have corresponding _cent{extension} files.
    # Returns a list of Path objects.

    unmatched_files = []
    all_tpx3_files = []

    for tpx3_dir in directory_list:
        # Get all .tpx3 files
        tpx3_files = list(Path(tpx3_dir).glob("*.tpx3"))

        if reprocess == True:
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
        unmatched_files.extend(tpx3_file for tpx3_file, h5_cent_file in zip(
            tpx3_files, h5_cent_files) if h5_cent_file not in existing_h5_files)

    if reprocess:
        return all_tpx3_files
    else:
        return unmatched_files


def converted_path(filepath: Union[str, Path] , extension: str = f_type.PARQUET, cent: bool = False):
    """
    Converts .tpx3 file path(s) to corresponding output file path(s).
    Handles individual strings, Path objects, lists, or numpy arrays.

    This is specific to CHX beamline pre and post data security. Is there a better way or place to store this?

    Returns Path objects.
    """
    if isinstance(filepath, (list, np.ndarray)):
        return [converted_path(fp, extension = extension, cent = cent) for fp in filepath]

    filepath = Path(str(filepath).replace("file:", ""))

    if extension not in vars(f_type).values():
        raise TypeError(f"path conversion to unknown file type {extension}")

    if "/nsls2/data/chx/proposals/" in str(filepath):
        out_path = str(filepath).replace("/assets/", "/Compressed_Data/")
    else:
        if not ("/nsls2/data/chx/legacy/" in str(filepath)):
            warnings.warn(
                "unexpected file path used, operation will proceed but it is suggested to confirm correct target directory")
        out_path = str(filepath)
    # else:
    #     raise ValueError(f"Unknown path format: {filepath}")

    return Path(out_path.replace(".tpx3", f"{"_cent" if cent else ""}{extension}"))


def save_df(df: pd.DataFrame, fpath: Union[str, Path]):
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
                index=False,   # important: do not rely on pandas index
                compression="snappy",
            )
        case _:
            raise TypeError(f"unknown/unimplemented file type: {fpath.suffix}")

    #df.to_hdf(fpath, key="df", format="table", mode="w")