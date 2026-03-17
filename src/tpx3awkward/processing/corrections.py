import numpy as np
import numba
import pandas as pd

def add_centroid_cols(
    df: pd.DataFrame, gap: bool = True
) -> pd.DataFrame:
    """
    Calculates centroid positions to the nearest pixel and the timestamp in nanoseconds.

    Parameters
    ----------
    df : pd.DataFrame
        Input centroided dataframe
    gap : bool = True
        Determines whether to implement large gap correction by adding 2 empty pixels offsets

    Returns
    -------
    pd.DataFrame
        Originally dataframe with new columns x, y, and t_ns added.
    """
    if gap:
        df.loc[df["xc"] >= 255.5, "xc"] += 2
        df.loc[df["yc"] >= 255.5, "yc"] += 2

    df["x"] = np.round(df["xc"]).astype(np.uint16)  # sometimes you just want to know the closest pixel
    df["y"] = np.round(df["yc"]).astype(np.uint16)
    df["t_ns"] = (df["t"].astype(np.float64) * 1.5625)  # better way to convert to ns while maintaining precision?
    if "t_corr" in df:
        df["t_corr_ns"] = (df["t_corr"].astype(np.float64) * 1.5625)

    return df


def trim_corr_file(mask_fpath: str = "/nsls2/users/jgoodrich/proposals/2025-1/qmicroscope/jgoodrich/new_clustering/bool_mask_total.csv"):
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

    if mask_fpath.suffix == '.npy':
        return np.load(mask_fpath).astype(bool)
    elif mask_fpath.suffix == '.csv':
        return np.loadtxt(mask_fpath, delimiter=',').astype(bool)
    else:
        print("Unsupported file format. Use .npy or .csv. Returning None.")
        return None


def trim_corr(df: pd.DataFrame, total_mask: np.ndarray) -> None:
    """
    Modify df in place by subtracting 16 from column 't' where mask is True.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing 'x' and 'y' columns.
    total_mask : np.ndarray
        Boolean mask array indexed by (x, y).
    """
    if total_mask is None:
        print("Trim correction mask is None. No changes applied.")
        return

    # Apply mask condition using direct NumPy indexing
    df_mask = total_mask[df['x'].to_numpy(), df['y'].to_numpy()]

    # Use `.loc[]` with a boolean mask of the same length as df
    df.loc[df_mask, 't'] -= 16
    return df


@numba.njit(cache=True, fastmath=True)
def timewalk_corr_exp(ToT, b=167.0, c=-0.016):
    return np.uint64(np.rint(b * np.exp(c * ToT) / 1.5625))


def timewalk_corr(t, tot, b=167.0, c=-0.016) -> None:
    """Applies timewalk correction in place."""
    return t - timewalk_corr_exp(tot, b, c)


@numba.njit(cache=True)
def tot_to_energy(tot, a, b, c, t):
    # prevent divide by 0
    if a == 0:
        return np.nan
    return ((a * t + tot - b) + np.sqrt(np.power(a, 2) * np.power(t, 2) + 2 * a * b * t + 4 * a * c - 2 * a * t * tot + np.power(b, 2) - 2 * b * tot + np.power(tot, 2))) / (2 * a)


@numba.njit(cache=True)
def estimate_energies(x, y, ToT, energy_calib):
    e = np.empty(len(x), dtype=np.float32)
    for i in range(len(e)):
        a, b, c, t = energy_calib[x[i]][y[i]][:]
        e[i] = tot_to_energy(ToT[i], a, b, c, t)
    return e