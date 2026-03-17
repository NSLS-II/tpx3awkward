import numpy as np
import pandas as pd

def empty_raw_df(include_energy: bool = False) -> pd.DataFrame:
    """
    Create an empty DataFrame with the expected columns from ingest_raw_data()
    and the specified data types.

    Parameters
    ----------
    include_energy : bool, optional
        Whether to include the 'e' column (energy estimates). Default is False.

    Returns
    -------
    pd.DataFrame
        Empty DataFrame with columns:
        ['x', 'y', 'ToT', 't', 'chip', 'cluster_id'] and appropriate dtypes
        or
        ['x', 'y', 'ToT', 'e', 't', 'chip', 'cluster_id'] if include_energy is True.
    """
    data = {
        "x": np.array([], dtype="u2"),         # uint16
        "y": np.array([], dtype="u2"),         # uint16
        "ToT": np.array([], dtype="u4"),       # uint32
        "t": np.array([], dtype="u8"),         # uint64
        "chip": np.array([], dtype="u1"),      # uint8
        "cluster_id": np.array([], dtype="u8")  # uint64
    }

    if include_energy:
        data["e"] = np.array([], dtype="float32")

    return pd.DataFrame(data)


def empty_cent_df(include_energy: bool = False, timewalk_correct: bool = True) -> pd.DataFrame:
    """
    Create an empty DataFrame with the expected columns from ingest_cent_data()
    and the specified data types.

    Parameters
    ----------
    include_energy : bool, optional
        Whether to include the 'e_sum' column (energy estimates). Default is False.

    Returns
    -------
    pd.DataFrame
        Empty DataFrame with columns:
        ['t', 'xc', 'yc', 'ToT_max', 'ToT_sum', 'e_sum', 'n'] and appropriate dtypes
    """
    data = {
        "t": np.array([], dtype="uint64"),       # uint64
        "xc": np.array([], dtype="float32"),     # float32
        "yc": np.array([], dtype="float32"),     # float32
        "ToT_max": np.array([], dtype="uint32"),  # uint32
        "ToT_sum": np.array([], dtype="uint32"),  # uint32
        "n": np.array([], dtype="u1")            # uint8 (ubyte)
    }

    if include_energy:
        data["e_sum"] = np.array([], dtype="float32")
    if timewalk_correct:
        data["t_corr"] = np.array([], dtype="uint64")

    return pd.DataFrame(data)