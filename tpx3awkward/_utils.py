import os
from pathlib import Path
from typing import TypeVar, Union, Dict, List, Tuple
import numpy as np
from numpy.typing import NDArray
import numba
import pandas as pd
import multiprocessing
from functools import partial
import warnings
import glob
from tqdm import tqdm 
import gc


IA = NDArray[np.uint64]
UnSigned = TypeVar("UnSigned", IA, np.uint64)


def raw_as_numpy(fpath: Union[str, Path]) -> IA:
    """
    Read raw tpx3 data file as a numpy array.

    Each entry is read as a uint8 (64bit unsigned-integer)

    Parameters
    ----------

    """
    return np.fromfile(fpath, dtype="<u8")


@numba.jit(nopython=True, cache=True)
def get_block(v: UnSigned, width: int, shift: int) -> UnSigned:
    return v >> np.uint64(shift) & np.uint64(2**width - 1)


@numba.jit(nopython=True, cache=True)
def matches_nibble(data, nibble) -> numba.boolean:
    return (int(data) >> 60) == nibble


@numba.jit(nopython=True, cache=True)
def is_packet_header(v: UnSigned) -> UnSigned:
    """Identify packet headers by magic number (TPX3 as ascii on lowest 8 bytes]"""
    return get_block(v, 32, 0) == 861425748


@numba.jit(nopython=True, cache=True)
def classify_array(data: IA) -> NDArray[np.uint8]:
    """
    Create an array the same size as the data classifying 64bit uint by type.

    0: an unknown type (!!)
    1: packet header (id'd via TPX3 magic number)
    2: photon event (id'd via 0xB upper nibble)
    3: TDC timstamp (id'd via 0x6 upper nibble)
    4: global timestap (id'd via 0x4 upper nibble)
    5: "command" data (id'd via 0x7 upper nibble)
    6: frame driven data (id'd via 0xA upper nibble) (??)
    """
    output = np.zeros_like(data, dtype="<u1")
    is_header = is_packet_header(data)
    output[is_header] = 1
    # get the highest nibble
    nibble = data >> np.uint(60)
    # probably a better way to do this, but brute force!
    output[~is_header & (nibble == 0xB)] = 2
    output[~is_header & (nibble == 0x6)] = 3
    output[~is_header & (nibble == 0x4)] = 4
    output[~is_header & (nibble == 0x7)] = 5
    output[~is_header & (nibble == 0xA)] = 6

    return output


@numba.jit(nopython=True, cache=True)
def _shift_xy(chip, row, col):
    # TODO sort out if this needs to be paremeterized
    out = np.zeros(2, "u4")
    if chip == 0:
        out[0] = row
        out[1] = col + np.uint(256)
    elif chip == 1:
        out[0] = np.uint(511) - row
        out[1] = np.uint(511) - col
    elif chip == 2:
        out[0] = np.uint(511) - row
        out[1] = np.uint(255) - col
    elif chip == 3:
        out[0] = row
        out[1] = col
    else:
        # TODO sort out how to get the chip number in here and make numba happy
        raise RuntimeError("Unknown chip id")
    return out


@numba.jit(nopython=True, cache=True)
def decode_xy(msg, chip):
    # these names and math are adapted from c++ code
    l_pix_addr = (msg >> np.uint(44)) & np.uint(0xFFFF)
    # This is laid out 16ibts which are 2 interleaved 8 bit unsigned ints
    #  CCCCCCCRRRRRRCRR
    #  |dcol ||spix|^||
    #  | 7   || 6  |1|2
    #
    # The high 7 bits of the column
    # '1111111000000000'
    dcol = (l_pix_addr & np.uint(0xFE00)) >> np.uint(8)
    # The high 6 bits of the row
    # '0000000111111000'
    spix = (l_pix_addr & np.uint(0x01F8)) >> np.uint(1)
    rowcol = _shift_xy(
        chip,
        # add the low 2 bits of the row
        # '0000000000000011'
        spix + (l_pix_addr & np.uint(0x3)),
        # add the low 1 bit of the column
        # '0000000000000100'
        dcol + ((l_pix_addr & np.uint(0x4)) >> np.uint(2)),
    )
    return rowcol[1], rowcol[0]


@numba.jit(nopython=True, cache=True)
def get_spidr(msg):
    return msg & np.uint(0xFFFF)


@numba.jit(nopython=True, cache=True)
def decode_message(msg, chip, heartbeat_time: np.uint64 = 0):
    """Decode TPX3 packages of the second type corresponding to photon events (id'd via 0xB upper nibble)

    Parameters
    ----------
        msg (uint64): tpx3 binary message
        chip (uint8): chip ID, 0..3
        heartbeat_time (uint64):

        # bit position   : ...  44   40   36   32   28   24   20   16   12    8 7  4 3  0
        # 0xFFFFC0000000 :    1111 1111 1111 1111 1100 0000 0000 0000 0000 0000 0000 0000
        # 0x3FFFFFFF     :    0000 0000 0000 0000 0011 1111 1111 1111 1111 1111 1111 1111
        # SPIDR          :                                       ssss ssss ssss ssss ssss
        # ToA            :                                                   tt tttt tttt
        # ToA_coarse     :                          ss ssss ssss ssss ssss sstt tttt tttt
        # pixel_bits     :                          ^^
        # FToA           :                                                           ffff
        # count          :                     ss ssss ssss ssss ssss sstt tttt tttt ffff   (FToA is subtracted)
        # phase          :                                                           pppp
        # 0x10000000     :                           1 0000 0000 0000 0000 0000 0000 0000
        # heartbeat_time :    hhhh hhhh hhhh hhhh hhhh hhhh hhhh hhhh hhhh hhhh hhhh hhhh
        # heartbeat_bits :                          ^^
        # global_time    :    hhhh hhhh hhhh hhss ssss ssss ssss ssss sstt tttt tttt ffff

        # count = (ToA_coarse << np.uint(4)) - FToA     # Counter value, in multiples of 1.5625 ns

    Returns
    ----------
        Arrays of pixel coordinates, ToT, and timestamps.
    """
    msg, heartbeat_time = np.uint64(msg), np.uint64(heartbeat_time)    # Force types
    x, y = decode_xy(msg, chip)  # or use x1, y1 = calculateXY(msg, chip) from the Vendor's code
    # ToA is 14 bits
    ToA = (msg >> np.uint(30)) & np.uint(0x3FFF)
    # ToT is 10 bits; report in ns
    ToT = ((msg >> np.uint(20)) & np.uint(0x3FF)) * 25
    # FToA is 4 bits
    FToA = (msg >> np.uint(16)) & np.uint(0xF)
    # SPIDR time is 16 bits
    SPIDR = np.uint64(get_spidr(msg))

    ToA_coarse = (SPIDR << np.uint(14)) | ToA
    # pixel_bits are the two highest bits of the SPIDR (i.e. the pixelbits range covers 262143 spidr cycles)
    pixel_bits = np.int8((ToA_coarse >> np.uint(28)) & np.uint(0x3))
    # heart_bits are the bits at the same positions in the heartbeat_time
    heart_bits = np.int8((heartbeat_time >> np.uint(28)) & np.uint(0x3))
    # Adjust heartbeat_time based on the difference between heart_bits and pixel_bits
    diff = heart_bits - pixel_bits
    # diff +/-1 occur when pixelbits step up
    # diff +/-3 occur when spidr counter overfills
    # diff can also be 0 -- then nothing happens -- but never +/-2
    if (diff == 1 or diff == -3) and (heartbeat_time > np.uint(0x10000000)):
        heartbeat_time -= np.uint(0x10000000)
    elif diff == -1 or diff == 3:
        heartbeat_time += np.uint(0x10000000)
    # Construct globaltime
    global_time = (heartbeat_time & np.uint(0xFFFFFFFC0000000)) | (ToA_coarse & np.uint(0x3FFFFFFF))
    # Phase correction
    phase = np.uint((x / 2) % 16) or np.uint(16)
    # Construct timestamp with phase correction
    ts = (global_time << np.uint(4)) - FToA + phase

    return x, y, ToT, ts


@numba.jit(nopython=True, cache=True)
def _ingest_raw_data(data):

    chips = np.zeros_like(data, dtype=np.uint8)
    x = np.zeros_like(data, dtype="u2")
    y = np.zeros_like(data, dtype="u2")
    tot = np.zeros_like(data, dtype="u4")
    ts = np.zeros_like(data, dtype="u8")
    heartbeat_lsb = None  #np.uint64(0)
    heartbeat_msb = None  #np.uint64(0)
    heartbeat_time = np.uint64(0)
    hb_init_flag = False    # Indicate when the heartbeat was set for the first time

    photon_count, chip_indx, msg_run_count, expected_msg_count = 0, 0, 0, 0

    for msg in data:
        if is_packet_header(msg):
            # Type 1: packet header (id'd via TPX3 magic number)
            #if expected_msg_count != msg_run_count:
                #print("Missing messages!", msg)

            # extract the chip number for the following photon events
            chip_indx = np.uint8(get_block(msg, 8, 32))

            # "number of pixels in chunk" is given in bytes not words and means all words in the chunk, not just "photons"
            expected_msg_count = get_block(msg, 16, 48) // 8
            msg_run_count = 0

        elif matches_nibble(msg, 0xB):
            # Type 2: photon event (id'd via 0xB upper nibble)
            chips[photon_count] = chip_indx
            _x, _y, _tot, _ts = decode_message(msg, chip_indx, heartbeat_time=heartbeat_time)
            x[photon_count] = _x
            y[photon_count] = _y
            tot[photon_count] = _tot
            ts[photon_count] = _ts

            # Adjust timestamps that were set before the first heartbeat was received
            if hb_init_flag and (photon_count > 0):
                prev_ts = ts[:photon_count]   # This portion needs to be adjusted
                # Find what the current timestamp would be without global heartbeat
                _, _, _, _ts_0 = decode_message(msg, chip_indx, heartbeat_time=np.uint64(0))
                # Check if there is a SPIDR rollover in the beginning of the file before the heartbeat
                head_max = max(prev_ts[:10])
                tail_min = min(prev_ts[-10:])
                if (head_max > tail_min) and (head_max - tail_min > 2**32):
                    prev_ts[prev_ts < (tail_min+head_max)/2] += np.uint64(2**34)
                    _ts_0 += np.uint64(2**34)
                ts[:photon_count] = prev_ts + (_ts - _ts_0)
            
            hb_init_flag = False
            photon_count += 1
            msg_run_count += 1

        elif matches_nibble(msg, 0x6):
            # Type 3: TDC timstamp (id'd via 0x6 upper nibble)
            # TODO: handle these!
            msg_run_count += 1
        
        elif matches_nibble(msg, 0x4):
            # Type 4: global timestap (id'd via 0x4 upper nibble)
            subheader = (msg >> np.uint(56)) & np.uint64(0x0F)
            if subheader == 0x4:
                # timer LSB, 32 bits of time -- needs to be received first, before MSB
                heartbeat_lsb = (msg >> np.uint(16)) & np.uint64(0xFFFFFFFF)
            elif subheader == 0x5:
                # timer MSB -- only matters if LSB has been received already
                if heartbeat_lsb is not None:
                    if heartbeat_msb is None:
                        hb_init_flag = True
                    heartbeat_msb = ((msg >> np.uint(16)) & np.uint64(0xFFFF)) << np.uint(32)
                    heartbeat_time = (heartbeat_msb | heartbeat_lsb)
                    # TODO the c++ code has large jump detection, do not understand why
            else:
                raise Exception(f"Unknown subheader {subheader} in the Global Timestamp message")
            pass

            msg_run_count += 1

        elif matches_nibble(msg, 0x7):
            # Type 5: "command" data (id'd via 0x7 upper nibble)
            # TODO handle this!
            msg_run_count += 1

        #else:
            #print(f"Exception 'Not supported: {msg}'")
            #raise Exception(f"Not supported: {msg}")

    # Check if there were no heartbeat messages and adjust for potential SPIDR rollovers
    if heartbeat_msb is None:
        #warnings.warn("No heartbeat messages received; decoded timestamps may be inaccurate.")
        #print("No heartbeat messages received; decoded timestamps may be inaccurate.")
        head_max = max(ts[:10])
        tail_min = min(ts[-10:])
        if (head_max > tail_min) and (head_max - tail_min > 2**32):
            ts[ts < (tail_min+head_max)/2] += np.uint64(2**34)

    # Sort the timestamps
    indx = np.argsort(ts[:photon_count], kind="mergesort") # is mergesort the best here? wondering if this could be optimized
    x, y, tot, ts, chips = x[indx], y[indx], tot[indx], ts[indx], chips[indx]

    return x, y, tot, ts, chips


def ingest_raw_data(data: IA) -> Dict[str, NDArray]:
    """
    Parse values out of raw timepix3 data stream.

    Parameters
    ----------
    data : NDArray[np.unint64]
        The stream of raw data from the timepix3

    Returns
    -------
    Dict[str, NDArray]
       Keys of x, y, ToT, chip_number
    """
    return {
        k.strip(): v
        for k, v in zip("x, y, ToT, t, chip".split(","), _ingest_raw_data(data))
    }


""" 
Some basic functions that help take the initial output of ingest_raw_data and finish the processing.
"""
def tpx_to_raw_df(fpath: Union[str, Path]) -> pd.DataFrame:
    """
    Parses a .tpx3 file and returns the raw data after timesorting.

    Parameters
    ----------
    fpath: Union[str, Path]
        The path to the .tpx3 data to be processed.

    Returns
    -------
    pd.DataFrame
       DataFrame of raw events from the .tpx3 file.
    """
    raw_df = pd.DataFrame(ingest_raw_data(raw_as_numpy(fpath)))
    return raw_df.sort_values("t").reset_index(drop=True) # should we specify the sorting algorithm? at this point? it should be sorted anyway, but I think dataframes need to be explicitly sorted for use in e.g. merge_asof?


def drop_zero_tot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes events which don't have positive ToT. Necessary step before clustering.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to have ToT filtered.

    Returns
    -------
    pd.DataFrame
       df with only the events with ToT > 0
    """
    return df[df["ToT"] > 0]


"""
Functions to help perform clustering and centroiding on raw data.
"""
TIMESTAMP_VALUE = 1.5625*1e-9 # each raw timestamp is 1.5625 seconds
MICROSECOND = 1e-6

# We have had decent success with these values, but do not know for sure if they are optimal.
DEFAULT_CLUSTER_RADIUS = 3
DEFAULT_CLUSTER_TW_MICROSECONDS = 0.3

DEFAULT_CLUSTER_TW = int(DEFAULT_CLUSTER_TW_MICROSECONDS * MICROSECOND / TIMESTAMP_VALUE)


def cluster_df_optimized(df, tw = DEFAULT_CLUSTER_TW, radius = DEFAULT_CLUSTER_RADIUS):
    events = df[["t", "x", "y", "ToT", "t"]].to_numpy()
    events[:, 0] = np.floor_divide(events[:, 0], tw)  # Bin timestamps into time windows

    labels = cluster_df(events, radius, tw)

    return labels, events[:, 1:]


@numba.jit(nopython=True, cache=True)
def cluster_df(events, radius = DEFAULT_CLUSTER_TW, tw = DEFAULT_CLUSTER_RADIUS):
    n = len(events)
    labels = np.full(n, -1, dtype=np.int64)
    cluster_id = 0

    max_time = radius * tw  # maximum time difference allowed for clustering
    radius_sq = radius ** 2  

    for i in range(n):
        if labels[i] == -1:  # if event is unclustered
            labels[i] = cluster_id
            for j in range(i + 1, n):  # scan forward only
                if events[j, 4] - events[i, 4] > max_time:  # early exit based on time
                    break
                # Compute squared Euclidean distance 
                dx = events[i, 0] - events[j, 0]
                dy = events[i, 1] - events[j, 1]
                dt = events[i, 2] - events[j, 2]
                distance_sq = dx * dx + dy * dy + dt * dt

                if distance_sq <= radius_sq: 
                    labels[j] = cluster_id
            cluster_id += 1

    return labels


@numba.jit(nopython=True, cache=True)
def group_indices(labels):
    """
    Group indices by cluster ID using pre-allocated arrays in a Numba-optimized way.

    Parameters
    ----------
    labels : np.ndarray
        Array of cluster labels for each event.
    num_clusters : int
        Number of unique clusters.
    max_cluster_size : int
        Maximum number of events in a single cluster.

    Returns
    -------
    np.ndarray
        A 2D NumPy array of shape (num_clusters, max_cluster_size), where each row corresponds to a cluster 
        and contains event indices padded with -1 for unused slots.
    """
    num_clusters = np.max(labels) + 1  # Assume no noise, all labels are valid clusters
    max_cluster_size = np.bincount(labels).max()
    cluster_array = -1 * np.ones((num_clusters, max_cluster_size), dtype=np.int32)
    cluster_counts = np.zeros(num_clusters, dtype=np.int32)

    for idx in range(labels.shape[0]):
        cluster_idx = labels[idx]  # Label is directly the cluster ID
        cluster_array[cluster_idx, cluster_counts[cluster_idx]] = idx
        cluster_counts[cluster_idx] += 1

    return cluster_array


@numba.jit(nopython=True, cache=True)
def centroid_clusters(
    cluster_arr: np.ndarray, events: np.ndarray
) -> tuple[np.ndarray]:  

    num_clusters = cluster_arr.shape[0]
    max_cluster = cluster_arr.shape[1]
    t = np.zeros(num_clusters, dtype="uint64")
    xc = np.zeros(num_clusters, dtype="float32")
    yc = np.zeros(num_clusters, dtype="float32")
    ToT_max = np.zeros(num_clusters, dtype="uint32")
    ToT_sum = np.zeros(num_clusters, dtype="uint32")
    n = np.zeros(num_clusters, dtype="ubyte")

    for cluster_id in range(num_clusters):
        _ToT_max = np.ushort(0)
        for event_num in range(max_cluster):
            event = cluster_arr[cluster_id, event_num]
            if event > -1:  # if we have an event here
                if events[event, 2] > _ToT_max:  # find the max ToT, assign, use that time
                    _ToT_max = events[event, 2]
                    t[cluster_id] = events[event, 3]
                    ToT_max[cluster_id] = _ToT_max
                xc[cluster_id] += events[event, 0] * events[event, 2]  # x and y centroids by time over threshold
                yc[cluster_id] += events[event, 1] * events[event, 2]
                ToT_sum[cluster_id] += events[event, 2]  # calcuate sum
                n[cluster_id] += np.ubyte(1)  # number of events in cluster
            else:
                break
        xc[cluster_id] /= ToT_sum[cluster_id]  # normalize
        yc[cluster_id] /= ToT_sum[cluster_id]

    return t, xc, yc, ToT_max, ToT_sum, n


def ingest_cent_data(
    data: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Performs the centroiding of a group of clusters.

    Parameters
    ----------
    data : np.ndarray
        The stream of cluster data from cluster_arr_to_cent()

    Returns
    -------
    Dict[str, np.ndarray]
       Keys of t, xc, yc, ToT_max, ToT_sum, and n (number of events) in each cluster.
    """
    return {
        k.strip(): v
        for k, v in zip(
            "t, xc, yc, ToT_max, ToT_sum, n".split(","),
            data,
        )
    }


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
    df["x"] = np.round(df["xc"]).astype(np.uint16) # sometimes you just want to know the closest pixel
    df["y"] = np.round(df["yc"]).astype(np.uint16)
    df["t_ns"] = (df["t"].astype(np.float64) * 1.5625) # better way to convert to ns while maintaining precision?

    return df


def add_centroid_cols_dask(
    df, gap: bool = True
):
    """
    Calculates centroid positions to the nearest pixel and the timestamp in nanoseconds.

    Parameters
    ----------
    df : dd.DataFrame
        Input centroided dataframe
    gap : bool = True
        Determines whether to implement large gap correction by adding 2 empty pixels offsets

    Returns
    -------
    dd.DataFrame
        Originally dataframe with new columns x, y, and t_ns added.
    """
    if gap:
        df['xc'] = df['xc'].mask(cond=df['xc'] >= 255.5, other= df['xc'] + 2)
        df['yc'] = df['yc'].mask(cond=df['yc'] >= 255.5, other= df['yc'] + 2)
    df["x"] = dd.DataFrame.round(df["xc"]).astype(np.uint16)
    df["y"] = dd.DataFrame.round(df["yc"]).astype(np.uint16)
    df["t_ns"] = df["t"] * 1.5625

    #df.compute()
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
def timewalk_corr_exp(ToT, b = 167.0, c = -0.016):
    return np.rint(b * np.exp(c * ToT) / 1.5625).astype(np.uint64)


def timewalk_corr(df: pd.DataFrame, b = 167.0, c = -0.016) -> None:
    """Applies timewalk correction in place."""
    df.loc[:, 't'] -= timewalk_corr_exp(df['ToT'].to_numpy(), b, c)
    return df


"""
Functions to help process multiple related .tpx3 files into Pandas dataframes stored in .h5 files.
"""
def empty_raw_df() -> pd.DataFrame:
    """
    Create an empty DataFrame with the expected columns from ingest_raw_data() 
    and the specified data types.

    Returns
    -------
    pd.DataFrame
        Empty DataFrame with columns:
        ['x', 'y', 'ToT', 't', 'chip', 'cluster_id'] and appropriate dtypes.
    """
    return pd.DataFrame({
        "x": np.array([], dtype="u2"),         # uint16
        "y": np.array([], dtype="u2"),         # uint16
        "ToT": np.array([], dtype="u4"),       # uint32
        "t": np.array([], dtype="u8"),         # uint64
        "chip": np.array([], dtype="u1"),      # uint8
        "cluster_id": np.array([], dtype="u8") # uint64
    })


def empty_cent_df() -> pd.DataFrame:
    """
    Create an empty DataFrame with the expected columns from ingest_cent_data() 
    and the specified data types.

    Returns
    -------
    pd.DataFrame
        Empty DataFrame with columns:
        ['t', 'xc', 'yc', 'ToT_max', 'ToT_sum', 'n'] and appropriate dtypes.
    """
    return pd.DataFrame({
        "t": np.array([], dtype="uint64"),       # uint64
        "xc": np.array([], dtype="float32"),     # float32
        "yc": np.array([], dtype="float32"),     # float32
        "ToT_max": np.array([], dtype="uint32"), # uint32
        "ToT_sum": np.array([], dtype="uint32"), # uint32
        "n": np.array([], dtype="u1")            # uint8 (ubyte)
    })


def find_unmatched_tpx3_files(directory_list, reprocess = False):
    
    #Finds .tpx3 files in the given directories that do not have corresponding _cent.h5 files.
    #Returns a list of Path objects.
    
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
        existing_h5_files = [p for p in h5_dir.glob("*_cent.h5")]
        
        # Check which _cent.h5 files are missingl
        unmatched_files.extend(tpx3_file for tpx3_file, h5_cent_file in zip(tpx3_files, h5_cent_files) if h5_cent_file not in existing_h5_files)

    if reprocess:
        return all_tpx3_files
    else:    
        return unmatched_files


def converted_path(filepath, cent=False):
    """
    Converts .tpx3 file path(s) to corresponding .h5 file path(s).
    Handles individual strings, Path objects, lists, or numpy arrays.

    This is specific to CHX beamline pre and post data security. Is there a better way or place to store this?
    
    Returns Path objects.
    """
    if isinstance(filepath, (list, np.ndarray)):
        return [converted_path(fp, cent) for fp in filepath]
    
    filepath = Path(str(filepath).replace("file:", ""))
    
    if "/nsls2/data/chx/proposals/" in str(filepath):
        h5_path = Path(str(filepath).replace("/assets/", "/Compressed_Data/").replace(".tpx3", "_cent.h5" if cent else ".h5"))
    elif "/nsls2/data/chx/legacy/" in str(filepath):
        h5_path = Path(str(filepath).replace(".tpx3", "_cent.h5" if cent else ".h5"))
    else:
        raise ValueError(f"Unknown path format: {filepath}")
    
    return h5_path


def save_df(df: pd.DataFrame, fpath: Union[str, Path]):
    """
    Save a Pandas DataFrame to an HDF5 file, ensuring that all necessary directories exist.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be saved.
    fpath : Union[str, Path]
        The full path to the output HDF5 file.
    """
    fpath = Path(fpath)  # Ensure fpath is a Path object

    # Create parent directories if they do not exist
    fpath.parent.mkdir(parents=True, exist_ok=True)

    # Save DataFrame
    df.to_hdf(fpath, key="df", format="table", mode="w")


def convert_tpx_file(
    tpx3_fpath: Union[str, Path], tw: float = DEFAULT_CLUSTER_TW, radius: int = DEFAULT_CLUSTER_RADIUS, trim_correct: bool = None, timewalk_correct: bool = False, print_details: bool = False, overwrite: bool = True
):
    """
    Convert a .tpx3 file into raw and centroided Pandas dataframes, which are stored in .h5 files.
    
    TO DO: Args to specify output directory (default will be same directory as .tpx3 file as is now).
    
    Parameters
    ----------
    tpx3_fpath : Union[str, Path]
        .tpx3 file path
    tw : float = DEFAULT_CLUSTER_TW_MICROSECONDS
        The time window, in Timepix timestamp units, to perform centroiding
    radius : int = DEFAULT_CLUSTER_RADIUS
        The radius, in pixels, to perform centroiding
    trim_correct : bool = False
        Whether to apply trim correction
    timewalk_correct : bool = False
        Whether to apply timewalk correction
    print_details : bool = False
        Boolean toggle about whether to print detailed data.
    overwrite : bool = True
        Boolean toggle about whether to overwrite pre-existing data.
        
    """
    if isinstance(tpx3_fpath, str):
        tpx3_fpath = Path(tpx3_fpath)

    if tpx3_fpath.exists():
        if tpx3_fpath.suffix == ".tpx3":

            h5_fpath = converted_path(tpx3_fpath, cent=False)
            cent_h5_fpath = converted_path(tpx3_fpath, cent=True)
        
            try: 
                
                tpx3_fpath_size = tpx3_fpath.stat().st_size  # Get file size
                have_df = h5_fpath.exists()       # Check if dfname exists
                have_dfc = cent_h5_fpath.exists()  # Check if dfcname exists
    
                if have_df and have_dfc and not overwrite:
                    
                    print("-> {} already processed, skipping.".format(tpx3_fpath.name))
                    return False
                    
                else:

                    if print_details:
                        print("-> Processing {}, size: {:.1f} MB".format(tpx3_fpath.name, tpx3_fpath_size/(1024*1024)))

                    if tpx3_fpath_size == 0:
                        num_events = 0
                    else:
                        df = drop_zero_tot(tpx_to_raw_df(tpx3_fpath))
                        num_events = df.shape[0]

                    if num_events > 0:
                    
                        if print_details:
                            print("Loading {} complete. {} events found.".format(tpx3_fpath.name, num_events))
        
                        if trim_correct is not None:
                            if print_details:
                                print("Performing trim correction on {}".format(tpx3_fpath.name))
                            df = trim_corr(df, trim_correct)               
                            
                        if timewalk_correct:
                            if print_details:
                                print("Performing timewalk correction on {}".format(tpx3_fpath.name))
                            df = timewalk_corr(df)
                    
                        cluster_labels, events = cluster_df_optimized(df, tw, radius)
                        df['cluster_id'] = cluster_labels
                        if print_details:
                            print("Clustering {} complete. {} clusters found. Saving {}...".format(tpx3_fpath.name, cluster_labels.max()+1, h5_fpath.name))
    
                        save_df(df, h5_fpath)
                        if print_details:
                            print("Saving {} complete. Centroiding...".format(h5_fpath.name))
                        
                        cluster_array = group_indices(cluster_labels)
                        data = centroid_clusters(cluster_array, events)
                        
                        cdf = pd.DataFrame(ingest_cent_data(data)).sort_values("t").reset_index(drop=True)
                        if print_details:
                            print("Centroiding complete. Saving to {}...".format(cent_h5_fpath.name))
                            # save cdf
    
                        save_df(cdf, cent_h5_fpath)                  
                        if print_details:
                            print("Saving {} complete. Checking file existence...".format(cent_h5_fpath.name))
                            
                        if cent_h5_fpath.exists():
                            if print_details:
                                print("Confirmed {} exists!".format(cent_h5_fpath.name))
                            to_return = True
                        else:
                            if print_details:
                                print("WARNING: {} doesn't exist but it should?!".format(cent_h5_fpath.name))
                            to_return = False

                        if print_details:
                            print("Moving onto next file...")
    
                        df, cdf, cluster_labels, events, cluster_array, data = None, None, None, None, None, None   
                        gc.collect()
                        return to_return

                    else: 

                        if print_details:
                            print("No events found! Saving empty dataframes.")
                        save_df(empty_raw_df(), h5_fpath) 
                        save_df(empty_cent_df(), cent_h5_fpath) 

                        gc.collect()

                    return True
                    
            except Exception as e:
                    
                if print_details:
                    print(f"Conversion of {tpx3_fpath.name} failed due to {e.__class__.__name__}: {e}, moving on.")
                    return False
                    
        else:
            if print_details:
                print("File was not a .tpx3 file. Moving onto next file.")
                return False

    else:
        if print_details:
            print("File does not exist. Moving onto next file.")
            return False


def convert_tpx3_files_parallel(
    fpaths: Union[List[str], List[Path]], num_workers: int = None, trim_correct: Union[str, Path] = None, **kwargs
):
    """
    Convert a list of .tpx3 files in parallel using multiprocessing and convert_tpx_file().
    
    Parameters
    ----------
    fpaths : Union[List[str], List[Path]]
        List of .tpx3 file paths to process.
    num_workers : int, optional
        Number of worker processes to use. Defaults to (CPU count - 4) to leave room for other tasks.
    trim_mask_fpath : str, optional
        Path to the trim correction mask. If None, no correction is applied.
    **kwargs : dict
        Additional keyword arguments passed to `convert_tpx_file()`.
    """
    if len(fpaths) > 0:
        if num_workers is None:
            max_workers = min(multiprocessing.cpu_count() - 4, len(fpaths))  # Leave 4 cores free
        else:
            max_workers = min(num_workers, len(fpaths))  # Don't use more workers than files
    
        # Load the mask once
        trim_mask = trim_corr_file(trim_correct)
    
        # Pass the preloaded mask to all workers
        worker_func = partial(convert_tpx_file, trim_correct=trim_mask, **kwargs)
    
        with multiprocessing.Pool(processes=max_workers) as pool:
            results = list(tqdm(pool.imap_unordered(worker_func, fpaths), total=len(fpaths), desc="Processing files"))
    
        # Count successes
        num_true = sum(results)
    else:
        num_true = 0
        
    print(f"Successfully converted {num_true} out of {len(fpaths)}!")


def convert_tpx3_files(
    fpaths: Union[List[str], List[Path]], trim_correct: Union[str, Path] = None, print_details: bool = True, **kwargs
):
    """
    Convert a list of .tpx3 files in a single process using convert_tpx_file().
    
    Parameters
    ----------
    fpaths : Union[List[str], List[Path]]
        List of .tpx3 file paths to process.
    trim_mask_fpath : str, optional
        Path to the trim correction mask. If None, no correction is applied.
    print_details : bool, optional
        Boolean toggle about whether to print detailed data. Default is True.
    **kwargs : dict
        Additional keyword arguments passed to `convert_tpx_file()`.
    """
    # Load the mask once (only if provided)
    trim_mask = trim_corr_file(trim_correct)

    # Process files sequentially with tqdm progress bar
    for file in tqdm(fpaths, desc="Processing files"):
        convert_tpx_file(file, trim_correct=trim_mask, print_details=print_details, **kwargs)

        
