import os
import numpy as np
from pathlib import Path
from numpy.typing import NDArray
from typing import TypeVar, Union, Dict, Set, List, Iterable
import numba
import pandas as pd
from scipy.spatial import KDTree
import multiprocessing
from tqdm import tqdm
import warnings
import gc
try:
    from pyCHX.chx_packages import db, get_sid_filenames
except ImportError:
    warnings.warn("Could not import pyCHX package. Proceeding without it...")

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


@numba.jit(nopython=True)
def get_block(v: UnSigned, width: int, shift: int) -> UnSigned:

    return v >> np.uint64(shift) & np.uint64(2**width - 1)


@numba.jit(nopython=True)
def matches_nibble(data, nibble) -> numba.boolean:
    return (int(data) >> 60) == nibble


@numba.jit(nopython=True)
def is_packet_header(v: UnSigned) -> UnSigned:
    """Identify packet headers by magic number (TPX3 as ascii on lowest 8 bytes]"""
    return get_block(v, 32, 0) == 861425748


@numba.jit(nopython=True)
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


@numba.jit(nopython=True)
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


@numba.jit(nopython=True)
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


@numba.jit(nopython=True)
def get_spidr(msg):
    return msg & np.uint(0xFFFF)


@numba.jit(nopython=True)
def decode_message(msg, chip, last_ts: np.uint64 = 0):
    """Decode TPX3 packages of the second type corresponding to photon events (id'd via 0xB upper nibble)

    Parameters
    ----------
        msg (uint64): tpx3 binary message
        chip (uint8): chip ID, 0..3
        last_ts (uint64): last timestamp from the previous message; defines the SPIDR rollover

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
    pixel_bits = int((ToA_coarse >> np.uint(28)) & np.uint(0x3))
    # heart_bits are the bits at the same positions in the heartbeat_time
    heartbeat_time = np.uint64(last_ts)
    heart_bits = int((heartbeat_time >> np.uint(28)) & np.uint(0x3))
    # Adjust heartbeat_time based on the difference between heart_bits and pixel_bits
    diff = heart_bits - pixel_bits
    # diff +/-1 occur when pixelbits step up
    # diff +/-3 occur when spidr counter overfills
    # diff can also be 0 -- then nothing happens -- but never +/-2
    if diff == 1 or diff == -3:
        heartbeat_time -= np.uint(0x10000000)
    elif diff == -1 or diff == 3:
        heartbeat_time += np.uint(0x10000000)
    # Construct globaltime
    global_time = (heartbeat_time & np.uint(0xFFFFC0000000)) | (ToA_coarse & np.uint(0x3FFFFFFF))
    # Phase correction
    phase = np.uint((x / 2) % 16) or np.uint(16)
    # Construct timestamp with phase correction
    ts = (global_time << np.uint(4)) - FToA + phase

    return x, y, ToT, ts


@numba.jit(nopython=True)
def decode_heartbeat(msg, heartbeat_lsb=np.uint64(0)):
    """Decode a heartbeat time message (idetified by its 0x4 upper nibble)

    Parameters
    ----------
        msg (uint64): tpx3 binary message with upper_nibble == 0x4

    Returns
    ----------
        Integer value of the heartbeat time.
    """
    subheader = (msg >> np.uint(56)) & np.uint(0x0F)
    if subheader == 0x4:
        # timer lsb, 32 bits of time
        heartbeat_lsb = (msg >> np.uint(16)) & np.uint(0xFFFFFFFF)
    elif subheader == 0x5:
        # timer msb
        time_msg = (msg >> np.uint(16)) & np.uint(0xFFFF)
        heartbeat_msb = time_msg << np.uint(32)
        # TODO the c++ code has large jump detection, do not understand why
        heartbeat_time = heartbeat_msb | heartbeat_lsb
    else:
        raise Exception(f"Unknown subheader {subheader} in the Global Timestamp message")
    
    return heartbeat_lsb, heartbeat_time


@numba.jit(nopython=True)
def _ingest_raw_data(data: IA, last_ts: np.uint64 = 0):

    chips = np.zeros_like(data, dtype=np.uint8)
    x = np.zeros_like(data, dtype="u2")
    y = np.zeros_like(data, dtype="u2")
    tot = np.zeros_like(data, dtype="u4")
    ts = np.zeros_like(data, dtype="u8")

    photon_count, chip_indx, msg_run_count = 0, 0, 0
    for msg in data:
        if is_packet_header(msg):
            # Type 1: packet header (id'd via TPX3 magic number)
            if expected_msg_count != msg_run_count:
                print("Missing messages!", msg)

            # extract the chip number for the following photon events
            chip_indx = np.uint8(get_block(msg, 8, 32))

            # "number of pixels in chunk" is given in bytes not words and means all words in the chunk, not just "photons"
            expected_msg_count = get_block(msg, 16, 48) // 8
            msg_run_count = 0

        elif matches_nibble(msg, 0xB):
            # Type 2: photon event (id'd via 0xB upper nibble)
            chips[photon_count] = chip_indx
            _x, _y, _tot, _ts = decode_message(msg, chip_indx, last_ts = last_ts)
            x[photon_count] = _x
            y[photon_count] = _y
            tot[photon_count] = _tot
            ts[photon_count] = last_ts = _ts
            photon_count += 1
            msg_run_count += 1

        elif matches_nibble(msg, 0x6):
            # Type 3: TDC timstamp (id'd via 0x6 upper nibble)
            # TODO: handle these!
            msg_run_count += 1
        
        elif matches_nibble(msg, 0x4):
            # Type 4: global timestap (id'd via 0x4 upper nibble)
            heartbeat_lsb, heartbeat_time = decode_heartbeat(msg, heartbeat_lsb)
            msg_run_count += 1

        elif matches_nibble(msg, 0x7):
            # Type 5: "command" data (id'd via 0x7 upper nibble)
            # TODO handle this!
            msg_run_count += 1

        else:
            raise Exception(f"Not supported: {msg}")


    # Sort the timestamps
    indx = np.argsort(ts[:photon_count], kind="mergesort")
    chips = chips[indx]
    x, y, tot, ts = x[indx], y[indx], tot[indx], ts[indx]

    return x, y, tot, ts, chips


def ingest_from_files(fpaths: List[Union[str, Path]]) -> Iterable[Dict[str, NDArray]]:
    """Parse values out of a sequence of timepix3 files with rollover of SPIDR counter.

    Parameters
    ----------
    fpaths : A sorted sequence of tpx3 filepaths.

    Returns
    -------
    An iterable over the parsing results, each encapsulated in a dictionary
    """

    last_ts = 0
    for fpath in fpaths:
        data = raw_as_numpy(fpath)
        x, y, tot, ts, chips = _ingest_raw_data(data, last_ts = last_ts)
        last_ts = ts[-1]
        yield {
            k.strip(): v
            for k, v in zip(
                "x, y, tot, timestamp, chips".split(","),
                (x, y, tot, ts, chips),
            )
        }


def ingest_raw_data(data: IA, last_ts: np.uint64 = 0) -> Dict[str, NDArray]:
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
        for k, v in zip("x, y, ToT, ts, chip_number".split(","), _ingest_raw_data(data, last_ts=last_ts))
    }

# ^-- tom wrote
# v-- justin wrote
""" 
Some basic functions that help take the initial output of ingest_raw_data and finish the processing.
"""


def raw_to_sorted_df(fpath: Union[str, Path]) -> pd.DataFrame:
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
    return raw_df.sort_values("timestamp").reset_index(drop=True)


def condense_raw_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Condenses the raw dataframe with only key information necesary for the analysis. Returns a dataframe with timestamp (renamed to t), x, y, and ToT.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame generated using raw_to_sorted_df().

    Returns
    -------
    pd.DataFrame
        Dataframe condensed to only contain pertinent information for analysis.
    """
    cdf = df[["timestamp", "x", "y", "ToT"]]
    cdf = cdf.rename(
        columns={"timestamp": "t"}
    )  # obviously not necessary, just easier to type 't' a lot than 'timestamp'
    return cdf


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
    fdf = df[df["ToT"] > 0]
    return fdf


"""
Functions to help perform clustering and centroiding on raw data.
"""
TIMESTAMP_VALUE = ((1e-9) / 4096) * 25
MICROSECOND = 10 ** (-6)

# We have had decent success with these values, but do not know for sure if they are optimal.
DEFAULT_CLUSTER_RADIUS = 2
DEFAULT_CLUSTER_TW_MICROSECONDS = 0.5

DEFAULT_CLUSTER_TW = int(DEFAULT_CLUSTER_TW_MICROSECONDS * MICROSECOND / TIMESTAMP_VALUE)


def neighbor_set_from_df(
    df: pd.DataFrame, tw: int = DEFAULT_CLUSTER_TW, radius: int = DEFAULT_CLUSTER_RADIUS
) -> tuple[np.ndarray, Set[tuple[int]]]:
    """
    Uses scipy.spatial's KDTree to cluster raw input data. Requires a time window for clustering adjacent pixels and the total search radius.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the raw data (after timesorting and ToT filtering).
    tw : int
        The time window to be considered "coincident" for clustering purposes
    radius : int
        The search radius, using Euclidean distance of x, y, timestamp/tw

    Returns
    -------
    np.ndarray
        Numpy representation of the raw events being used in the clustering.
    Set[tuple[int]]
        An set of tuples of the indices of the clustered events.  The outer set is each cluster, and the inner tuples are the events in each cluster.
    """
    events = np.array(
        df[["t", "x", "y", "ToT", "t"]].values
    )  # first three columns are for search radius of KDTree
    events[:, 0] = np.floor_divide(events[:, 0], tw)  # bin by the time window
    tree = KDTree(events[:, :3])  # generate KDTree based off the coordinates
    neighbors = tree.query_ball_tree(
        tree, radius
    )  # compare tree against itself to find neighbors within the search radius
    clusters = set(tuple(n) for n in neighbors)  # turn the list of lists into a set of tuples
    return events, clusters


def cluster_stats(
    clusters: Set[tuple[int]]
) -> tuple[int]:
    """
    Determines basic information about cluster information, such as the number of clusters and size of the largest cluster.

    Parameters
    ----------
    clusters : Set[tuple[int]]
        The set of tuples of clusters from neighbor_set_from_df()

    Returns
    -------
    int
        The total number of clusters
    int
        The number of events in the largest cluster
    """
    num_clusters = len(clusters)
    max_cluster = max(map(len, clusters))
    return num_clusters, max_cluster


def create_cluster_arr(
    clusters: Set[tuple[int]], num_clusters: int, max_cluster: int
) -> np.ndarray:  # is there a better way to do this?
    """
    Converts the clusters from a set of tuples of indices to an 2D numpy array format which can be efficiently iterated through with Numba.

    Parameters
    ----------
    clusters : Set[tuple[int]]
        The set of tuples of clusters from neighbor_set_from_df()
    num_clusters : int
        The total number of clusters
    max_cluster : int
        The number of events in the largest cluster

    Returns
    -------
    np.ndarray
        The cluster data now in a 2D numpy array.
    """
    cluster_arr = np.full(
        (num_clusters, max_cluster), -1, dtype=np.int64
    )  # fill with -1; these will be passed later
    for cluster_num, cluster in enumerate(clusters):
        for event_num, event in enumerate(cluster):
            cluster_arr[cluster_num, event_num] = event
    return cluster_arr


@numba.jit(nopython=True)
def cluster_arr_to_cent(
    cluster_arr: np.ndarray, events: np.ndarray, num_clusters: int, max_cluster: int
) -> tuple[np.ndarray]:
    """
    Performs the centroiding of a group of clusters using Numba.  Note I originally attempted to unpack the clusters using list comprehensions, but this approach is significantly faster.

    Parameters
    ----------
    clusters : Set[tuple[int]]
        The set of tuples of clusters from neighbor_set_from_df()
    num_clusters : int
        The total number of clusters
    max_cluster : int
        The number of events in the largest clust

    Returns
    -------
    tuple[np.ndarray]
        t, xc, yc, ToT_max, ToT_sum, and n (number of events) in each cluster.
    """
    t = np.zeros(num_clusters, dtype="uint64")
    xc = np.zeros(num_clusters, dtype="float32")
    yc = np.zeros(num_clusters, dtype="float32")
    ToT_max = np.zeros(num_clusters, dtype="uint16")
    ToT_sum = np.zeros(num_clusters, dtype="uint16")
    n = np.zeros(num_clusters, dtype="ubyte")

    for cluster_id in range(num_clusters):
        _ToT_max = np.ushort(0)
        for event_num in range(max_cluster):
            event = cluster_arr[cluster_id, event_num]
            if event > -1:  # if we have an event here
                if events[event, 3] > _ToT_max:  # find the max ToT, assign, use that time
                    _ToT_max = events[event, 3]
                    t[cluster_id] = events[event, 4]
                    ToT_max[cluster_id] = _ToT_max
                xc[cluster_id] += events[event, 1] * events[event, 3]  # x and y centroids by time over threshold
                yc[cluster_id] += events[event, 2] * events[event, 3]
                ToT_sum[cluster_id] += events[event, 3]  # calcuate sum
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


def cent_to_numpy(
    cluster_arr: np.ndarray, events: int, num_clusters: int, max_cluster: int
) -> Dict[str, np.ndarray]:
    """
    Wrapper function to perform ingest_cent_data(cluster_arr_to_cent())

    Parameters
    ----------
    cluster_arr : np.ndarray
        The array of cluster events from create_cluster_arr()
    events : int
        Number of photon events
    num_clusters : int
        The total number of clusters
    max_cluster : int
        The number of events in the largest clust

    Returns
    -------
    Dict[str, np.ndarray]
       Keys of t, xc, yc, ToT_max, ToT_sum, and n (number of events) in each cluster.
    """
    return ingest_cent_data(cluster_arr_to_cent(cluster_arr, events, num_clusters, max_cluster))


def cent_to_df(
    cd_np: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """
    Returns the centroided dataframe from the zipped inputs.

    Parameters
    ----------
    cd_np : Dict[str, np.ndarray]
        Dictionary of the clustered data.

    Returns
    -------
    pd.DataFrame
        Time sorted dataframe of the centroids.
    """
    cent_df = pd.DataFrame(cd_np)
    return cent_df.sort_values("t").reset_index(drop=True)


def raw_df_to_cluster_df(
    raw_df: pd.DataFrame, tw: int = DEFAULT_CLUSTER_TW, radius: int = DEFAULT_CLUSTER_RADIUS
) -> pd.DataFrame:
    """
    Uses functions defined herein to take Dataframe of raw data and return dataframe of clustered data.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Pandas DataFrame of the raw data
    tw : int
        The time window to be considered "coincident" for clustering purposes
    radius : int
        The search radius, using Euclidean distance of x, y, timestamp/tw

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame of the centroided data.
    """
    filt_cond_raw_df = drop_zero_tot(condense_raw_df(raw_df))
    events, clusters = neighbor_set_from_df(filt_cond_raw_df, tw, radius)
    num_clusters, max_cluster = cluster_stats(clusters)
    cluster_arr = create_cluster_arr(clusters, num_clusters, max_cluster)
    return cent_to_df(cent_to_numpy(cluster_arr, events, num_clusters, max_cluster))


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
        df.loc[df['xc'] >= 255.5, 'xc'] += 2
        df.loc[df['yc'] >= 255.5, 'yc'] += 2
    df["x"] = np.round(df["xc"]).astype(np.uint16)
    df["y"] = np.round(df["yc"]).astype(np.uint16)
    df["t_ns"] = df["t"] / 4096 * 25

    return df


"""
A bunch of functions to help process multiple related .tpx3 files into Pandas dataframes stored in .h5 files.
""" 
RAW_H5_SUFFIX = ""
CENT_H5_SUFFIX = "_cent"
CONCAT_H5_SUFFIX = "_cent"


def extract_fpaths_from_sid(
    sid: int
) -> List[str]:
    """
    Extract file paths from a given sid.
    
    Parameters 
    ----------
    sid : int
        Short ID of a BlueSky scan
        
    Returns
    -------
    List[str]
        Filepaths of the written .tpx3, as recorded in Tiled    
    """
    return list(db[sid].table()["tpx3_files_raw_filepaths"].to_numpy()[0])


def extract_uid_from_fpaths(
    fpaths: List[str]
) -> str:
    """
    Extract scan unique ID from file paths.
    
    Parameters
    ----------
    fpaths : List[str]
        List of the filepaths.
        
    Returns
    -------
    str
        String of the first file's unique ID.
    
    """
    return os.path.basename(fpaths[0])[:23]


def extract_dir_from_fpaths(
    fpaths: List[str]
) -> str:
    """
    Extract directory from file paths.
    
    Parameters
    ----------
    fpaths : List[str]
        List of the filepaths.
        
    Returns
    -------
    str 
        String of the first file's directory.     
    
    """
    return os.path.dirname(fpaths[0])


def extract_uid_from_sid(
    sid: int
) -> str:
    """
    Extract user ID from a given sid.
    
    Parameters
    ----------
    sid : int
    
    Returns
    -------
    str
        String of the short ID's corresponding unique ID.
        
    """
    return extract_uid_from_fpaths(extract_fpaths_from_sid(sid))


def convert_file(
    fpath: Union[str, Path], time_window_microsecond: float = DEFAULT_CLUSTER_TW_MICROSECONDS, radius: int = DEFAULT_CLUSTER_RADIUS, print_details: bool = False
):
    """
    Convert a .tpx3 file into raw and centroided Pandas dataframes, which are stored in .h5 files.
    
    Parameters
    ----------
    fpath : Union[str, Path]
        .tpx3 file path
    time_window_microsecond : float = DEFAULT_CLUSTER_TW_MICROSECONDS
        The time window, in microseconds, to perform centroiding
    radius : int = DEFAULT_CLUSTER_RADIUS
        The radius, in pixels, to perform centroiding
    print_details : bool = False
        Boolean toggle about whether to print detailed data.
    """
    fname, ext = os.path.splitext(fpath)
    dfname = "{}{}.h5".format(fname, RAW_H5_SUFFIX)
    dfcname = "{}{}.h5".format(fname, CONCAT_H5_SUFFIX)
    
    if ext == ".tpx3" and os.path.exists(fpath):
        file_size = os.path.getsize(fpath)
        have_df = os.path.exists(dfname)
        have_dfc = os.path.exists(dfcname)

        if have_df and have_dfc:
            print("-> {} exists, skipping.".format(dfname))
        else:
            print("-> Processing {}, size: {:.1f} MB".format(fpath, file_size/1000000))
            time_window = time_window_microsecond * 1e-6
            time_stamp_conversion = 6.1e-12
            timedif = int(time_window / time_stamp_conversion)
            
            if print_details:
                print("Loading {} data into dataframe...".format(fpath))
            df = raw_to_sorted_df(fpath)
            num_events = df.shape[0]
            
            if print_details:
                print("Loading {} complete. {} events found. Saving to: {}".format(fpath, num_events, dfname))
            df.to_hdf(dfname, key='df', mode='w')
            
            if print_details:
                print("Saving {} complete. Beginning clustering...".format(dfname))
            df_c = raw_df_to_cluster_df(df, timedif, radius)
            num_clusters = df_c.shape[0]
            
            if print_details:
                print("Clustering {} complete. {} clusters found. Saving to {}".format(fpath, num_clusters, dfcname))
            df_c.to_hdf(dfcname, key='df', mode='w')
            print("Saving {} complete. Moving onto next file.".format(dfcname))
    else:
        print("File not found. Moving onto next file.")
        
            
def convert_tpx3_parallel(
    fpaths: Union[str, Path], num_workers: int = None
):
    """
    Convert a list of .tpx3 files in a parallel processing pool.
    
    Parameters
    ----------
    fpaths : Union[str, Path]
        .tpx3 file paths to convert in a parallel processing pool.
    num_workers : int = None
        Number of parallel workers to employ.
    """
    if num_workers == None:
        num_cores = multiprocessing.cpu_count()
        max_workers = num_cores-1
    else:
        max_workers = num_workers
    
    with multiprocessing.Pool(processes=max_workers) as pool:
        pool.map(convert_file, fpaths)
    
    print("Parallel conversion complete")
    

def convert_tpx3_st(fpaths: Union[str, Path]):
    """
    Convert a list of .tpx3 files in a single thread.
    
    Parameters
    ----------
    fpaths : Union[str, Path]
        .tpx3 file paths to convert in a single thread.
    """
    for file in fpaths:
        convert_file(file)
        

def get_cent_files(
    uid: str, dir_name: Union[str, Path]
) -> List[str]:
    """
    Gets a list of the centroided .h5 files from a given uid, sorted by sequence number.
    
    Parameters
    ----------
    uid : str
        The unique ID of the scan of which we want to get the files.
        
    dir_name : Union[str, path]
        Directory to look in for the files.
        
    Returns
    -------
    List[str]
        List of the centroided file paths.
    """
    cent_files = [
        os.path.join(dir_name, file)
        for file in os.listdir(dir_name)
        if file.endswith("{}.h5".format(CENT_H5_SUFFIX)) and str(uid) in file and len(os.path.basename(file)) == 44
    ]

    cent_files.sort(key=lambda f: int(os.path.splitext(os.path.basename(f))[0].split("_")[-2]))
    return cent_files


def concat_cent_files(
    cfpaths: List[Union[str, Path]]
):
    """
    Concatenates several centroided files together.
    
    Parameters
    ----------
    cfpaths : List[str, Path]
        List of the centroided .h5 files to concatenate together.
    """
    dir_name = os.path.dirname(cfpaths[0])
    uid = extract_uid_from_fpaths(cfpaths)
    
    dfs = [pd.read_hdf(fpath, key='df') for fpath in tqdm(cfpaths)]
    combined_df = pd.concat(dfs).reset_index(drop=True)
    
    save_path = os.path.join(dir_name, "{}{}.h5".format(uid, CONCAT_H5_SUFFIX))
    combined_df.to_hdf(save_path, key='df', mode='w')
    
    print("-> Saving complete.")
    

def get_con_cent_file(
    sid: int
) -> str:
    """
    Gets the location of the concatenated centroid files of a given sid.
    
    Parameters
    ----------
    sid : int
        Short ID of whichto get the centroided file path
        
    Returns
    -------
    str
        Path of the centroided file.
    """
    fpaths = extract_fpaths_from_sid(sid)
    uid = extract_uid_from_fpaths(fpaths)
    dir_name = extract_dir_from_fpaths(fpaths)
    cfpath = os.path.join(dir_name, "{}{}.h5".format(uid, CONCAT_H5_SUFFIX))
    
    if os.path.exists(cfpath):
        return cfpath
    else:
        print("-> Warning: {} does not exist".format(cfpath))
        return None

    
def convert_sids(
    sids: List[int]
):
    """
    Convert given sids by converting each .tpx3 file and then concatenating results together into a master dataframe.
    
    Parameters
    ----------
    sids : List[int]
        List of BlueSky scans' short IDs to convert.
    """
    
    for sid in sids:
        print("\n\n---> Beginning sid: {} <---\n".format(sid))
        
        tpx3fpaths = extract_fpaths_from_sid(sid)
        dir_name = extract_dir_from_fpaths(tpx3fpaths)
        num_tpx = len(tpx3fpaths)
        uid = extract_uid_from_fpaths(tpx3fpaths)

        convert_tpx3_parallel(tpx3fpaths, num_workers=16)
        centfpaths = get_cent_files(uid, dir_name)
        num_cent = len(centfpaths)

        if num_tpx == num_cent:
            print("---> Conversion numbers match")
            concat_cent_files(centfpaths)
        else:
            print("---> Warning: conversion mismatch: tpx3={}, cent={}".format(num_tpx, num_cent))

        print("---> Done with {}!".format(sid))
        gc.collect() 
