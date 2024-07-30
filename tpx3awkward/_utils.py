import os
from pathlib import Path
from typing import TypeVar, Union, Dict, List, Tuple
import numpy as np
from numpy.typing import NDArray
import numba
import pandas as pd
from scipy.spatial import KDTree
import multiprocessing
from tqdm import tqdm
import warnings


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
    heartbeat_time = np.uint64(0)

    photon_count, chip_indx, msg_run_count, expected_msg_count = 0, 0, 0, 0
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
            _x, _y, _tot, _ts = decode_message(msg, chip_indx, heartbeat_time=heartbeat_time)
            x[photon_count] = _x
            y[photon_count] = _y
            tot[photon_count] = _tot
            ts[photon_count] = _ts
            
            if (photon_count > 0) and (_ts > ts[photon_count-1]) and (_ts - ts[photon_count-1] > 2**30):
                prev_ts = ts[:photon_count]   # This portion needs to be adjusted
                # Find what the current timestamp would be without global heartbeat
                _, _, _, _ts_0 = decode_message(msg, chip_indx, heartbeat_time=np.uint64(0))
                # Check if there is a SPIDR rollover in the beginning of the file but before heartbeat was received
                head_max = max(prev_ts[:10])
                tail_min = min(prev_ts[-10:])
                # Compare the difference with some big number (e.g. 1/4 of SPIDR)
                if (head_max > tail_min) and (head_max - tail_min > 2**32):
                    prev_ts[prev_ts < 2**33] += np.uint64(2**34)
                    _ts_0 += np.uint64(2**34)
                # Ajust already processed timestamps
                ts[:photon_count] = prev_ts + (_ts - _ts_0)
            
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
                # timer LSB, 32 bits of time
                heartbeat_lsb = (msg >> np.uint(16)) & np.uint64(0xFFFFFFFF)
            elif subheader == 0x5:
                # timer MSB -- only matters if LSB has been received already
                if heartbeat_lsb is not None:
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

        else:
            raise Exception(f"Not supported: {msg}")


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
    return raw_df.sort_values("t").reset_index(drop=True) # should we specify the sorting algorithm? at this point, it should be sorted anyway, but I think dataframes need to be explicitly sorted for use in e.g. merge_asof?


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


def cluster_df(
    df: pd.DataFrame, tw: int = DEFAULT_CLUSTER_TW, radius: int = DEFAULT_CLUSTER_RADIUS
) -> tuple[np.ndarray, np.ndarray]:
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
        df[["t", "x", "y", "ToT", "t"]].values # raw data stored in dataframe. duplicate timestamp column as the first instance is windowed
    )  # first three columns are for search radius of KDTree
    events[:, 0] = np.floor_divide(events[:, 0], tw)  # bin by the time window
    tree = KDTree(events[:, :3])  # generate KDTree based off the coordinates (t/timewindow, x, y)
    neighbors = tree.query_ball_tree(
        tree, radius
    )  # compare tree against itself to find neighbors within the search radius
    if len(neighbors) >= 2147483647: # performance is marginally better if can use int32 for the indices, so check for that
        dtype = np.int64
    else:
        dtype = np.int32
    return pd.DataFrame(neighbors).fillna(-1).astype(dtype).drop_duplicates().values, events[:, 1:] # a bit weird, but faster than using sets and list operators to unpack neighbors


@numba.jit(nopython=True, cache=True)
def centroid_clusters(
    cluster_arr: np.ndarray, events: np.ndarray
) -> tuple[np.ndarray]:  
    """
    Performs the centroiding of a group of clusters using Numba.  Note I originally attempted to unpack the clusters using list comprehensions, but this approach is significantly faster.

    Parameters
    ----------
    clusters : nd.array
        The numpy representation of the clusters' event indices.
    events : nd.array
        The numpy represetation of the event data.

    Returns
    -------
    tuple[np.ndarray]
        t, xc, yc, ToT_max, ToT_sum, and n (number of events) in each cluster.
    """
    num_clusters = cluster_arr.shape[0]
    max_cluster = cluster_arr.shape[1]
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


def raw_df_to_cent_df(
    df: pd.DataFrame, tw: int = DEFAULT_CLUSTER_TW, radius: int = DEFAULT_CLUSTER_RADIUS
) -> pd.DataFrame:
    """
    Uses functions defined herein to take Dataframe of raw data and return dataframe of clustered data.

    Parameters
    ----------
    df : pd.DataFrame
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
    fdf = drop_zero_tot(df)
    cluster_arr, events = cluster_df(fdf, tw, radius)
    data = centroid_clusters(cluster_arr, events)
    return pd.DataFrame(ingest_cent_data(data)).sort_values("t").reset_index(drop=True) # should we specify the sort type here? this should be *almost* completely sorted already


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


"""
Functions to help process multiple related .tpx3 files into Pandas dataframes stored in .h5 files.
""" 
RAW_H5_SUFFIX = ""
CENT_H5_SUFFIX = "_cent"
CONCAT_H5_SUFFIX = "_cent"


def convert_tpx_file(
    fpath: Union[str, Path], time_window_microsecond: float = DEFAULT_CLUSTER_TW_MICROSECONDS, radius: int = DEFAULT_CLUSTER_RADIUS, print_details: bool = False, overwrite: bool = True
):
    """
    Convert a .tpx3 file into raw and centroided Pandas dataframes, which are stored in .h5 files.
    
    TO DO: Args to specify output directory (default will be same directory as .tpx3 file as is now).
    
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
    overwrite : bool = True
        Boolean toggle about whether to overwrite pre-existing data.
        
    """
    fname, ext = os.path.splitext(fpath)
    dfname = "{}{}.h5".format(fname, RAW_H5_SUFFIX)
    dfcname = "{}{}.h5".format(fname, CONCAT_H5_SUFFIX)
    
    if ext == ".tpx3" and os.path.exists(fpath):
        
        try: 
            
            file_size = os.path.getsize(fpath)
            have_df = os.path.exists(dfname)
            have_dfc = os.path.exists(dfcname)

            if have_df and have_dfc and not overwrite:
                
                print("-> {} exists, skipping.".format(dfname))
                
            else:
                
                if print_details:
                    print("-> Processing {}, size: {:.1f} MB".format(fpath, file_size/1000000))
                
                time_window = time_window_microsecond * 1e-6
                time_stamp_conversion = 6.1e-12
                timedif = int(time_window / time_stamp_conversion)

                if print_details:
                    print("Loading {} data into dataframe...".format(fpath))
                    
                df = tpx_to_raw_df(fpath)
                num_events = df.shape[0]

                if print_details:
                    print("Loading {} complete. {} events found. Saving to: {}".format(fpath, num_events, dfname))
                    
                df.to_hdf(dfname, key="df", format="table", mode="w")

                if print_details:
                    print("Saving {} complete. Beginning clustering...".format(dfname))
                    
                df_c = raw_df_to_cent_df(df, timedif, radius)
                
                num_clusters = df_c.shape[0]

                if print_details:
                    print("Clustering {} complete. {} clusters found. Saving to {}".format(fpath, num_clusters, dfcname))
                    
                df_c.to_hdf(dfcname, key="df", format="table", mode="w")
                
                if print_details:
                    print("Saving {} complete. Moving onto next file.".format(dfcname))
                
        except Exception as e:
                
            if print_details:
                print("Conversion of {} failed due to {}, moving on.".format(fpath,e))
                
    else:
        if print_details:
            print("File not found. Moving onto next file.")


def convert_tpx3_files_parallel(fpaths: Union[List[str], List[Path]], num_workers: int = None):
    """
    Convert a list .tpx3 files in parallel using multiprocessing and convert_tpx_file().
    
    TO DO: Accept more arguments for convert_tpx_file.
    
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
    overwrite : bool = True
        Boolean toggle about whether to overwrite pre-existing data.
        
    """
    if num_workers is None:
        max_workers = multiprocessing.cpu_count()
    else:
        max_workers = num_workers

    with multiprocessing.Pool(processes=max_workers) as pool:
        for _ in tqdm(pool.imap_unordered(convert_tpx_file, fpaths), total=len(fpaths)):
            pass
        
        
def convert_tpx3_files(fpaths: Union[List[str], List[Path]], print_details = True):
    """
    Convert a list .tpx3 files in a single process using convert_tpx_file().
    
    TO DO: Accept more arguments for convert_tpx_file.
    
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
    overwrite : bool = True
        Boolean toggle about whether to overwrite pre-existing data.
        
    """
    for file in fpaths:
        convert_tpx_file(file, print_details = print_details)
        
        