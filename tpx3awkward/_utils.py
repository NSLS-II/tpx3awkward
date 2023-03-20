import os
import numpy as np
from pathlib import Path
from numpy.typing import NDArray
from typing import TypeVar, Union, Dict, Set
import numba
import pandas as pd
from scipy.spatial import KDTree
import concurrent.futures
import multiprocessing
import time
from tqdm import tqdm

IA = NDArray[np.uint64]
UnSigned = TypeVar("UnSigned", IA, np.uint64)

def raw_as_numpy(fpath: Union[str, Path]) -> IA:
    """
    Read raw tpx3 data file as a numpy array.

    Each entry is read as a uint8 (64bit unsigned-integer)

    Parameters
    ----------

    """
    with open(fpath, "rb") as fin:
        return np.frombuffer(fin.read(), dtype="<u8")


@numba.jit(nopython=True)
def get_block(v: UnSigned, width: int, shift: int) -> UnSigned:
    return v >> np.uint64(shift) & np.uint64(2**width - 1)


@numba.jit(nopython=True)
def is_packet_header(v: UnSigned) -> UnSigned:
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
    # identify packet headers by magic number (TPX3 as ascii on lowest 8 bytes]
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
def _ingest_raw_data(data: IA):
    types = np.zeros_like(data, dtype="<u1")
    # identify packet headers by magic number (TPX3 as ascii on lowest 8 bytes]
    is_header = is_packet_header(data)
    types[is_header] = 1
    # get the highest nibble
    nibble = data >> np.uint(60)
    # probably a better way to do this, but brute force!
    types[~is_header & (nibble == 0xB)] = 2
    types[~is_header & (nibble == 0x6)] = 3
    types[~is_header & (nibble == 0x4)] = 4
    types[~is_header & (nibble == 0x7)] = 5

    # sort out how many photons we have
    total_photons = np.sum(types == 2)

    # allocate the return arrays
    x = np.zeros(total_photons, dtype="u2")
    y = np.zeros(total_photons, dtype="u2")
    pix_addr = np.zeros(total_photons, dtype="u2")
    ToA = np.zeros(total_photons, dtype="u2")
    ToT = np.zeros(total_photons, dtype="u4")
    FToA = np.zeros(total_photons, dtype="u2")
    SPIDR = np.zeros(total_photons, dtype="u2")
    chip_number = np.zeros(total_photons, dtype="u1")
    basetime = np.zeros(total_photons, dtype="u8")
    timestamp = np.zeros(total_photons, dtype="u8")

    photon_offset = 0
    chip = np.uint16(0)
    expected_msg_count = np.uint16(0)
    msg_run_count = np.uint(0)

    heartbeat_lsb = np.uint64(0)
    heartbeat_msb = np.uint64(0)
    heartbeat_time = np.uint64(0)
    # loop over the packet headers (can not vectorize this with numpy)
    for j in range(len(data)):
        msg = data[j]
        typ = types[j]
        if typ == 1:
            # 1: packet header (id'd via TPX3 magic number)
            if expected_msg_count != msg_run_count:
                print("missing messages!", msg)
            # extract scalar information from the header

            # "number of pixels in chunk" is given in bytes not words
            # and means all words in the chunk, not just "photons"
            expected_msg_count = get_block(msg, 16, 48) // 8
            # what chip we are on
            chip = np.uint8(get_block(msg, 8, 32))
            msg_run_count = 0
        elif typ == 2 or typ == 6:
            #  2: photon event (id'd via 0xB upper nibble)
            #  6: frame driven data (id'd via 0xA upper nibble) (??)

            # |

            # pixAddr is 16 bits
            # these names and math are adapted from c++ code
            l_pix_addr = pix_addr[photon_offset] = (msg >> np.uint(44)) & np.uint(0xFFFF)
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
            col = x[photon_offset] = rowcol[1]
            y[photon_offset] = rowcol[0]
            # ToA is 14 bits
            ToA[photon_offset] = (msg >> np.uint(30)) & np.uint(0x3FFF)
            # ToT is 10 bits
            # report in ns
            ToT[photon_offset] = ((msg >> np.uint(20)) & np.uint(0x3FF)) * 25
            # FToA is 4 bits
            l_FToA = FToA[photon_offset] = (msg >> np.uint(16)) & np.uint(0xF)
            # SPIDR time is 16 bits
            SPIDR[photon_offset] = msg & np.uint(0xFFFF)
            # chip number (this is a constant)
            chip_number[photon_offset] = chip
            # heartbeat time
            basetime[photon_offset] = heartbeat_time

            ToA_coarse = (SPIDR[photon_offset] << np.uint(14)) | ToA[photon_offset]
            pixelbits = int((ToA_coarse >> np.uint(28)) & np.uint(0x3))
            heartbeat_time_bits = int((heartbeat_time >> np.uint(28)) & np.uint(0x3))
            diff = heartbeat_time_bits - pixelbits
            if diff == 1 or diff == -3:
                heartbeat_time -= np.uint(0x10000000)
            elif diff == -1 or diff == 3:
                heartbeat_time += np.uint(0x10000000)
            globaltime = (heartbeat_time & np.uint(0xFFFFC0000000)) | (ToA_coarse & np.uint(0x3FFFFFFF))


            timestamp[photon_offset] = (globaltime << np.uint(12)) - (l_FToA << np.uint(8))
            # correct for phase shift
            phase = np.uint((col / 2) % 16)
            if phase == 0:
                timestamp[photon_offset] += (16 << 8)
            else:
                timestamp[photon_offset] += (phase << 8)

            photon_offset += 1
            msg_run_count += 1
        elif typ == 3:
            #  3: TDC timstamp (id'd via 0x6 upper nibble)
            # TODO: handle these!
            msg_run_count += 1
        elif typ == 4:
            #  4: global timestap (id'd via 0x4 upper nibble)
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
                raise Exception("unknown header")

            msg_run_count += 1
        elif typ == 5:
            #  5: "command" data (id'd via 0x7 upper nibble)
            # TODO handle this!
            msg_run_count += 1
        else:
            raise Exception("Not supported")

    return x, y, pix_addr, ToA, ToT, FToA, SPIDR, chip_number, basetime, timestamp


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
       Keys of x, y, pix_addr, ToA, ToT, FToA, SPIDR, chip_number
    """
    return {
        k.strip(): v
        for k, v in zip(
            "x, y, pix_addr, ToA, ToT, FToA, SPIDR, chip_number, basetime, timestamp".split(","),
            _ingest_raw_data(data),
        )
    }


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
    return raw_df.sort_values('timestamp').reset_index(drop=True)


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
    cdf = df[['timestamp', 'x', 'y', 'ToT']]
    cdf = cdf.rename(columns={'timestamp': 't'}) # obviously not necessary, just easier to type 't' a lot than 'timestamp'
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
    fdf = df[df['ToT'] > 0]
    return fdf


# The next set of code is dedicated to performing clustering and centroiding on the raw data.
TIMESTAMP_VALUE = ((1e-9)/4096)*25
MICROSECOND = 10**(-6)

# We have had decent success with these values, but do not know for sure if they are optimal.
DEFAULT_CLUSTER_RADIUS = 2 
DEFAULT_CLUSTER_TW_MICROSECONDS = 0.5 

DEFAULT_CLUSTER_TW = int(DEFAULT_CLUSTER_TW_MICROSECONDS*MICROSECOND/TIMESTAMP_VALUE)


def neighbor_set_from_df(df: pd.DataFrame, tw: int = DEFAULT_CLUSTER_TW, radius: int = DEFAULT_CLUSTER_RADIUS) -> tuple[np.ndarray, Set[tuple[int]]]:
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
    events = np.array(df[['t', 'x', 'y', 'ToT', 't']].values) # first three columns are for search radius of KDTree
    events[:,0] = np.floor_divide(events[:,0], DEFAULT_CLUSTER_TW) # bin by the time window
    tree = KDTree(events[:,:3]) # generate KDTree based off the coordinates
    neighbors = tree.query_ball_tree(tree, DEFAULT_CLUSTER_RADIUS) # compare tree against itself to find neighbors within the search radius
    clusters = set(tuple(n) for n in neighbors) # turn the list of lists into a set of tuples
    return events, clusters


def cluster_stats(clusters: Set[tuple[int]]) -> tuple[int]:
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


def create_cluster_arr(clusters: Set[tuple[int]], num_clusters: int, max_cluster: int) -> np.ndarray: # is there a better way to do this?
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
    cluster_arr = np.full((num_clusters, max_cluster), -1, dtype=np.int64) # fill with -1; these will be passed later
    for cluster_num, cluster in enumerate(clusters): 
        for event_num, event in enumerate(cluster):
            cluster_arr[cluster_num, event_num] = event 
    return cluster_arr


@numba.jit(nopython=True)
def cluster_arr_to_cent(cluster_arr: np.ndarray, events: np.ndarray, num_clusters: int, max_cluster: int) -> tuple[np.ndarray]:
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
    t = np.zeros(num_clusters, dtype='uint64')
    xc = np.zeros(num_clusters, dtype='float32')
    yc = np.zeros(num_clusters, dtype='float32')
    ToT_max = np.zeros(num_clusters, dtype='uint16')
    ToT_sum = np.zeros(num_clusters, dtype='uint16')
    n = np.zeros(num_clusters, dtype='ubyte')
 
    for cluster_id in range(num_clusters):
        _ToT_max = np.ushort(0)
        for event_num in range(max_cluster):
            event = cluster_arr[cluster_id,event_num]
            if event > -1: # if we have an event here
                if events[event,3] > _ToT_max: # find the max ToT, assign, use that time
                    _ToT_max = events[event,3]
                    t[cluster_id] = events[event,4]
                    ToT_max[cluster_id] = _ToT_max
                xc[cluster_id] += events[event,1]*events[event,3] # x and y centroids by time over threshold
                yc[cluster_id] += events[event,2]*events[event,3]
                ToT_sum[cluster_id] += events[event,3] # calcuate sum
                n[cluster_id] += np.ubyte(1) # number of events in cluster
            else:
                break
        xc[cluster_id] /= ToT_sum[cluster_id] # normalize
        yc[cluster_id] /= ToT_sum[cluster_id]

    return t, xc, yc, ToT_max, ToT_sum, n 


def ingest_cent_data(data: np.ndarray) -> Dict[str, np.ndarray]:
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
            't, xc, yc, ToT_max, ToT_sum, n'.split(','),
            data,
        )
    }


def cent_to_numpy(cluster_arr, events, num_clusters, max_cluster) -> Dict[str, np.ndarray]:
    """
    Wrapper function to perform ingest_cent_data(cluster_arr_to_cent())

    Parameters
    ----------
    cluster_arr : np.ndarray
        The array of cluster events from create_cluster_arr()
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
    
    
def cent_to_df(cd_np: Dict[str, np.ndarray]) -> pd.DataFrame:
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
    return cent_df.sort_values('t').reset_index(drop=True)    


def raw_df_to_cluster_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Uses functions defined herein to take Dataframe of raw data and return dataframe of clustered data.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Pandas DataFrame of the raw data

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame of the centroided data.
    """ 
    filt_cond_raw_df = drop_zero_tot(condense_raw_df(raw_df))
    events, clusters = neighbor_set_from_df(filt_cond_raw_df)
    num_clusters, max_cluster = cluster_stats(clusters)
    cluster_arr = create_cluster_arr(clusters, num_clusters, max_cluster)
    return cent_to_df(cent_to_numpy(cluster_arr, events, num_clusters, max_cluster))


def process_raw_data(fpath: Union[str, Path], cluster: bool = True, sid: Union[str, int, None] = None, save_dir: Union[str, Path, None] = None, save_raw: bool = False, save_cent: bool = False, prints: bool = False) -> tuple[pd.DataFrame, Union[pd.DataFrame, None]]:
    """
    Loads raw data from a .tpx3 file, with options to cluster/centroid, and save the data (raw and/or centroids) in .h5 files in specified output directory.

    Parameters
    ----------
    fpath : Union[str, Path]
        Filepath of the .tpx3 file to be processed
    cluster : bool = True
        Whether to perform clustering/centroiding on the raw data.
    sid : Union[str, None] = None
        The sid of the raw id, to be used in the output file name(s).
    save_dir : Union[str, Path, None] = None
        The directory to save output .h5 files.
    save_raw : bool = False
        Whether or not to save the raw data to .h5.
    save_cent : bool = False
        Whether or not to save the centroided data to .h5.
    prints : bool = False
        Whether or not to print out status updates during execution.     
        

    Returns
    -------
    tuple[pd.DataFrame, None]
       Pandas DataFrame of the raw data and the clustered/centroided data, if generated.
    """ 
    if not isinstance(fpath, (str, Path)):
        raise TypeError('file_path must be a string or a path-like object')
    
    path_list = os.path.split(fpath)
    path_dir = path_list[0]
    fname = path_list[1]
    fname_str = os.path.splitext(fname)[0]
    fname_ext = os.path.splitext(fpath)[1]
    
    if not fpath.endswith('.tpx3') or not os.path.isfile(fpath):     
        raise ValueError('file_path does not point to a file with a .tpx3 extension')    
        
    if save_raw or save_cent:     
        if not os.path.isdir(save_dir):
            raise ValueError('save_dir is not a valid path')

    fsize = os.path.getsize(fpath)/(1*(10**6)) 

    if prints: print('--> Loading "{}" ({} MB, sid="{}"): {}'.format(fpath, fsize, sid, time.ctime()))
    if fsize > 0:
        rs_df = raw_to_sorted_df(fpath)
        raw_length = rs_df.shape[0]
        if prints: print('--> {} .tpx3 unpacked, {} events: {}'.format(sid, raw_length, time.ctime()))

        if save_raw:
            save_fname = '{}_raw.h5'.format(sid)
            save_path = save_dir + save_fname
            with pd.HDFStore(save_path, mode='w') as store:
                store.put('rs_df', rs_df)
            if prints: print('--> Saved raw to "{}" (sid="{}"): {}'.format(save_path, sid, time.ctime()))

        if cluster:
            cent_df = raw_df_to_cluster_df(rs_df)
            cent_length = cent_df.shape[0]

            if prints: print('--> Centroids computed (sid="{}"):, {} clusters: {}'.format(sid, cent_length, time.ctime()))

            if save_cent:
                save_fname = '{}_centroid.h5'.format(sid)
                save_path = save_dir + save_fname
                if prints: print('--> Saving centroid to "{}" (sid="{}"): {}'.format(save_path, sid, time.ctime()))
                with pd.HDFStore(save_path, mode='w') as store:
                    store.put('c_df', cent_df)

        else:
            c_df = None
            
    else:
        if prints: print('--> "{}" is empty, skipping.'.format(fpath))
  
    return rs_df, c_df


def _process_raw_data(args):
    """
    Helper(?) function to call process_raw_data within a concurrent.futures.ProcessPoolExecutor
    """ 
    return process_raw_data(args[0], args[1], args[2], args[3], args[4], args[5], args[6]) # there are probably better ways to do this, but it works, and I am a physicist not a computer scientist :)


def convert_directory(directory: Union[str, Path], cluster: bool = True, sid: Union[str, int, None] = None, save_dir: Union[str, Path, None] = None, save_raw: bool = False, save_cent: bool = False, prints: bool = False):
    """
    A function that converts an entire directory of .tpx3 files by calling process_raw_data in a parallel processing pool. If you only want to convert files with a certain sid in the filename, you can pass that in, otherwise will do all files.
    
    Parameters
    ----------
    directory : Union[str, Path]
        Path to the directory to grab .tpx3 files from to convert
    cluster : bool = True
        Whether to perform clustering/centroiding on the raw data.
    sid : Union[str, None] = None
        The sid, which is in the raw file path, to be converted. If None, will convert all files in the directory.
    save_dir : Union[str, Path, None] = None
        The directory to save output .h5 files.
    save_raw : bool = False
        Whether or not to save the raw data to .h5.
    save_cent : bool = False
        Whether or not to save the centroided data to .h5.
    prints : bool = False
        Whether or not to print out status updates during execution.     
        
    """    
    if sid == None:
        if prints: print('-> Converting {}'.format(directory)) 
        files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.tpx3')]
    else:
        if prints: print('-> Converting {} with filename containing {}'.format(directory, sid))
        files = [os.path.join(directory, file) for file in os.listdir(directory) if (file.endswith('.tpx3') and (str(sid) in file))]
    
    args = [[fpath, cluster, os.path.splitext(os.path.split(fpath)[1])[0], save_dir, save_raw, save_cent, prints] for fpath in files] # this probably highlights that I am not an advanced Python programmer :)
    
    max_workers = 16 # I am 100% sure DSSI will have better ways to manage resources to do this. Please feel free to adjust.
    
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            executor.map(_process_raw_data, args)
    except Exception as e:
        print(e)
        
    print('Done!')

    
def concat_data_seq(directory: Union[str, Path], sid: Union[str, None] = None, datatype: str = 'centroid', save: bool = False, prints: bool = False):
    """
    A function that concatenates several subscans in different .h5 files together into one big "master" scan.
    
    Parameters
    ----------
    directory : Union[str, Path]
        Path to the directory to grab .tpx3 files from to convert
    sid : Union[str, Path]
        The sid, which is in the .h5 file path, to be converted. If None, will concatenate tall files in the directory.
    datatype : str = 'centroid'
        Whether to concatenate the raw or centroided data.
    save : bool = False
        Whether to save the concatenated to new .h5 file or not.
    prints : bool = False
        Whether or not to print out status updates during execution.     
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing the concatenated data.
    """ 
    if prints: print('-> Concatenating {} with filename containing {}: {}'.format(directory, sid, time.ctime())) 
    files = [os.path.join(directory, file) for file in os.listdir(directory) if (file.endswith('.h5') and (str(sid) in file) and str(datatype) in file and not 'all' in file)]
    
    if datatype == 'centroid':
        dflag = 'c_df'
    elif datatype == 'raw':
        dflag = 'rs_df'
        
    start_t = 0
    dfs = []
    for ind, fpath in enumerate(tqdm(files)):
        file_size = os.path.getsize(fpath)
        if file_size > 0:
            with pd.HDFStore(fpath, mode='r') as store:
                new_df = store.get(dflag)
            new_df['t'] = new_df['t'] + start_t
            start_t = new_df['t'].iloc[-1]
            dfs.append(new_df)
            
    if prints: print('-> Concatenating dataframe list: {}'.format(time.ctime()))
            
    dfs = pd.concat(dfs).reset_index(drop=True)
    
    if prints: print('-> Concatenating complete: {}'.format(time.ctime()))
    
    if save:
        save_fname = '{}_all_{}.h5'.format(sid, datatype)
        save_path = directory + save_fname
        if prints: print('-> Saving all {} {} to "{}": {}'.format(sid, datatype, save_fname, time.ctime()))
        with pd.HDFStore(save_path, mode='w') as store:
            store.put(dflag, dfs)
            
        if prints: print('-> Saving complete.')
            
    return dfs


def add_centroid_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates centroid positions to the nearest pixel and the timestamp in nanoseconds.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input centroided dataframe
        
    Returns
    -------
    pd.DataFrame
        Originally dataframe with new columns x, y, and t_ns added.
    """    
    df['x'] = np.round(df['xc']).astype(np.uint16)
    df['y'] = np.round(df['yc']).astype(np.uint16)
    df['t_ns'] = df['t']/4096*25
    
    return df
