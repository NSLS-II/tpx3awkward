import numba
import numpy as np
import pandas as pd

from .corrections import estimate_energies, timewalk_corr_exp, trim_corr

TIMESTAMP_VALUE = 1.5625 * 1e-9  # each raw timestamp is 1.5625 seconds
MICROSECOND = 1e-6

# We have had decent success with these values, but do not know for sure if they are optimal.
DEFAULT_CLUSTER_RADIUS = 3
DEFAULT_CLUSTER_TW_MICROSECONDS = 0.3

DEFAULT_CLUSTER_TW = int(DEFAULT_CLUSTER_TW_MICROSECONDS * MICROSECOND / TIMESTAMP_VALUE)


def cluster(
    df,
    tw=DEFAULT_CLUSTER_TW,
    radius=DEFAULT_CLUSTER_RADIUS,
    include_energy: bool = False,
):
    cols = ["t", "x", "y", "ToT", "t"]

    if include_energy:
        cols.append("e")

    events = df[cols].to_numpy()
    events[:, 0] = np.floor_divide(events[:, 0], tw)  # Bin timestamps into time windows

    labels = get_cluster_labels(events, tw, radius)

    return labels, events[:, 1:]


@numba.jit(nopython=True, cache=True)
def get_cluster_labels(events, tw=DEFAULT_CLUSTER_TW, radius=DEFAULT_CLUSTER_RADIUS):
    n = len(events)
    labels = np.full(n, -1, dtype=np.int64)
    cluster_id = 0

    max_time = radius * tw  # maximum time difference allowed for clustering
    radius_sq = radius**2

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
    cluster_arr: np.ndarray,
    events: np.ndarray,
    include_energy: bool = False,
    timewalk_correct: bool = False,
) -> tuple[np.ndarray]:

    num_clusters = cluster_arr.shape[0]
    max_cluster = cluster_arr.shape[1]
    t = np.zeros(num_clusters, dtype="uint64")
    xc = np.zeros(num_clusters, dtype="float32")
    yc = np.zeros(num_clusters, dtype="float32")
    ToT_max = np.zeros(num_clusters, dtype="uint32")
    ToT_sum = np.zeros(num_clusters, dtype="uint32")
    n = np.zeros(num_clusters, dtype="ubyte")
    e_sum = np.zeros(num_clusters, dtype="float32")
    t_corr = np.zeros(num_clusters, dtype="uint64")

    for cluster_id in range(num_clusters):
        _ToT_max = np.ushort(0)
        for event_num in range(max_cluster):
            event = cluster_arr[cluster_id, event_num]
            if event > -1:  # if we have an event here
                if events[event, 2] > _ToT_max:  # find the max ToT, assign, use that time
                    _ToT_max = events[event, 2]
                    t[cluster_id] = events[event, 3]
                    if timewalk_correct:
                        t_corr[cluster_id] = events[event, 3] - timewalk_corr_exp(_ToT_max)
                    ToT_max[cluster_id] = _ToT_max
                xc[cluster_id] += events[event, 0] * events[event, 2]  # x and y centroids by time over threshold
                yc[cluster_id] += events[event, 1] * events[event, 2]
                ToT_sum[cluster_id] += events[event, 2]  # calcuate sum
                n[cluster_id] += np.ubyte(1)  # number of events in cluster

                if include_energy:
                    e_sum[cluster_id] += events[event, 4]
            else:
                break

        if ToT_sum[cluster_id] != 0:
            xc[cluster_id] /= ToT_sum[cluster_id]  # normalize
            yc[cluster_id] /= ToT_sum[cluster_id]

    return t, xc, yc, ToT_max, ToT_sum, n, e_sum, t_corr


def ingest_cent_data(
    data: np.ndarray, include_energy: bool = False, timewalk_correct: bool = False
) -> dict[str, np.ndarray]:
    """
    Performs the centroiding of a group of clusters.

    Parameters
    ----------
    data : np.ndarray
        The stream of cluster data from cluster_arr_to_cent()
    include_energy : bool, optional
        Whether the data includes energy sum (e_sum). Default is False.

    Returns
    -------
    Dict[str, np.ndarray]
        A dictionary with keys:
        ['t', 'xc', 'yc', 'ToT_max', 'ToT_sum', 'n']
        or
        ['t', 'xc', 'yc', 'ToT_max', 'ToT_sum', 'e_sum', 'n'] if include_energy=True
    """
    key_string = "t,xc,yc,ToT_max,ToT_sum,n"

    if include_energy:
        key_string += ",e_sum"
    if timewalk_correct:
        key_string += ",t_corr"

    keys = key_string.split(",")
    return dict(zip(keys, data, strict=False))


def cluster_raw_df(
    df: pd.DataFrame,
    tw: float = DEFAULT_CLUSTER_TW,
    radius: int = DEFAULT_CLUSTER_RADIUS,
    energy_calib: np.ndarray = None,
    timewalk_correct: bool = False,
    trim_correct: bool = False,
) -> pd.DataFrame:
    include_energy = isinstance(energy_calib, np.ndarray)
    # apply gap (needed for correct pixel mapping to energy calibrations)
    if trim_correct:
        df = trim_corr(df, trim_correct)
    if include_energy:
        df["e"] = estimate_energies(df["x"].to_numpy(), df["y"].to_numpy(), df["ToT"].to_numpy(), energy_calib)
    cluster_labels, events = cluster(df, tw, radius, include_energy=include_energy)
    df["cluster_id"] = cluster_labels
    cluster_array = group_indices(cluster_labels)
    data = centroid_clusters(
        cluster_array,
        events,
        include_energy=include_energy,
        timewalk_correct=timewalk_correct,
    )

    return (
        pd.DataFrame(ingest_cent_data(data, include_energy=include_energy, timewalk_correct=timewalk_correct))
        .sort_values("t")
        .reset_index(drop=True)
    )
