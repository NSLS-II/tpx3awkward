import _utils as tpx
import pandas as pd
import numpy as np
import numba
import os
from tiled.client import from_uri
import time 
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from io import StringIO
import itertools as itertools
from tqdm import tqdm
from fast_histogram import histogram2d
from scipy.optimize import curve_fit


TIMESTAMP_VALUE = 1.5625*1e-9 # each raw timestamp is 1.5625 seconds
MICROSECOND = 1e-6
DEFAULT_CLUSTER_RADIUS = 3
DEFAULT_CLUSTER_TW_MICROSECONDS = 0.3
DEFAULT_CLUSTER_TW = int(DEFAULT_CLUSTER_TW_MICROSECONDS * MICROSECOND / TIMESTAMP_VALUE)

tw = DEFAULT_CLUSTER_TW
radius = DEFAULT_CLUSTER_RADIUS

### modified functions for clustering below, probably no reason to change
def cluster_df_optimized(df, tw = tw, radius = radius):
    events = df[["t", "x", "y", "ToT", "t"]].values
    events[:, 0] = np.floor_divide(events[:, 0], tw)  # Bin timestamps into time windows

    labels = cluster_df3(events, radius, tw)

    return labels, events[:, 1:]


@numba.jit(nopython=True,cache=True)
def cluster_df3(events, radius, tw):
    n = len(events)
    labels = np.full(n, -1, dtype=np.int64)
    cluster_id = 0

    max_time = radius * tw  # Maximum time difference allowed for clustering
    radius_sq = radius ** 2  # Use squared radius to avoid unnecessary sqrt computation

    for i in range(n):
        if labels[i] == -1:  # If event is unclustered
            labels[i] = cluster_id
            for j in range(i + 1, n):  # Scan forward only
                if events[j, 4] - events[i, 4] > max_time:  # Early exit based on time
                    break
                # Compute squared Euclidean distance (without sqrt for performance)
                dx = events[i, 0] - events[j, 0]
                dy = events[i, 1] - events[j, 1]
                dt = events[i, 2] - events[j, 2]
                distance_sq = dx * dx + dy * dy + dt * dt

                if distance_sq <= radius_sq:  # Compare squared distance
                    labels[j] = cluster_id
            cluster_id += 1

    return labels

@numba.jit(nopython=True,cache=True)
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


@numba.jit(nopython=True,cache=True)
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

    return t, xc, yc, ToT_max, ToT_sum, 


def timewalk_corr_exp(ToT, b = 38.2, c = -0.0054): 
    return (b*np.exp(c*ToT)).astype(np.uint64)

def timewalk_corr_exp_fit(ToT, b = 38.2, c = -0.0054):
    return (b*np.exp(c*ToT)).astype(np.float64)

def find_timewalk_params(df, dt_range_low = -25, dt_range_hi = 45, ToT_limit = 700, params_p0 = [38, -0.005]):
    """
    Unique Parameters (see parent function for parameters not mentioned)
    ----------
    t_ns_adjusted: dt mask to filter for dt values between the dt_range_low and dt_range_hi parameters
    df_new: new dataframe using the t_ns_mask filter
    df_means: dataframe to house the mean ToT values per t_ns_adjusted value (grouped dataframe)
    
    """
    
    t_ns_mask = (df['t_ns_adjusted'] >= dt_range_low) & (df['t_ns_adjusted'] <= dt_range_hi) 
    df_new = df.loc[t_ns_mask]
    df_means = df_new.groupby('ToT')['t_ns_adjusted'].mean().reset_index()
    
    df_means = df_means.loc[df_means['ToT'] <= ToT_limit]
    
    popt, pcov = curve_fit(timewalk_corr_exp_fit, df_means['ToT'], df_means['t_ns_adjusted'], p0 = params_p0, bounds = ((0, -np.inf),(np.inf, 0)))
    b_fit = popt[0]
    c_fit = popt[1]
    
    plt.figure()
    
    plt.plot(df_means['ToT'], df_means['t_ns_adjusted'], '.')
    plt.plot(df_means['ToT'], timewalk_corr_exp(df_means['ToT'], *popt), '-r', label = f"Exp decay fit \n b = {b_fit :.4f} \n c = {c_fit :.4f} ")
    plt.legend()
    plt.show()

    return b_fit, c_fit

def plot_timewalk_distr(df, dt_range_low = -25, dt_range_hi = 50, ToT_limit = 700, params_p0 = [38, -0.005], timewalk_applied = False):
    """
    Unique Parameters (see parent function for parameters not mentioned)
    ----------
    idx: Index value of the largest ToT value in a cluster
    t_max_tot: Time value of the largest ToT value in a cluster
    valid_clusters: clusters with more than 1 event
    df_filtered: dataframe which filters for clusters with more than 1 event
    x_bins, y_bins: Sizes of the x and y bins used for the 2d histogram
    x_range, y_range: x and y ranges of the 2d histogram plot 
    histogram: Computed 2d histogram of t_ns_adjusted vs ToT values
    tot_vals, exp_vals: tot and exponential decay function values to be plotted
    im: plot of the histogram variable 
    hist1d, bins1d: counts and bins of the 1d histogram of delta t (t_ns_adjusted) values. 
    max_index: Index of the maximum hist1d count value to be excluded from the plot
    sigma: calculated standard deviation of the 1d histogram
    
    """
    
    # Compute max ToT per cluster
    idx = df.groupby('cluster_id')['ToT'].idxmax()
    t_max_tot = df.loc[idx, ['cluster_id', 't']].set_index('cluster_id')
    
    df = df.set_index('cluster_id')
    df['t_max_tot'] = t_max_tot['t']
    df = df.reset_index()
    df['t_adjusted'] = (df['t'] - df['t_max_tot']).astype(np.int64)
    df['t_ns_adjusted'] = df['t_adjusted'].astype(np.float64) * 1.5625
    
    # Filter clusters with more than one event
    valid_clusters = df['cluster_id'].value_counts()[lambda x: x > 1].index
    df_filtered = df[df['cluster_id'].isin(valid_clusters)]
    
    ## Finding best timewalk params 
    print("Fitting for best timewalk params...")
    if timewalk_applied!= True: 
        b_fit, c_fit = find_timewalk_params(df_filtered, dt_range_low, dt_range_hi, ToT_limit, params_p0)
    
    fig, (ax2d, ax1d) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2, 1]})
    
    # Histogram parameters
    x_bins, y_bins = 27, int(300 / 1.5625)
    x_range, y_range = (25, 700), (-150, 150)
    
    # Compute 2D histogram
    histogram = histogram2d(
        df_filtered['ToT'].values,
        df_filtered['t_ns_adjusted'].values,
        bins=(x_bins, y_bins),
        range=[x_range, y_range]
    )
    
    tot_vals = np.arange(25, 725, 25)
    exp_vals = timewalk_corr_exp(tot_vals)+1.5625/2
    
    # Plot 2D histogram
    im = ax2d.imshow(
        histogram.T, 
        origin='lower', 
        extent=[*x_range, *y_range], 
        aspect='auto', 
        cmap='inferno',
        norm=colors.LogNorm(vmax=1e3, vmin=1)
    )
    if timewalk_applied != True:
        ax2d.plot(tot_vals, exp_vals)
    ax2d.set(xlabel='Event ToT', ylabel='Event Δt (ns)')
    fig.colorbar(im, ax=ax2d, label='Counts')
    
    # Compute 1D histogram
    hist_1d, bins_1d = np.histogram(df_filtered['t_ns_adjusted'], bins=y_bins, range=y_range)
    
    # Exclude maximum bin and compute standard deviation
    max_index = np.argmax(hist_1d)
    filtered_data = df_filtered[
        (df_filtered['t_ns_adjusted'] != 0) & 
        (-125 < df_filtered['t_ns_adjusted']) & 
        (df_filtered['t_ns_adjusted'] < 125)
    ]['t_ns_adjusted'].values
    sigma = np.std(filtered_data)
    
    # Plot 1D histogram excluding the max bin
    ax1d.step(np.delete(bins_1d[:-1], max_index), np.delete(hist_1d, max_index), where='mid', color='black')
    ax1d.set(xlabel='Event Δt (ns)', ylabel='Counts', title=f'σ = {sigma:.2f}')
    ax1d.set_xlim(y_range)
    ax1d.grid(True, which='both', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    if timewalk_applied != True:
        return b_fit, c_fit


def timewalk_main(input_files, dt_range_low = -25, dt_range_hi = 40, ToT_limit = 700, params_p0 = [38, -0.005], timewalk_applied = False):
    """
    Parameters
    ----------
    
    input_files: List of files to be input and analyzed to determine b and c fit values
    dt_range_low: Low end of dt range for decay exponential fit
    dt_range_hi: High end of dt range for decay exponential fit
    ToT_limit: Limit value for ToTs to be fit
    params_p0: Iniital guess of best b and c values
    timewalk_applied: Boolean value to be assigned for other functions to determine if the timewalk correction has yet been applied
    all_dfs: List to house all the dataframes to be concatenated
    files_processed: List to house the file names of all files processed
    df: Individual dataframe created from an individual tpx3 file
    cluster_labels, events: The cluster labels and events found from the cluster_df_optimized function used to assign cluster id's
    b_fit, c_fit: Found best b and c fit values to use in the exponential decay function
    
    """
    

    all_dfs = []
    files_processed = []
    
    for file in input_files:
        files_processed.append(file)
        df = tpx.drop_zero_tot(tpx.tpx_to_raw_df(file, tram_correct=False))
        
        cluster_labels, events = cluster_df_optimized(df, tw, radius)
        df['cluster_id'] = cluster_labels
        
        all_dfs.append(df)

    df = pd.concat(all_dfs, ignore_index=True)
    
    b_fit, c_fit = plot_timewalk_distr(df, dt_range_low, dt_range_hi, ToT_limit, params_p0, timewalk_applied)
    
    
    print("After TW correction...")
    df['t'] -= timewalk_corr_exp(df['ToT'], b = b_fit, c = c_fit) 

    plot_timewalk_distr(df, timewalk_applied = True)

    return b_fit, c_fit

