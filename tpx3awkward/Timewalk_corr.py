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
from _utils import cluster_df_optimized


TIMESTAMP_VALUE = 1.5625*1e-9 # each raw timestamp is 1.5625 seconds
MICROSECOND = 1e-6
DEFAULT_CLUSTER_RADIUS = 3
DEFAULT_CLUSTER_TW_MICROSECONDS = 0.3
DEFAULT_CLUSTER_TW = int(DEFAULT_CLUSTER_TW_MICROSECONDS * MICROSECOND / TIMESTAMP_VALUE)

tw = DEFAULT_CLUSTER_TW
radius = DEFAULT_CLUSTER_RADIUS

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

