from scipy.ndimage import uniform_filter
import numpy as np
import pandas as pd
import time
from pathlib import Path
from typing import Dict, Optional
from urban_analysis_lib import *


def compute_cv_on_front(X1985, front_mask, window):
    """
    Compute CV of 1985 built density only inside the expansion ring:
        front_mask = LCC2015 - LCC1985
    """

    X = X1985.astype(np.float32)
    M = front_mask.astype(np.float32)

    # Weighted sum of X inside ring
    sum_X = uniform_filter(X * M, window, mode='constant')
    # How many pixels of mask in the window
    count_M = uniform_filter(M, window, mode='constant')

    # Mean
    mean = sum_X / (count_M + 1e-6)

    # Binary: X² = X
    sum_X2 = uniform_filter((X * M)**2, window, mode='constant')
    mean_sq = sum_X2 / (count_M + 1e-6)

    var = mean_sq - mean**2
    var = np.clip(var, 0, None)
    std = np.sqrt(var)

    cv = std / (mean + 1e-6)

    # Outside ring = NaN
    cv[M == 0] = np.nan
    cv[count_M < 1] = np.nan

    return cv



def analyze_local_density_cv(
    wsf_data: np.ndarray,
    analysis_year: int = 1985,
    delimiter_year: int = 2015,
    center_row: int=0,
    center_col:int=0,
    window_size: int = 5,
    analyzer: Optional[BuiltAreaAnalyzer] = None,
    output_csv: Optional[str] = None
) -> Dict:

    start_time = time.time()

    if analyzer is None:
        analyzer = BuiltAreaAnalyzer()

    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")

    pixel_size_km = 0.03
    pixel_area_km2 = pixel_size_km ** 2
    window_size_km = window_size * pixel_size_km

    print(f"\n{'='*70}")
    print(f"Analyzing Local CV (2015 LCC FRONT ONLY: 2015 LCC minus 1985 LCC)")
    print(f"{'='*70}\n")

    # ------------------------------------------------------------
    # Step 1: Extract 2015 LCC (analysis boundary)
    # ------------------------------------------------------------
    print(f"Step 1: Extracting {delimiter_year} LCC...")
    mask_2015 = analyzer.extract_year_mask(wsf_data, delimiter_year)
    # lcc_2015, lcc_2015_size = analyzer.find_city_specific_lcc(mask_2015,cen)

    lcc_2015, lcc_2015_size = analyzer.find_city_specific_lcc(mask_2015,center_row,center_col,80)

    if lcc_2015_size == 0:
        return {"error": "No 2015 LCC detected"}

    # bounding box
    coords_2015 = np.argwhere(lcc_2015 == 1)
    row_min, row_max = coords_2015[:, 0].min(), coords_2015[:, 0].max()
    col_min, col_max = coords_2015[:, 1].min(), coords_2015[:, 1].max()

    # ------------------------------------------------------------
    # Step 2: Extract 1985 LCC and compute FRONT = LCC2015 − LCC1985
    # ------------------------------------------------------------
    print(f"\nStep 2: Extracting {analysis_year} LCC...")
    mask_1985 = analyzer.extract_year_mask(wsf_data, analysis_year)
    lcc_1985, lcc_1985_size = analyzer.find_largest_connected_component(mask_1985)

    # compute expansion ring
    front_mask_full = (lcc_2015 == 1) & (lcc_1985 == 0)

    # crop all three arrays
    region_1985 = mask_1985[row_min:row_max+1, col_min:col_max+1].astype(np.uint8)
    region_2015_lcc = lcc_2015[row_min:row_max+1, col_min:col_max+1]
    front_mask = front_mask_full[row_min:row_max+1, col_min:col_max+1]

    region_shape = region_1985.shape
    region_area_km2 = region_shape[0] * region_shape[1] * pixel_area_km2

    print(f"  Front-mask pixels: {front_mask.sum():,} (2015 minus 1985 LCC)")
    print(f"  Region shape: {region_shape}")

    if front_mask.sum() == 0:
        return {"error": "Expansion ring is empty (1985 already covers 2015 LCC)"}

    # ------------------------------------------------------------
    # Step 3: CV Computation ONLY INSIDE THE FRONT RING
    # ------------------------------------------------------------
    print("\nStep 3: Computing CV inside expansion ring...")
    t0 = time.time()

    cv_map = compute_cv_on_front(region_1985, front_mask, window_size)

    print(f"  CV computed in {time.time() - t0:.2f}s")

    # ------------------------------------------------------------
    # Step 4: Extract valid CV values (only inside ring)
    # ------------------------------------------------------------
    print("\nStep 4: Extracting valid CV values...")
    valid_mask = front_mask & ~np.isnan(cv_map)

    masked_arr = np.ma.array(cv_map, mask=~valid_mask)

    # ------------------------------------------------------------
    # Step 5: CV statistics
    # ------------------------------------------------------------
    print("\nStep 5: Computing CV statistics...")

    cv_stats = {
        "min": float(masked_arr.min()),
        "max": float(masked_arr.max()),
        "mean": float(masked_arr.mean()),
        "median": float(np.ma.median(masked_arr)),
        "std": float(masked_arr.std()),
        "valid_count": int(masked_arr.count()),
    }

    print(f"  Valid CV count: {cv_stats['valid_count']:,}")
    print(f"  Mean CV: {cv_stats['mean']:.4f}")

    # ------------------------------------------------------------
    # Step 6: Save output
    # ------------------------------------------------------------
    if output_csv:
        df = pd.DataFrame(cv_map)
        df.to_csv(output_csv, index=False, header=False)
        print(f"\nSaved CV map to: {output_csv}")

    print(f"\nTotal time: {time.time() - start_time:.2f}s")
    print(f"{'='*70}\n")

    return {
        "analysis_year": analysis_year,
        "delimiter_year": delimiter_year,
        "window_size": window_size,
        "window_size_km": round(window_size_km, 4),
        "region_shape": region_shape,
        "region_area_km2": round(region_area_km2, 2),
        "front_pixel_count": int(front_mask.sum()),
        "cv_map": cv_map,
        "cv_stats": cv_stats,
    }





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_cv_heatmap(csv_file, title=None, cmap="viridis", figsize=(14, 8),csv_output=None):
    """
    Plot a heatmap of the CV raster saved by analyze_local_density_cv().
    """
    import matplotlib as mpl
    mpl.rcParams['xtick.labelsize'] = 16
    mpl.rcParams['ytick.labelsize'] = 16
    mpl.rcParams['axes.labelsize']=16
    mpl.rcParams['legend.fontsize']=16

    print(f"\nLoading CV map from: {csv_file}")
    cv = pd.read_csv(csv_file, header=None).values

    # Optional: convert all-zero rows to NaN if needed
    # (Sometimes uniform_filter returns edges that are all-zero)
    cv_masked = np.ma.array(cv, mask=np.isnan(cv))

    plt.figure(figsize=figsize)
    im = plt.imshow(cv_masked, cmap=cmap, interpolation="nearest")
    plt.colorbar(im, label="Coefficient of Variation (CV)")

    if title:
        plt.title(title)

    plt.axis("off")
    plt.tight_layout()
    

    if csv_output is not None:
        plt.savefig(csv_output)
    
    plt.show()









#!/usr/bin/env python3
"""
Plot histogram of CV values from CSV file
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_cv_histogram(
    csv_path: str,
    bins: int = 50,
    figsize: tuple = (12, 6),
    output_path: str = None,
    title: str = None,
    exclude_zeros: bool = True,
    log_scale: bool = False
):
    """
    Plot histogram of CV values from a CSV file.
    
    Parameters
    ----------
    csv_path : str
        Path to CSV file containing CV map
    bins : int, default=50
        Number of histogram bins
    figsize : tuple, default=(12, 6)
        Figure size (width, height)
    output_path : str, optional
        Path to save figure. If None, displays interactively.
    title : str, optional
        Custom title for the plot. If None, uses default.
    exclude_zeros : bool, default=True
        If True, excludes zero/NaN values from histogram
    log_scale : bool, default=False
        If True, uses log scale for y-axis
        
    Returns
    -------
    dict
        Dictionary with statistics:
        - 'count': number of values
        - 'mean': mean CV
        - 'median': median CV
        - 'std': standard deviation
        - 'min': minimum CV
        - 'max': maximum CV
        - 'percentiles': dict with 25th, 50th, 75th, 90th, 95th, 99th percentiles
    
    Example
    -------
    >>> stats = plot_cv_histogram(
    ...     'cv_map_paris_1985.csv',
    ...     bins=100,
    ...     output_path='cv_histogram.png'
    ... )
    >>> print(f"Mean CV: {stats['mean']:.4f}")
    """
    
    print(f"\n{'='*70}")
    print(f"Plotting CV Histogram")
    print(f"{'='*70}")
    print(f"Loading: {csv_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path, header=None)
    cv_map = df.values
    
    print(f"  Shape: {cv_map.shape}")
    print(f"  Total pixels: {cv_map.size:,}")
    
    # Flatten and remove NaN/zeros if requested
    cv_values = cv_map.flatten()
    
    if exclude_zeros:
        # Remove NaN and zeros
        cv_values = cv_values[~np.isnan(cv_values) & (cv_values > 0)]
        print(f"  Valid values (excluding NaN and zeros): {len(cv_values):,}")
    else:
        # Remove only NaN
        cv_values = cv_values[~np.isnan(cv_values)]
        print(f"  Valid values (excluding NaN): {len(cv_values):,}")
    
    if len(cv_values) == 0:
        print("  ⚠️  No valid values to plot!")
        return None
    
    # Calculate statistics
    stats = {
        'count': int(len(cv_values)),
        'mean': float(cv_values.mean()),
        'median': float(np.median(cv_values)),
        'std': float(cv_values.std()),
        'min': float(cv_values.min()),
        'max': float(cv_values.max()),
        'percentiles': {
            'p25': float(np.percentile(cv_values, 25)),
            'p50': float(np.percentile(cv_values, 50)),
            'p75': float(np.percentile(cv_values, 75)),
            'p90': float(np.percentile(cv_values, 90)),
            'p95': float(np.percentile(cv_values, 95)),
            'p99': float(np.percentile(cv_values, 99))
        }
    }
    
    print(f"\n  Statistics:")
    print(f"    Count:  {stats['count']:,}")
    print(f"    Mean:   {stats['mean']:.6f}")
    print(f"    Median: {stats['median']:.6f}")
    print(f"    Std:    {stats['std']:.6f}")
    print(f"    Min:    {stats['min']:.6f}")
    print(f"    Max:    {stats['max']:.6f}")
    print(f"  Percentiles:")
    print(f"    25th: {stats['percentiles']['p25']:.6f}")
    print(f"    50th: {stats['percentiles']['p50']:.6f}")
    print(f"    75th: {stats['percentiles']['p75']:.6f}")
    print(f"    90th: {stats['percentiles']['p90']:.6f}")
    print(f"    95th: {stats['percentiles']['p95']:.6f}")
    print(f"    99th: {stats['percentiles']['p99']:.6f}")
    
    # Create figure
    print(f"\n  Creating histogram...")
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histogram
    n, bins_edges, patches = ax.hist(
        cv_values, 
        bins=bins, 
        edgecolor='black', 
        linewidth=0.5,
        alpha=0.7,
        color='steelblue'
    )
    
    # Add vertical lines for mean and median
    ax.axvline(stats['mean'], color='red', linestyle='--', 
               linewidth=2, label=f"Mean = {stats['mean']:.4f}")
    ax.axvline(stats['median'], color='green', linestyle='--', 
               linewidth=2, label=f"Median = {stats['median']:.4f}")
    
    # Labels and title
    ax.set_xlabel('Coefficient of Variation (CV)', fontsize=16)
    ax.set_ylabel('Frequency', fontsize=16)
    
    if title is None:
        title = f'Distribution of Local CV Values\n(n={stats["count"]:,}, mean={stats["mean"]:.4f}, std={stats["std"]:.4f})'

    
    # Log scale if requested
    if log_scale:
        ax.set_yscale('log')
        ax.set_ylabel('Frequency (log scale)', fontsize=16)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    ax.legend(fontsize=16, loc='upper right')
    
    # Add text box with statistics
    textstr = '\n'.join([
        f'Count: {stats["count"]:,}',
        f'Mean: {stats["mean"]:.4f}',
        f'Median: {stats["median"]:.4f}',
        f'Std: {stats["std"]:.4f}',
        f'Range: [{stats["min"]:.4f}, {stats["max"]:.4f}]'
    ])
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=props)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"  ✓ Saved: {output_path}")
        print(f"  File size: {file_size_mb:.2f} MB")
    else:
        plt.show()
    
    plt.close(fig)
    
    print(f"{'='*70}\n")
    
    return stats


def plot_cv_histogram_with_spatial_map(
    csv_path: str,
    bins: int = 50,
    figsize: tuple = (16, 6),
    output_path: str = None,
    title: str = None,
    exclude_zeros: bool = True,
    vmin: float = None,
    vmax: float = None
):
    """
    Plot histogram alongside spatial map of CV values.
    
    Creates a 2-panel figure:
    - Left: Spatial map (heatmap) of CV values
    - Right: Histogram of CV values
    
    Parameters
    ----------
    csv_path : str
        Path to CSV file containing CV map
    bins : int, default=50
        Number of histogram bins
    figsize : tuple, default=(16, 6)
        Figure size (width, height)
    output_path : str, optional
        Path to save figure. If None, displays interactively.
    title : str, optional
        Custom title for the figure
    exclude_zeros : bool, default=True
        If True, excludes zero/NaN values from histogram (but shows in map)
    vmin, vmax : float, optional
        Min/max values for colormap. If None, uses data range.
        
    Returns
    -------
    dict
        Statistics dictionary (same as plot_cv_histogram)
    
    Example
    -------
    >>> stats = plot_cv_histogram_with_spatial_map(
    ...     'cv_map_paris_1985.csv',
    ...     output_path='cv_analysis.png'
    ... )
    """
    
    print(f"\n{'='*70}")
    print(f"Plotting CV Histogram with Spatial Map")
    print(f"{'='*70}")
    print(f"Loading: {csv_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path, header=None)
    cv_map = df.values
    
    print(f"  Shape: {cv_map.shape}")
    print(f"  Total pixels: {cv_map.size:,}")
    
    # Flatten and compute statistics
    cv_values = cv_map.flatten()
    
    if exclude_zeros:
        cv_values_clean = cv_values[~np.isnan(cv_values) & (cv_values > 0)]
        print(f"  Valid values (excluding NaN and zeros): {len(cv_values_clean):,}")
    else:
        cv_values_clean = cv_values[~np.isnan(cv_values)]
        print(f"  Valid values (excluding NaN): {len(cv_values_clean):,}")
    
    if len(cv_values_clean) == 0:
        print("  ⚠️  No valid values to plot!")
        return None
    
    # Calculate statistics
    stats = {
        'count': int(len(cv_values_clean)),
        'mean': float(cv_values_clean.mean()),
        'median': float(np.median(cv_values_clean)),
        'std': float(cv_values_clean.std()),
        'min': float(cv_values_clean.min()),
        'max': float(cv_values_clean.max()),
        'percentiles': {
            'p25': float(np.percentile(cv_values_clean, 25)),
            'p50': float(np.percentile(cv_values_clean, 50)),
            'p75': float(np.percentile(cv_values_clean, 75)),
            'p90': float(np.percentile(cv_values_clean, 90)),
            'p95': float(np.percentile(cv_values_clean, 95)),
            'p99': float(np.percentile(cv_values_clean, 99))
        }
    }
    
    print(f"\n  Statistics:")
    print(f"    Mean:   {stats['mean']:.6f}")
    print(f"    Median: {stats['median']:.6f}")
    print(f"    Std:    {stats['std']:.6f}")
    
    # Create figure with 2 subplots
    print(f"\n  Creating figure...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Left panel: Spatial map
    if vmin is None:
        vmin = np.nanpercentile(cv_map, 1)  # Use 1st percentile to avoid outliers
    if vmax is None:
        vmax = np.nanpercentile(cv_map, 99)  # Use 99th percentile
    
    im = ax1.imshow(cv_map, cmap='viridis', vmin=vmin, vmax=vmax, interpolation='nearest')
    ax1.set_title('Spatial Distribution of Local CV', fontsize=12, pad=10)
    ax1.set_xlabel('Column (pixel)', fontsize=10)
    ax1.set_ylabel('Row (pixel)', fontsize=10)
    ax1.axis('equal')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('CV Value', fontsize=10)
    
    # Right panel: Histogram
    n, bins_edges, patches = ax2.hist(
        cv_values_clean, 
        bins=bins, 
        edgecolor='black', 
        linewidth=0.5,
        alpha=0.7,
        color='steelblue'
    )
    
    # Add vertical lines for mean and median
    ax2.axvline(stats['mean'], color='red', linestyle='--', 
                linewidth=2, label=f"Mean = {stats['mean']:.4f}")
    ax2.axvline(stats['median'], color='green', linestyle='--', 
                linewidth=2, label=f"Median = {stats['median']:.4f}")
    
    ax2.set_xlabel('Coefficient of Variation (CV)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of CV Values', fontsize=12, pad=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=9)
    
    # Add statistics text box
    textstr = '\n'.join([
        f'n = {stats["count"]:,}',
        f'μ = {stats["mean"]:.4f}',
        f'σ = {stats["std"]:.4f}',
        f'Range: [{stats["min"]:.3f}, {stats["max"]:.3f}]'
    ])
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax2.text(0.98, 0.97, textstr, transform=ax2.transAxes, 
             fontsize=9, verticalalignment='top', horizontalalignment='right',
             bbox=props)
    
    # Overall title
    if title:
        fig.suptitle(title, fontsize=14, y=1.02)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"  ✓ Saved: {output_path}")
        print(f"  File size: {file_size_mb:.2f} MB")
    else:
        plt.show()
    
    plt.close(fig)
    
    print(f"{'='*70}\n")
    
    return stats




#!/usr/bin/env python3
"""
Plot histogram of CV values from CSV file
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_cv_histogram(
    csv_path: str,
    bins: int = 50,
    figsize: tuple = (12, 6),
    output_path: str = None,
    title: str = None,
    exclude_zeros: bool = True,
    log_scale: bool = False
):
    """
    Plot histogram of CV values from a CSV file.
    
    Parameters
    ----------
    csv_path : str
        Path to CSV file containing CV map
    bins : int, default=50
        Number of histogram bins
    figsize : tuple, default=(12, 6)
        Figure size (width, height)
    output_path : str, optional
        Path to save figure. If None, displays interactively.
    title : str, optional
        Custom title for the plot. If None, uses default.
    exclude_zeros : bool, default=True
        If True, excludes zero/NaN values from histogram
    log_scale : bool, default=False
        If True, uses log scale for y-axis
        
    Returns
    -------
    dict
        Dictionary with statistics:
        - 'count': number of values
        - 'mean': mean CV
        - 'median': median CV
        - 'std': standard deviation
        - 'min': minimum CV
        - 'max': maximum CV
        - 'percentiles': dict with 25th, 50th, 75th, 90th, 95th, 99th percentiles
    
    Example
    -------
    >>> stats = plot_cv_histogram(
    ...     'cv_map_paris_1985.csv',
    ...     bins=100,
    ...     output_path='cv_histogram.png'
    ... )
    >>> print(f"Mean CV: {stats['mean']:.4f}")
    """
    
    print(f"\n{'='*70}")
    print(f"Plotting CV Histogram")
    print(f"{'='*70}")
    print(f"Loading: {csv_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path, header=None)
    cv_map = df.values
    
    print(f"  Shape: {cv_map.shape}")
    print(f"  Total pixels: {cv_map.size:,}")
    
    # Flatten and remove NaN/zeros if requested
    cv_values = cv_map.flatten()
    
    if exclude_zeros:
        # Remove NaN and zeros
        cv_values = cv_values[~np.isnan(cv_values) & (cv_values > 0)]
        print(f"  Valid values (excluding NaN and zeros): {len(cv_values):,}")
    else:
        # Remove only NaN
        cv_values = cv_values[~np.isnan(cv_values)]
        print(f"  Valid values (excluding NaN): {len(cv_values):,}")
    
    if len(cv_values) == 0:
        print("  ⚠️  No valid values to plot!")
        return None
    
    # Calculate statistics
    stats = {
        'count': int(len(cv_values)),
        'mean': float(cv_values.mean()),
        'median': float(np.median(cv_values)),
        'std': float(cv_values.std()),
        'min': float(cv_values.min()),
        'max': float(cv_values.max()),
        'percentiles': {
            'p25': float(np.percentile(cv_values, 25)),
            'p50': float(np.percentile(cv_values, 50)),
            'p75': float(np.percentile(cv_values, 75)),
            'p90': float(np.percentile(cv_values, 90)),
            'p95': float(np.percentile(cv_values, 95)),
            'p99': float(np.percentile(cv_values, 99))
        }
    }
    
    print(f"\n  Statistics:")
    print(f"    Count:  {stats['count']:,}")
    print(f"    Mean:   {stats['mean']:.6f}")
    print(f"    Median: {stats['median']:.6f}")
    print(f"    Std:    {stats['std']:.6f}")
    print(f"    Min:    {stats['min']:.6f}")
    print(f"    Max:    {stats['max']:.6f}")
    print(f"  Percentiles:")
    print(f"    25th: {stats['percentiles']['p25']:.6f}")
    print(f"    50th: {stats['percentiles']['p50']:.6f}")
    print(f"    75th: {stats['percentiles']['p75']:.6f}")
    print(f"    90th: {stats['percentiles']['p90']:.6f}")
    print(f"    95th: {stats['percentiles']['p95']:.6f}")
    print(f"    99th: {stats['percentiles']['p99']:.6f}")
    
    # Create figure
    print(f"\n  Creating histogram...")
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histogram
    n, bins_edges, patches = ax.hist(
        cv_values, 
        bins=bins, 
        edgecolor='black', 
        linewidth=0.5,
        alpha=0.7,
        color='steelblue'
    )
    
    # Add vertical lines for mean and median
    ax.axvline(stats['mean'], color='red', linestyle='--', 
               linewidth=2, label=f"Mean = {stats['mean']:.4f}")
    ax.axvline(stats['median'], color='green', linestyle='--', 
               linewidth=2, label=f"Median = {stats['median']:.4f}")
    
    # Labels and title
    ax.set_xlabel('Coefficient of Variation (CV)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    
    if title is None:
        title = f'Distribution of Local CV Values\n(n={stats["count"]:,}, mean={stats["mean"]:.4f}, std={stats["std"]:.4f})'
    ax.set_title(title, fontsize=14, pad=15)
    
    # Log scale if requested
    if log_scale:
        ax.set_yscale('log')
        ax.set_ylabel('Frequency (log scale)', fontsize=12)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    ax.legend(fontsize=10, loc='upper right')
    
    # Add text box with statistics
    textstr = '\n'.join([
        f'Count: {stats["count"]:,}',
        f'Mean: {stats["mean"]:.4f}',
        f'Median: {stats["median"]:.4f}',
        f'Std: {stats["std"]:.4f}',
        f'Range: [{stats["min"]:.4f}, {stats["max"]:.4f}]'
    ])
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=props)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"  ✓ Saved: {output_path}")
        print(f"  File size: {file_size_mb:.2f} MB")
    else:
        plt.show()
    
    plt.close(fig)
    
    print(f"{'='*70}\n")
    
    return stats


def plot_cv_histogram_with_spatial_map(
    csv_path: str,
    bins: int = 50,
    figsize: tuple = (16, 6),
    output_path: str = None,
    title: str = None,
    exclude_zeros: bool = True,
    vmin: float = None,
    vmax: float = None
):
    """
    Plot histogram alongside spatial map of CV values.
    
    Creates a 2-panel figure:
    - Left: Spatial map (heatmap) of CV values
    - Right: Histogram of CV values
    
    Parameters
    ----------
    csv_path : str
        Path to CSV file containing CV map
    bins : int, default=50
        Number of histogram bins
    figsize : tuple, default=(16, 6)
        Figure size (width, height)
    output_path : str, optional
        Path to save figure. If None, displays interactively.
    title : str, optional
        Custom title for the figure
    exclude_zeros : bool, default=True
        If True, excludes zero/NaN values from histogram (but shows in map)
    vmin, vmax : float, optional
        Min/max values for colormap. If None, uses data range.
        
    Returns
    -------
    dict
        Statistics dictionary (same as plot_cv_histogram)
    
    Example
    -------
    >>> stats = plot_cv_histogram_with_spatial_map(
    ...     'cv_map_paris_1985.csv',
    ...     output_path='cv_analysis.png'
    ... )
    """
    
    print(f"\n{'='*70}")
    print(f"Plotting CV Histogram with Spatial Map")
    print(f"{'='*70}")
    print(f"Loading: {csv_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path, header=None)
    cv_map = df.values
    
    print(f"  Shape: {cv_map.shape}")
    print(f"  Total pixels: {cv_map.size:,}")
    
    # Flatten and compute statistics
    cv_values = cv_map.flatten()
    
    if exclude_zeros:
        cv_values_clean = cv_values[~np.isnan(cv_values) & (cv_values > 0)]
        print(f"  Valid values (excluding NaN and zeros): {len(cv_values_clean):,}")
    else:
        cv_values_clean = cv_values[~np.isnan(cv_values)]
        print(f"  Valid values (excluding NaN): {len(cv_values_clean):,}")
    
    if len(cv_values_clean) == 0:
        print("  ⚠️  No valid values to plot!")
        return None
    
    # Calculate statistics
    stats = {
        'count': int(len(cv_values_clean)),
        'mean': float(cv_values_clean.mean()),
        'median': float(np.median(cv_values_clean)),
        'std': float(cv_values_clean.std()),
        'min': float(cv_values_clean.min()),
        'max': float(cv_values_clean.max()),
        'percentiles': {
            'p25': float(np.percentile(cv_values_clean, 25)),
            'p50': float(np.percentile(cv_values_clean, 50)),
            'p75': float(np.percentile(cv_values_clean, 75)),
            'p90': float(np.percentile(cv_values_clean, 90)),
            'p95': float(np.percentile(cv_values_clean, 95)),
            'p99': float(np.percentile(cv_values_clean, 99))
        }
    }
    
    print(f"\n  Statistics:")
    print(f"    Mean:   {stats['mean']:.6f}")
    print(f"    Median: {stats['median']:.6f}")
    print(f"    Std:    {stats['std']:.6f}")
    
    # Create figure with 2 subplots
    print(f"\n  Creating figure...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Left panel: Spatial map
    if vmin is None:
        vmin = np.nanpercentile(cv_map, 1)  # Use 1st percentile to avoid outliers
    if vmax is None:
        vmax = np.nanpercentile(cv_map, 99)  # Use 99th percentile
    
    im = ax1.imshow(cv_map, cmap='viridis', vmin=vmin, vmax=vmax, interpolation='nearest')
    ax1.set_title('Spatial Distribution of Local CV', fontsize=12, pad=10)
    ax1.set_xlabel('Column (pixel)', fontsize=10)
    ax1.set_ylabel('Row (pixel)', fontsize=10)
    ax1.axis('equal')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('CV Value', fontsize=10)
    
    # Right panel: Histogram
    n, bins_edges, patches = ax2.hist(
        cv_values_clean, 
        bins=bins, 
        edgecolor='black', 
        linewidth=0.5,
        alpha=0.7,
        color='steelblue'
    )
    
    # Add vertical lines for mean and median
    ax2.axvline(stats['mean'], color='red', linestyle='--', 
                linewidth=2, label=f"Mean = {stats['mean']:.4f}")
    ax2.axvline(stats['median'], color='green', linestyle='--', 
                linewidth=2, label=f"Median = {stats['median']:.4f}")
    
    ax2.set_xlabel('Coefficient of Variation (CV)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of CV Values', fontsize=12, pad=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=9)
    
    # Add statistics text box
    textstr = '\n'.join([
        f'n = {stats["count"]:,}',
        f'μ = {stats["mean"]:.4f}',
        f'σ = {stats["std"]:.4f}',
        f'Range: [{stats["min"]:.3f}, {stats["max"]:.3f}]'
    ])
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax2.text(0.98, 0.97, textstr, transform=ax2.transAxes, 
             fontsize=9, verticalalignment='top', horizontalalignment='right',
             bbox=props)
    
    # Overall title
    if title:
        fig.suptitle(title, fontsize=14, y=1.02)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"  ✓ Saved: {output_path}")
        print(f"  File size: {file_size_mb:.2f} MB")
    else:
        plt.show()
    
    plt.close(fig)
    
    print(f"{'='*70}\n")
    
    return stats


def compare_cities_cv(
    city_data: dict,
    output_path: str = None,
    figsize: tuple = (16, 10),
    exclude_zeros: bool = True
):
    """
    Compare CV distributions across multiple cities.
    
    Creates a comprehensive comparison figure with:
    - Box plots showing distributions
    - Summary statistics table
    - Overlaid histograms (optional)
    
    Parameters
    ----------
    city_data : dict
        Dictionary mapping city names to CV map CSV paths.
        Example: {
            'Paris': 'cv_paris_1985.csv',
            'London': 'cv_london_1985.csv',
            'Berlin': 'cv_berlin_1985.csv'
        }
    output_path : str, optional
        Path to save figure
    figsize : tuple, default=(16, 10)
        Figure size (width, height)
    exclude_zeros : bool, default=True
        If True, excludes zero/NaN values
        
    Returns
    -------
    pd.DataFrame
        Comparison table with statistics for each city
        
    Example
    -------
    >>> cities = {
    ...     'Paris': 'cv_paris_1985.csv',
    ...     'London': 'cv_london_1985.csv',
    ...     'New York': 'cv_nyc_1985.csv'
    ... }
    >>> comparison_df = compare_cities_cv(cities, output_path='city_comparison.png')
    >>> print(comparison_df)
    """
    
    print(f"\n{'='*70}")
    print(f"Comparing CV Across {len(city_data)} Cities")
    print(f"{'='*70}\n")
    
    # Load data for all cities
    city_cv_data = {}
    city_stats = {}
    
    for city_name, csv_path in city_data.items():
        print(f"Loading {city_name}: {csv_path}")
        
        df = pd.read_csv(csv_path, header=None)
        cv_map = df.values
        cv_values = cv_map.flatten()
        
        if exclude_zeros:
            cv_values = cv_values[~np.isnan(cv_values) & (cv_values > 0)]
        else:
            cv_values = cv_values[~np.isnan(cv_values)]
        
        city_cv_data[city_name] = cv_values
        
        # Calculate statistics
        city_stats[city_name] = {
            'count': len(cv_values),
            'mean': cv_values.mean(),
            'median': np.median(cv_values),
            'std': cv_values.std(),
            'min': cv_values.min(),
            'max': cv_values.max(),
            'p25': np.percentile(cv_values, 25),
            'p75': np.percentile(cv_values, 75),
            'p90': np.percentile(cv_values, 90),
            'p95': np.percentile(cv_values, 95)
        }
        
        print(f"  n={len(cv_values):,}, mean={cv_values.mean():.4f}, median={np.median(cv_values):.4f}")
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(city_stats).T
    comparison_df = comparison_df.round(6)
    
    print(f"\n{'='*70}")
    print("Summary Statistics:")
    print(f"{'='*70}")
    print(comparison_df.to_string())
    print(f"{'='*70}\n")
    
    # Create figure with 3 panels
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1.5, 1], hspace=0.3, wspace=0.3)
    
    # Panel 1: Box plot comparison
    ax1 = fig.add_subplot(gs[0, :])
    
    # Prepare data for box plot
    box_data = [city_cv_data[city] for city in city_data.keys()]
    box_labels = list(city_data.keys())
    
    bp = ax1.boxplot(box_data, labels=box_labels, patch_artist=True,
                     showmeans=True, meanline=True,
                     medianprops=dict(color='red', linewidth=2),
                     meanprops=dict(color='blue', linewidth=2, linestyle='--'))
    
    # Color boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(city_data)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax1.set_ylabel('CV Value', fontsize=12)
    ax1.set_title('Distribution Comparison (Box Plot)', fontsize=14, pad=15)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add legend for median/mean
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=2, label='Median'),
        Line2D([0], [0], color='blue', linewidth=2, linestyle='--', label='Mean')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Panel 2: Overlaid histograms
    ax2 = fig.add_subplot(gs[1, :])
    
    for (city_name, cv_values), color in zip(city_cv_data.items(), colors):
        ax2.hist(cv_values, bins=50, alpha=0.5, label=city_name, 
                color=color, edgecolor='black', linewidth=0.5)
    
    ax2.set_xlabel('CV Value', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Overlaid Distributions', fontsize=14, pad=15)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Bar chart of key metrics
    ax3 = fig.add_subplot(gs[2, 0])
    
    cities = list(city_data.keys())
    means = [city_stats[city]['mean'] for city in cities]
    medians = [city_stats[city]['median'] for city in cities]
    
    x = np.arange(len(cities))
    width = 0.35
    
    ax3.bar(x - width/2, means, width, label='Mean', color='steelblue', alpha=0.7)
    ax3.bar(x + width/2, medians, width, label='Median', color='coral', alpha=0.7)
    
    ax3.set_xlabel('City', fontsize=11)
    ax3.set_ylabel('CV Value', fontsize=11)
    ax3.set_title('Mean vs Median', fontsize=12, pad=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(cities, rotation=45, ha='right')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Panel 4: Variability comparison (std deviation)
    ax4 = fig.add_subplot(gs[2, 1])
    
    stds = [city_stats[city]['std'] for city in cities]
    
    ax4.bar(cities, stds, color='mediumseagreen', alpha=0.7)
    ax4.set_xlabel('City', fontsize=11)
    ax4.set_ylabel('Standard Deviation', fontsize=11)
    ax4.set_title('CV Variability', fontsize=12, pad=10)
    ax4.set_xticklabels(cities, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Multi-City CV Comparison', fontsize=16, y=0.995)
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"✓ Saved: {output_path}")
        print(f"File size: {file_size_mb:.2f} MB\n")
    else:
        plt.show()
    
    plt.close(fig)
    
    return comparison_df


def compare_cities_simple(
    city_data: dict,
    output_path: str = None,
    figsize: tuple = (12, 6),
    exclude_zeros: bool = True
):
    """
    Simple comparison showing just mean and key percentiles.
    
    Best for quick comparisons and presentations.
    
    Parameters
    ----------
    city_data : dict
        Dictionary mapping city names to CV map CSV paths
    output_path : str, optional
        Path to save figure
    figsize : tuple, default=(12, 6)
        Figure size
    exclude_zeros : bool, default=True
        Exclude zeros/NaN
        
    Returns
    -------
    pd.DataFrame
        Summary statistics table
        
    Example
    -------
    >>> cities = {'Paris': 'cv_paris.csv', 'London': 'cv_london.csv'}
    >>> df = compare_cities_simple(cities, output_path='simple_comparison.png')
    """
    
    print(f"\n{'='*70}")
    print(f"Simple Comparison: {len(city_data)} Cities")
    print(f"{'='*70}\n")
    
    # Load and compute statistics
    stats_list = []
    
    for city_name, csv_path in city_data.items():
        print(f"Loading {city_name}...")
        
        df = pd.read_csv(csv_path, header=None)
        cv_map = df.values
        cv_values = cv_map.flatten()
        
        if exclude_zeros:
            cv_values = cv_values[~np.isnan(cv_values) & (cv_values > 0)]
        else:
            cv_values = cv_values[~np.isnan(cv_values)]
        
        stats_list.append({
            'City': city_name,
            'Count': len(cv_values),
            'Mean': cv_values.mean(),
            'Median': np.median(cv_values),
            'Std': cv_values.std(),
            'P90': np.percentile(cv_values, 90),
            'P95': np.percentile(cv_values, 95),
            'Max': cv_values.max()
        })
    
    summary_df = pd.DataFrame(stats_list)
    summary_df = summary_df.round(6)
    
    print("\nSummary:")
    print(summary_df.to_string(index=False))
    print()
    
    # Create simple bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    cities = summary_df['City'].values
    x = np.arange(len(cities))
    
    # Left: Mean and Median
    ax1.bar(x - 0.2, summary_df['Mean'], 0.4, label='Mean', 
            color='steelblue', alpha=0.7)
    ax1.bar(x + 0.2, summary_df['Median'], 0.4, label='Median', 
            color='coral', alpha=0.7)
    
    ax1.set_ylabel('CV Value', fontsize=12)
    ax1.set_title('Central Tendency', fontsize=13, pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(cities, rotation=45, ha='right')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for i, (mean, median) in enumerate(zip(summary_df['Mean'], summary_df['Median'])):
        ax1.text(i - 0.2, mean, f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
        ax1.text(i + 0.2, median, f'{median:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Right: Variability (Std and 90th percentile)
    ax2.bar(x - 0.2, summary_df['Std'], 0.4, label='Std Dev', 
            color='mediumseagreen', alpha=0.7)
    ax2.bar(x + 0.2, summary_df['P90'], 0.4, label='90th %ile', 
            color='orange', alpha=0.7)
    
    ax2.set_ylabel('CV Value', fontsize=12)
    ax2.set_title('Variability & High Values', fontsize=13, pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(cities, rotation=45, ha='right')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add values
    for i, (std, p90) in enumerate(zip(summary_df['Std'], summary_df['P90'])):
        ax2.text(i - 0.2, std, f'{std:.3f}', ha='center', va='bottom', fontsize=8)
        ax2.text(i + 0.2, p90, f'{p90:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('City Comparison: Local CV Statistics', fontsize=14, y=0.98)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}\n")
    else:
        plt.show()
    
    plt.close(fig)
    
    return summary_df


  