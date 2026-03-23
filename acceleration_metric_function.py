#!/usr/bin/env python3
"""
Urban Growth Analysis: Compare 1985 urbanization to LCC within 2015 boundary
=============================================================================

This script:
1. Downloads WSF data for a city
2. Extracts the LCC boundary from 2015
3. Finds all urbanized areas in 1985 within that 2015 LCC boundary
4. Finds the 1985 LCC
5. Calculates: (1985 urbanized area) - (1985 LCC area)

This metric shows how much "secondary urbanization" existed in 1985
within what would eventually become the 2015 LCC.
"""

import numpy as np
from pathlib import Path


# Import the urban analysis library functions
# (Assuming the library code is in a file called urban_analysis_lib.py)
from urban_analysis_lib import (
    WSFTileManager,
    BuiltAreaAnalyzer,
    geocode_city,
    print_system_info
)
import matplotlib.pyplot as plt



def analyze_urban_growth(city_name: str, 
                        radius_km: float = 50,
                        output_dir: str = "./output") -> dict:
    """
    Analyze urban growth by comparing 1985 urbanization to LCC.
    
    Adds:
    - Ratio of secondary urbanization (1985) to 2015 LCC area
    - Mean size of secondary urban clusters in 1985
    """
    
    print_system_info()


    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Step 1: Geocode city
    print(f"\n{'='*70}")
    print(f"STEP 1: Geocoding {city_name}")
    print(f"{'='*70}")
    
    center_lat, center_lon = geocode_city(city_name)

    # Step 2: Download WSF data
    print(f"\n{'='*70}")
    print(f"STEP 2: Downloading WSF Evolution Data")
    print(f"{'='*70}")
    
    tile_manager = WSFTileManager(cache_dir="./wsf_cache")
    download_result = tile_manager.download_region(center_lat, center_lon, radius_km)
    
    # Step 3: Load data
    print(f"\n{'='*70}")
    print(f"STEP 3: Loading and Merging Tiles")
    print(f"{'='*70}")
    
    analyzer = BuiltAreaAnalyzer()
    wsf_data, metadata = analyzer.load_tiles_from_download_result(download_result)
    center_row, center_col=analyzer.latlon_to_pixel(center_lat,center_lon,metadata['transform'])

    pixel_area_km2 = 0.03 * 0.03  # 30m × 30m pixels
    MIN_SECONDARY_CLUSTER_AREA_KM2 = 2.0
    MIN_SECONDARY_CLUSTER_PIXELS = int(MIN_SECONDARY_CLUSTER_AREA_KM2 / pixel_area_km2)
    
    # Step 4: Extract 2015 LCC boundary
    print(f"\n{'='*70}")
    print(f"STEP 4: Extracting 2015 LCC Boundary")
    print(f"{'='*70}")
    
    mask_2015 = analyzer.extract_year_mask(wsf_data, 2015)
    print(f"  Total urbanized pixels in 2015: {mask_2015.sum():,}")
    
    lcc_2015_mask, lcc_2015_size = analyzer.find_city_specific_lcc(mask_2015,center_row,center_col,max_distance_km=80)
    lcc_2015_area_km2 = lcc_2015_size * pixel_area_km2
    
    print(f"  2015 LCC size: {lcc_2015_size:,} pixels")
    print(f"  2015 LCC area: {lcc_2015_area_km2:.3f} km²")
    
    if lcc_2015_size == 0:
        return {
            'success': False,
            'error': 'No LCC found in 2015',
            'city': city_name
        }
    
    # Step 5: Extract 1985 urbanization within 2015 LCC boundary
    print(f"\n{'='*70}")
    print(f"STEP 5: Analyzing 1985 Urbanization within 2015 LCC Boundary")
    print(f"{'='*70}")
    
    mask_1985 = analyzer.extract_year_mask(wsf_data, 1985)
    print(f"  Total urbanized pixels in 1985 (entire region): {mask_1985.sum():,}")
    
    mask_1985_in_2015_lcc = mask_1985 & lcc_2015_mask
    pixels_1985_in_2015_lcc = mask_1985_in_2015_lcc.sum()
    area_1985_in_2015_lcc_km2 = pixels_1985_in_2015_lcc * pixel_area_km2
    
    print(f"  1985 urbanized pixels within 2015 LCC boundary: {pixels_1985_in_2015_lcc:,}")
    print(f"  1985 urbanized area within 2015 LCC boundary: {area_1985_in_2015_lcc_km2:.3f} km²")
    
    # Step 6: Find 1985 LCC and secondary cluster statistics
    print(f"\n{'='*70}")
    print(f"STEP 6: Finding 1985 Clusters within 2015 LCC Boundary")
    print(f"{'='*70}")
    
    from scipy import ndimage
    
    labeled_1985, num_clusters_1985 = ndimage.label(mask_1985_in_2015_lcc)
    print(f"  Total number of urbanized clusters in 1985: {num_clusters_1985:,}")
    
    if num_clusters_1985 == 0:
        lcc_1985_in_region_size = 0
        lcc_1985_in_region_area_km2 = 0.0
        mean_secondary_cluster_pixels = 0.0
        mean_secondary_cluster_km2 = 0.0
        mean_top10_secondary_pixels = 0.0
        mean_top10_secondary_km2 = 0.
    else:
        component_sizes = np.bincount(labeled_1985.ravel())[1:]
        largest_label = component_sizes.argmax() + 1
        lcc_1985_in_region_size = int(component_sizes[largest_label - 1])
        lcc_1985_in_region_area_km2 = lcc_1985_in_region_size * pixel_area_km2
        
        if num_clusters_1985 > 1:
            secondary_cluster_sizes = component_sizes[component_sizes != lcc_1985_in_region_size]

# Apply minimum size threshold (≥ 5 km²)
            secondary_cluster_sizes = secondary_cluster_sizes[
            secondary_cluster_sizes >= MIN_SECONDARY_CLUSTER_PIXELS
]
            
            # --- Mean size of the 10 largest secondary clusters ---
            if secondary_cluster_sizes.size > 0:
                # Sort descending
                secondary_cluster_sizes_sorted = np.sort(secondary_cluster_sizes)[::-1]

                # Take up to 10 largest
                top_10_secondary = secondary_cluster_sizes_sorted[:10]

                mean_top10_secondary_pixels = top_10_secondary.mean()
                mean_top10_secondary_km2 = mean_top10_secondary_pixels * pixel_area_km2
            else:
                mean_top10_secondary_pixels = 0.0
                mean_top10_secondary_km2 = 0.0


            if secondary_cluster_sizes.size > 0:
                mean_secondary_cluster_pixels = secondary_cluster_sizes.mean()
                mean_secondary_cluster_km2 = mean_secondary_cluster_pixels * pixel_area_km2
            else:
                mean_secondary_cluster_pixels = 0.0
                mean_secondary_cluster_km2 = 0.0

        else:
            mean_secondary_cluster_pixels = 0.0
            mean_secondary_cluster_km2 = 0.0
        
        print(f"  1985 LCC size: {lcc_1985_in_region_size:,} pixels")
        print(f"  1985 LCC area: {lcc_1985_in_region_area_km2:.3f} km²")
        print(f"  Mean secondary cluster size: "
              f"{mean_secondary_cluster_pixels:.1f} pixels "
              f"({mean_secondary_cluster_km2:.4f} km²)")
        

                # --- Histogram of secondary cluster sizes ---
    if num_clusters_1985 > 1:
        secondary_cluster_sizes_pixels = secondary_cluster_sizes
        secondary_cluster_sizes_km2 = secondary_cluster_sizes_pixels * pixel_area_km2

        plt.figure(figsize=(8, 5))
        plt.hist(
            secondary_cluster_sizes_km2,
            bins=30,
            edgecolor="black",
            log=True
        )

        plt.xlabel("Cluster area (km²)")
        plt.ylabel("Frequency (log scale)")
        plt.title(
            f"1985 Secondary Urban Cluster Size Distribution\n"
            f"{city_name} (within 2015 LCC)"
        )

        hist_file = output_path / f"{city_name.replace(' ', '_')}_secondary_cluster_histogram.png"
        plt.tight_layout()
        plt.savefig(hist_file, dpi=200)
        plt.close()

        print(f"  ✓ Saved cluster size histogram to: {hist_file}")

    
    # Step 7: Calculate secondary urbanization metrics
    print(f"\n{'='*70}")
    print(f"STEP 7: Calculating Secondary Urbanization")
    print(f"{'='*70}")
    
    secondary_urbanization_pixels = pixels_1985_in_2015_lcc - lcc_1985_in_region_size
    secondary_urbanization_km2 = area_1985_in_2015_lcc_km2 - lcc_1985_in_region_area_km2
    
    if area_1985_in_2015_lcc_km2 > 0:
        secondary_pct = (secondary_urbanization_km2 / area_1985_in_2015_lcc_km2) * 100
    else:
        secondary_pct = 0.0
    
    if lcc_1985_in_region_area_km2 > 0:
        secondary_relative = (secondary_urbanization_km2 / lcc_1985_in_region_area_km2) * 100
    else:
        secondary_relative = 0.0
    
    if lcc_2015_area_km2 > 0:
        secondary_to_2015_lcc_ratio = secondary_urbanization_km2 / lcc_2015_area_km2
    else:
        secondary_to_2015_lcc_ratio = 0.0
    
    print(f"  Secondary urbanization area: {secondary_urbanization_km2:.3f} km²")
    print(f"  Secondary / 2015 LCC area:   {secondary_to_2015_lcc_ratio:.4f}")
    
    # Step 8: Save results
    print(f"\n{'='*70}")
    print(f"STEP 8: Saving Results")
    print(f"{'='*70}")
    
    results = {
        'success': True,
        'city': city_name,
        'center_lat': center_lat,
        'center_lon': center_lon,
        'radius_km': radius_km,
        
        # 2015 metrics
        'lcc_2015_area_km2': round(lcc_2015_area_km2, 3),
        'lcc_2015_pixels': int(lcc_2015_size),
        
        # 1985 metrics
        'urbanized_1985_in_2015_lcc_km2': round(area_1985_in_2015_lcc_km2, 3),
        'urbanized_1985_in_2015_lcc_pixels': int(pixels_1985_in_2015_lcc),
        'lcc_1985_in_2015_lcc_km2': round(lcc_1985_in_region_area_km2, 3),
        'lcc_1985_in_2015_lcc_pixels': int(lcc_1985_in_region_size),
        'num_clusters_1985': int(num_clusters_1985),
        
        # Secondary urbanization
        'secondary_urbanization_km2': round(secondary_urbanization_km2, 3),
        'secondary_urbanization_pixels': int(secondary_urbanization_pixels),
        'secondary_urbanization_pct': round(secondary_pct, 2),
        'secondary_relative_to_lcc': round(secondary_relative, 2),
        
        # New metrics
        'secondary_to_2015_lcc_ratio': round(secondary_to_2015_lcc_ratio, 4),
        'mean_secondary_cluster_km2': round(mean_secondary_cluster_km2, 4),
        'mean_secondary_cluster_pixels': round(mean_secondary_cluster_pixels, 1),
        'mean_top10_secondary_cluster_km2': round(mean_top10_secondary_km2, 3),
        'mean_top10_secondary_cluster_pixels': round(mean_top10_secondary_pixels, 1)

    }
    
    import json
    output_file = output_path / f"{city_name.replace(' ', '_')}_growth_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  ✓ Saved results to: {output_file}")
    print(f"\n{'='*70}")
    print(f"ANALYSIS COMPLETE!")
    print(f"{'='*70}\n")
    
    return results
