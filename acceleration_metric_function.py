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


def analyze_urban_growth(city_name: str, 
                        radius_km: float = 50,
                        output_dir: str = "./output") -> dict:
    """
    Analyze urban growth by comparing 1985 urbanization to LCC.
    
    Parameters
    ----------
    city_name : str
        Name of the city to analyze
    radius_km : float
        Radius for data download (default: 50km)
    output_dir : str
        Directory for outputs
        
    Returns
    -------
    dict
        Analysis results with areas and metrics
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
    
    pixel_area_km2 = 0.03 * 0.03  # 30m × 30m pixels
    
    # Step 4: Extract 2015 LCC boundary
    print(f"\n{'='*70}")
    print(f"STEP 4: Extracting 2015 LCC Boundary")
    print(f"{'='*70}")
    
    mask_2015 = analyzer.extract_year_mask(wsf_data, 2015)
    print(f"  Total urbanized pixels in 2015: {mask_2015.sum():,}")
    
    lcc_2015_mask, lcc_2015_size = analyzer.find_largest_connected_component(mask_2015)
    lcc_2015_area_km2 = lcc_2015_size * pixel_area_km2
    
    print(f"  2015 LCC size: {lcc_2015_size:,} pixels")
    print(f"  2015 LCC area: {lcc_2015_area_km2:.3f} km²")
    
    if lcc_2015_size == 0:
        print("\n⚠️  ERROR: No LCC found in 2015!")
        return {
            'success': False,
            'error': 'No LCC found in 2015',
            'city': city_name
        }
    
    # Step 5: Extract 1985 urbanization within 2015 LCC boundary
    print(f"\n{'='*70}")
    print(f"STEP 5: Analyzing 1985 Urbanization within 2015 LCC Boundary")
    print(f"{'='*70}")
    
    # Get all urbanized areas in 1985
    mask_1985 = analyzer.extract_year_mask(wsf_data, 1985)
    print(f"  Total urbanized pixels in 1985 (entire region): {mask_1985.sum():,}")
    
    # Intersect with 2015 LCC boundary
    mask_1985_in_2015_lcc = mask_1985 & lcc_2015_mask
    pixels_1985_in_2015_lcc = mask_1985_in_2015_lcc.sum()
    area_1985_in_2015_lcc_km2 = pixels_1985_in_2015_lcc * pixel_area_km2
    
    print(f"  1985 urbanized pixels within 2015 LCC boundary: {pixels_1985_in_2015_lcc:,}")
    print(f"  1985 urbanized area within 2015 LCC boundary: {area_1985_in_2015_lcc_km2:.3f} km²")
    
    # Step 6: Find 1985 LCC and count all clusters within the 2015 LCC boundary
    print(f"\n{'='*70}")
    print(f"STEP 6: Finding 1985 LCC and Counting Clusters within 2015 LCC Boundary")
    print(f"{'='*70}")
    
    # Find connected components in the 1985 urbanization within 2015 LCC
    from scipy import ndimage
    
    # Label all connected components
    labeled_1985, num_clusters_1985 = ndimage.label(mask_1985_in_2015_lcc)
    
    print(f"  Total number of urbanized clusters in 1985: {num_clusters_1985:,}")
    
    if num_clusters_1985 == 0:
        print("  ⚠️  No urbanized areas found in 1985 within 2015 LCC boundary")
        lcc_1985_in_region_size = 0
        lcc_1985_in_region_area_km2 = 0.0
    else:
        # Get the largest connected component
        component_sizes = np.bincount(labeled_1985.ravel())[1:]  # Skip background (0)
        largest_label = component_sizes.argmax() + 1
        lcc_1985_in_region_size = int(component_sizes[largest_label - 1])
        lcc_1985_in_region_area_km2 = lcc_1985_in_region_size * pixel_area_km2
        
        print(f"  1985 LCC size (within 2015 LCC boundary): {lcc_1985_in_region_size:,} pixels")
        print(f"  1985 LCC area (within 2015 LCC boundary): {lcc_1985_in_region_area_km2:.3f} km²")
        
        # Additional cluster statistics
        if num_clusters_1985 > 1:
            sorted_sizes = np.sort(component_sizes)[::-1]
            print(f"\n  Cluster size distribution:")
            print(f"    Largest (LCC): {sorted_sizes[0]:,} pixels ({sorted_sizes[0]*pixel_area_km2:.3f} km²)")
            if len(sorted_sizes) > 1:
                print(f"    2nd largest: {sorted_sizes[1]:,} pixels ({sorted_sizes[1]*pixel_area_km2:.3f} km²)")
            if len(sorted_sizes) > 2:
                print(f"    3rd largest: {sorted_sizes[2]:,} pixels ({sorted_sizes[2]*pixel_area_km2:.3f} km²)")
            if len(sorted_sizes) > 3:
                print(f"    Median size: {int(np.median(sorted_sizes)):,} pixels")
                print(f"    Smallest: {sorted_sizes[-1]:,} pixels ({sorted_sizes[-1]*pixel_area_km2:.6f} km²)")
    
    # Step 7: Calculate difference
    print(f"\n{'='*70}")
    print(f"STEP 7: Calculating Secondary Urbanization")
    print(f"{'='*70}")
    
    secondary_urbanization_pixels = pixels_1985_in_2015_lcc - lcc_1985_in_region_size
    secondary_urbanization_km2 = area_1985_in_2015_lcc_km2 - lcc_1985_in_region_area_km2
    
    print(f"\n  Result:")
    print(f"  {'='*60}")
    print(f"  1985 urbanized area (in 2015 LCC): {area_1985_in_2015_lcc_km2:.3f} km²")
    print(f"  1985 LCC area (in 2015 LCC):       {lcc_1985_in_region_area_km2:.3f} km²")
    print(f"  {'─'*60}")
    print(f"  Secondary urbanization:            {secondary_urbanization_km2:.3f} km²")
    print(f"  Number of urbanized clusters:      {num_clusters_1985:,}")
    print(f"  {'='*60}")
    
    # Calculate percentage
    if area_1985_in_2015_lcc_km2 > 0:
        secondary_pct = (secondary_urbanization_km2 / area_1985_in_2015_lcc_km2) * 100
        print(f"  Secondary as % of 1985 urban:      {secondary_pct:.2f}%")
    
    if lcc_1985_in_region_area_km2 > 0:
        secondary_relative = (secondary_urbanization_km2 / lcc_1985_in_region_area_km2) * 100
        print(f"  Secondary relative to 1985 LCC:    {secondary_relative:.2f}%")
    
    print(f"\n  Interpretation:")
    print(f"  In 1985, within what would become the 2015 LCC boundary,")
    print(f"  there were {num_clusters_1985:,} separate urbanized clusters,")
    print(f"  with {secondary_urbanization_km2:.3f} km² of urbanized areas")
    print(f"  that were NOT part of the main connected component.")
    print(f"  This represents scattered/fragmented urbanization.")
    
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
        
        # 1985 metrics (within 2015 LCC boundary)
        'urbanized_1985_in_2015_lcc_km2': round(area_1985_in_2015_lcc_km2, 3),
        'urbanized_1985_in_2015_lcc_pixels': int(pixels_1985_in_2015_lcc),
        'lcc_1985_in_2015_lcc_km2': round(lcc_1985_in_region_area_km2, 3),
        'lcc_1985_in_2015_lcc_pixels': int(lcc_1985_in_region_size),
        'num_clusters_1985': int(num_clusters_1985),
        
        # Secondary urbanization
        'secondary_urbanization_km2': round(secondary_urbanization_km2, 3),
        'secondary_urbanization_pixels': int(secondary_urbanization_pixels),
        'secondary_urbanization_pct': round(secondary_pct, 2) if area_1985_in_2015_lcc_km2 > 0 else 0.0,
        'secondary_relative_to_lcc': round(secondary_relative, 2) if lcc_1985_in_region_area_km2 > 0 else 0.0
    }
    
    # Save to file
    import json
    output_file = output_path / f"{city_name.replace(' ', '_')}_growth_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  ✓ Saved results to: {output_file}")
    
    print(f"\n{'='*70}")
    print(f"ANALYSIS COMPLETE!")
    print(f"{'='*70}\n")
    
    return results


if __name__ == "__main__":
    import sys
    
    # Get city name from command line or use default
    if len(sys.argv) > 1:
        city = sys.argv[1]
    else:
        city = "Zurich, Switzerland"
    
    # Optional: radius parameter
    if len(sys.argv) > 2:
        radius = float(sys.argv[2])
    else:
        radius = 50  # km
    
    print(f"\n{'#'*70}")
    print(f"# Urban Growth Analysis")
    print(f"# City: {city}")
    print(f"# Radius: {radius} km")
    print(f"{'#'*70}\n")
    
    results = analyze_urban_growth(city, radius_km=radius)
    
    if results['success']:
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"City: {results['city']}")
        print(f"\nSecondary urbanization in 1985:")
        print(f"  {results['secondary_urbanization_km2']} km²")
        print(f"  ({results['secondary_urbanization_pct']}% of 1985 urban area)")
        print(f"\nUrban fragmentation in 1985:")
        print(f"  {results['num_clusters_1985']} separate urbanized clusters")
        print("="*70 + "\n")
    else:
        print(f"\n⚠️  Analysis failed: {results.get('error', 'Unknown error')}\n")