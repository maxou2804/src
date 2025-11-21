#!/usr/bin/env python3
"""
Urban Analysis Library - Optimized Functions with Persistent Cluster IDs
=========================================================================

High-performance library for analyzing urban evolution from WSF data.
Uses Numba JIT compilation for 5-10x performance improvements.

NEW: Persistent cluster ID system - IDs are never reused. When clusters disappear,
their IDs are retired. New clusters get sequential IDs (e.g., 10, 11, 12...).

Author: Urban Analytics  
Version: 2.1 (Optimized + Persistent IDs)
"""

import numpy as np
import pandas as pd
import requests
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from scipy import ndimage
import rasterio
from rasterio.merge import merge
from geopy.geocoders import Nominatim
import time

# Try to import Numba for optimization
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("⚠️  Warning: Numba not available. Install with: pip install numba")
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


# ============================================================================
# NUMBA-OPTIMIZED FUNCTIONS
# ============================================================================

@jit(nopython=True, cache=True)
def calculate_overlap_from_labeled_numba(labeled1, labeled2, label1_id, label2_id):
    """Fast overlap calculation between two labeled regions (~10× faster)."""
    intersection = 0
    size1 = 0
    size2 = 0
    
    height, width = labeled1.shape
    
    for i in range(height):
        for j in range(width):
            is_label1 = (labeled1[i, j] == label1_id)
            is_label2 = (labeled2[i, j] == label2_id)
            
            if is_label1:
                size1 += 1
            if is_label2:
                size2 += 1
            if is_label1 and is_label2:
                intersection += 1
    
    return intersection, size1, size2


@jit(nopython=True, cache=True)
def calculate_centroids_batch_numba(labeled_array, label_ids):
    """
    Calculate centroids for multiple labels at once (~5× faster).
    Returns: Array of shape (n_labels, 2) with [row, col] centroids
    """
    n_labels = len(label_ids)
    centroids = np.zeros((n_labels, 2), dtype=np.float64)
    counts = np.zeros(n_labels, dtype=np.int64)
    
    height, width = labeled_array.shape
    
    for i in range(height):
        for j in range(width):
            pixel_label = labeled_array[i, j]
            
            for label_idx in range(n_labels):
                if pixel_label == label_ids[label_idx]:
                    centroids[label_idx, 0] += i
                    centroids[label_idx, 1] += j
                    counts[label_idx] += 1
                    break
    
    for label_idx in range(n_labels):
        if counts[label_idx] > 0:
            centroids[label_idx, 0] /= counts[label_idx]
            centroids[label_idx, 1] /= counts[label_idx]
        else:
            centroids[label_idx, 0] = -1
            centroids[label_idx, 1] = -1
    
    return centroids


@jit(nopython=True, cache=True)
def calculate_radial_distances_numba(centroids, lcc_centroid):
    """Calculate radial distances from LCC centroid for all clusters."""
    n = centroids.shape[0]
    distances = np.zeros(n, dtype=np.float64)
    
    lcc_row, lcc_col = lcc_centroid
    
    for i in range(n):
        if centroids[i, 0] >= 0:
            dr = centroids[i, 0] - lcc_row
            dc = centroids[i, 1] - lcc_col
            distances[i] = np.sqrt(dr * dr + dc * dc)
        else:
            distances[i] = -1
    
    return distances


@jit(nopython=True, cache=True)
def match_clusters_fast_numba(labeled_current, labeled_previous, 
                               current_label_ids, previous_label_ids,
                               overlap_threshold=0.3):
    """
    Fast cluster matching using labeled arrays (~8× faster).
    Returns: Array mapping current label indices to previous indices (-1 if no match)
    """
    n_current = len(current_label_ids)
    n_previous = len(previous_label_ids)
    matches = np.full(n_current, -1, dtype=np.int32)
    
    height, width = labeled_current.shape
    
    for curr_idx in range(n_current):
        curr_label = current_label_ids[curr_idx]
        best_overlap = 0.0
        best_prev_idx = -1
        
        for prev_idx in range(n_previous):
            prev_label = previous_label_ids[prev_idx]
            
            # Calculate overlap
            intersection, size_curr, size_prev = calculate_overlap_from_labeled_numba(
                labeled_current, labeled_previous, curr_label, prev_label
            )
            
            if size_curr > 0:
                overlap_ratio = intersection / size_curr
                
                if overlap_ratio > overlap_threshold and overlap_ratio > best_overlap:
                    best_overlap = overlap_ratio
                    best_prev_idx = prev_idx
        
        if best_prev_idx >= 0:
            matches[curr_idx] = best_prev_idx
    
    return matches


@jit(nopython=True, cache=True)
def detect_mergers_fast_numba(lcc_current, lcc_previous, labeled_previous, 
                               prev_label_ids, overlap_threshold=0.5):
    """
    Fast merger detection (~8× faster).
    Returns: Array of merged label indices
    """
    height, width = lcc_current.shape
    n_prev = len(prev_label_ids)
    overlaps = np.zeros(n_prev, dtype=np.int64)
    prev_sizes = np.zeros(n_prev, dtype=np.int64)
    
    # Count sizes and overlaps
    for i in range(height):
        for j in range(width):
            if labeled_previous[i, j] > 0:
                for k in range(n_prev):
                    if labeled_previous[i, j] == prev_label_ids[k]:
                        prev_sizes[k] += 1
                        if lcc_current[i, j] == 1 and lcc_previous[i, j] == 0:
                            overlaps[k] += 1
                        break
    
    # Find merged clusters
    merged_list = []
    for k in range(n_prev):
        if prev_sizes[k] > 0:
            overlap_ratio = overlaps[k] / prev_sizes[k]
            if overlap_ratio >= overlap_threshold:
                merged_list.append(k)
    
    return np.array(merged_list, dtype=np.int32)


@jit(nopython=True, cache=True)
def coarse_grain_mask_numba(mask, factor):
    """
    Coarse grain a mask by taking the most common label in each block.
    Much faster than the pure Python version (~10-20x speedup).
    
    Hard limit: Supports labels 0-100 only. Higher labels treated as 0.
    
    Args:
        mask: Input mask array (labels should be 0-100)
        factor: Coarse graining factor (e.g., 2 for 2x2 blocks)
    
    Returns:
        Coarse-grained mask
    """
    height, width = mask.shape
    new_height = height // factor
    new_width = width // factor
    
    coarse_mask = np.zeros((new_height, new_width), dtype=np.uint32)
    
    for i in range(new_height):
        for j in range(new_width):
            # Count labels in this block
            max_count = 0
            most_common = 0
            
            # Fixed array for labels 0-100 (efficient allocation)
            label_counts = np.zeros(101, dtype=np.int32)
            
            # Count labels in block
            for di in range(factor):
                for dj in range(factor):
                    row = i * factor + di
                    col = j * factor + dj
                    if row < height and col < width:
                        label = mask[row, col]
                        # Only count labels 0-100
                        if label <= 100:
                            label_counts[label] += 1
            
            # Find most common label
            for label in range(101):
                if label_counts[label] > max_count:
                    max_count = label_counts[label]
                    most_common = label
            
            coarse_mask[i, j] = most_common
    
    return coarse_mask


@jit(nopython=True, cache=True)
def extract_year_mask_numba(wsf_data, year):
    """Extract binary mask for a specific year (~2× faster)."""
    mask = np.zeros(wsf_data.shape, dtype=np.uint8)
    height, width = wsf_data.shape
    
    for i in range(height):
        for j in range(width):
            if wsf_data[i, j] > 0 and wsf_data[i, j] <= year:
                mask[i, j] = 1
    
    return mask


@jit(nopython=True, cache=True)
def create_circular_mask_numba(height, width, center_row, center_col, radius):
    """Create circular mask (~5-10× faster)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            dist = np.sqrt((i - center_row)**2 + (j - center_col)**2)
            if dist <= radius:
                mask[i, j] = 1
    
    return mask


@jit(nopython=True, cache=True)
def create_rgb_from_labeled_array_numba(labeled_array, cluster_label_ids, 
                                        cluster_colors):
    """
    Create RGB image from labeled array with specific colors.
    
    Args:
        labeled_array: 2D array with cluster labels
        cluster_label_ids: Array of label IDs to colorize
        cluster_colors: Array of shape (n_clusters, 3) with RGB colors
    
    Returns:
        RGB array of shape (height, width, 3)
    """
    height, width = labeled_array.shape
    n_clusters = len(cluster_label_ids)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Set background built areas to gray
    for i in range(height):
        for j in range(width):
            if labeled_array[i, j] == 0:
                rgb[i, j, 0] = 128
                rgb[i, j, 1] = 128
                rgb[i, j, 2] = 128
    
    # Color each cluster
    for cluster_idx in range(n_clusters):
        label_id = cluster_label_ids[cluster_idx]
        color = cluster_colors[cluster_idx]
        
        for i in range(height):
            for j in range(width):
                if labeled_array[i, j] == label_id:
                    rgb[i, j, 0] = color[0]
                    rgb[i, j, 1] = color[1]
                    rgb[i, j, 2] = color[2]
    
    return rgb


# ============================================================================
# DATA DOWNLOAD AND MANAGEMENT
# ============================================================================

class WSFTileManager:
    """Manage downloading multiple WSF Evolution tiles for a region."""
    
    BASE_URL = "https://download.geoservice.dlr.de/WSF_EVO/files/"
    TILE_SIZE_DEGREES = 2  # Tiles are 2° × 2°
    
    def __init__(self, cache_dir: str = "./wsf_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_tile_name(self, lat: float, lon: float) -> str:
        """Get tile name for a specific lat/lon."""
        lat_tile = int(np.floor(lat / self.TILE_SIZE_DEGREES) * self.TILE_SIZE_DEGREES)
        lon_tile = int(np.floor(lon / self.TILE_SIZE_DEGREES) * self.TILE_SIZE_DEGREES)
        return f"WSFevolution_v1_{lon_tile}_{lat_tile}.tif"
    
    def calculate_required_tiles(self, center_lat: float, center_lon: float, 
                                 radius_km: float) -> List[Tuple[str, dict]]:
        """Calculate all tiles needed to cover a circular region."""
        lat_radius_deg = radius_km / 111.0
        lon_radius_deg = radius_km / (111.0 * np.cos(np.radians(center_lat)))
        
        min_lat = center_lat - lat_radius_deg
        max_lat = center_lat + lat_radius_deg
        min_lon = center_lon - lon_radius_deg
        max_lon = center_lon + lon_radius_deg
        
        tiles = []
        
        min_lat_tile = int(np.floor(min_lat / self.TILE_SIZE_DEGREES) * self.TILE_SIZE_DEGREES)
        max_lat_tile = int(np.floor(max_lat / self.TILE_SIZE_DEGREES) * self.TILE_SIZE_DEGREES)
        min_lon_tile = int(np.floor(min_lon / self.TILE_SIZE_DEGREES) * self.TILE_SIZE_DEGREES)
        max_lon_tile = int(np.floor(max_lon / self.TILE_SIZE_DEGREES) * self.TILE_SIZE_DEGREES)
        
        lat_tile = min_lat_tile
        while lat_tile <= max_lat_tile:
            lon_tile = min_lon_tile
            while lon_tile <= max_lon_tile:
                tile_name = f"WSFevolution_v1_{lon_tile}_{lat_tile}.tif"
                tile_info = {
                    'lat_tile': lat_tile,
                    'lon_tile': lon_tile,
                    'bounds': (lat_tile, lat_tile + self.TILE_SIZE_DEGREES,
                              lon_tile, lon_tile + self.TILE_SIZE_DEGREES)
                }
                tiles.append((tile_name, tile_info))
                lon_tile += self.TILE_SIZE_DEGREES
            lat_tile += self.TILE_SIZE_DEGREES
        
        return tiles
    
    def download_tile(self, tile_name: str, force_redownload: bool = False) -> Tuple[bool, Path]:
        """Download a tile if not already cached."""
        tile_path = self.cache_dir / tile_name
        
        if tile_path.exists() and not force_redownload:
            print(f"  ✓ Using cached: {tile_name}")
            return True, tile_path
        
        url = self.BASE_URL + tile_name
        
        try:
            print(f"  ⬇ Downloading: {tile_name}")
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(tile_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"    Progress: {percent:.1f}%", end='\r', flush=True)
            
            print(f"\n  ✓ Downloaded: {tile_name} ({total_size/(1024*1024):.1f} MB)")
            return True, tile_path
            
        except requests.exceptions.RequestException as e:
            print(f"  ✗ Failed to download {tile_name}: {e}")
            if tile_path.exists():
                tile_path.unlink()
            return False, None
    
    def download_region(self, center_lat: float, center_lon: float, 
                       radius_km: float) -> Dict:
        """Download all tiles needed to cover a region."""
        tiles = self.calculate_required_tiles(center_lat, center_lon, radius_km)
        
        print(f"\n{'='*70}")
        print(f"Downloading WSF Evolution data")
        print(f"{'='*70}")
        print(f"Center: {center_lat:.4f}°, {center_lon:.4f}°")
        print(f"Radius: {radius_km} km")
        print(f"Tiles required: {len(tiles)}")
        print(f"{'='*70}\n")
        
        results = {
            'center': (center_lat, center_lon),
            'radius_km': radius_km,
            'tiles': [],
            'successful': [],
            'failed': []
        }
        
        for tile_name, tile_info in tiles:
            success, tile_path = self.download_tile(tile_name)
            
            results['tiles'].append({
                'name': tile_name,
                'info': tile_info,
                'success': success,
                'path': tile_path
            })
            
            if success:
                results['successful'].append(tile_name)
            else:
                results['failed'].append(tile_name)
        
        print(f"\n{'='*70}")
        print(f"✓ Download complete!")
        print(f"  Successful: {len(results['successful'])}/{len(tiles)}")
        if results['failed']:
            print(f"  Failed: {len(results['failed'])}")
        print(f"{'='*70}\n")
        
        return results


# ============================================================================
# BUILT AREA ANALYSIS
# ============================================================================

class BuiltAreaAnalyzer:
    """Analyze urban built areas from WSF Evolution data."""
    
    def __init__(self, use_numba: bool = NUMBA_AVAILABLE):
        self.years = list(range(1985, 2016))
        self.pixel_area_km2 = (30 * 30) / 1e6  # 30m × 30m pixels
        self.pixel_size_km = 0.03
        self.use_numba = use_numba and NUMBA_AVAILABLE
        
        if not self.use_numba and NUMBA_AVAILABLE:
            print("⚠️  Numba disabled by user")
        elif not NUMBA_AVAILABLE:
            print("⚠️  Numba not available - install with: pip install numba")
    
    def load_tiles_from_download_result(self, download_result: Dict) -> Tuple[np.ndarray, Dict]:
        """Load and merge tiles from download results."""
        successful_tiles = []
        
        for tile_data in download_result['tiles']:
            if tile_data['success'] and tile_data['path']:
                try:
                    src = rasterio.open(tile_data['path'])
                    successful_tiles.append(src)
                except Exception as e:
                    print(f"Warning: Could not open {tile_data['name']}: {e}")
        
        if not successful_tiles:
            raise ValueError("No tiles could be loaded")
        
        if len(successful_tiles) == 1:
            print("Loading single tile...")
            data = successful_tiles[0].read(1)
            metadata = {
                'transform': successful_tiles[0].transform,
                'crs': successful_tiles[0].crs,
                'shape': data.shape
            }
        else:
            print(f"Merging {len(successful_tiles)} tiles...")
            mosaic, transform = merge(successful_tiles)
            data = mosaic[0]
            metadata = {
                'transform': transform,
                'crs': successful_tiles[0].crs,
                'shape': data.shape
            }
        
        for src in successful_tiles:
            src.close()
        
        print(f"✓ Loaded WSF data: {data.shape} ({data.nbytes / (1024*1024):.1f} MB)")
        
        return data, metadata
    def extract_built_area_bbox(self, data: np.ndarray, 
                                transform: rasterio.Affine,
                                center_lat: float, 
                                center_lon: float,
                                size_km: float = 10) -> Tuple[np.ndarray, dict]:
        """
        Extract a bounding box around a location from mosaicked data.
        
        Args:
            data: Mosaicked WSF Evolution array
            transform: Affine transform from the mosaic
            center_lat: Latitude of center point
            center_lon: Longitude of center point
            size_km: Size of bounding box in kilometers (default 10km = 10km x 10km)
        
        Returns:
            Array with WSF data and metadata dictionary
        """
        # Convert lat/lon to pixel coordinates using the transform
        from rasterio.transform import rowcol
        row, col = rowcol(transform, center_lon, center_lat)
        
        # Check if center point is within data bounds
        if row < 0 or row >= data.shape[0] or col < 0 or col >= data.shape[1]:
            print(f"\n⚠️  Warning: Center point ({center_lat}, {center_lon}) is outside data bounds!")
            print(f"  Center pixel: ({row}, {col})")
            print(f"  Data shape: {data.shape}")
            print(f"  Adjusting to use full available data...")
            
            # Use the full data instead
            return data, {
                'transform': transform,
                'width': data.shape[1],
                'height': data.shape[0],
                'center_lat': center_lat,
                'center_lon': center_lon,
                'size_km': size_km,
                'adjusted': True
            }
        
        # Calculate pixel extent (approximately)
        # At 30m resolution: 1km ≈ 33.33 pixels
        pixels_per_km = 1000 / 30  # 30m pixel size
        half_size_pixels = int((size_km / 2) * pixels_per_km)
        
        # Define extraction bounds with clipping
        row_start = max(0, row - half_size_pixels)
        row_end = min(data.shape[0], row + half_size_pixels)
        col_start = max(0, col - half_size_pixels)
        col_end = min(data.shape[1], col + half_size_pixels)
        
        # Validate bounds
        if row_start >= row_end or col_start >= col_end:
            print(f"\n⚠️  Warning: Invalid extraction bounds!")
            print(f"  Requested size: {size_km} km × {size_km} km")
            print(f"  Row range: {row_start} to {row_end} (height: {row_end - row_start})")
            print(f"  Col range: {col_start} to {col_end} (width: {col_end - col_start})")
            print(f"  Using full data instead...")
            
            return data, {
                'transform': transform,
                'width': data.shape[1],
                'height': data.shape[0],
                'center_lat': center_lat,
                'center_lon': center_lon,
                'size_km': size_km,
                'adjusted': True
            }
        
        # Extract subset
        subset = data[row_start:row_end, col_start:col_end]
        
        # Additional validation
        if subset.shape[0] == 0 or subset.shape[1] == 0:
            print(f"\n⚠️  Warning: Extracted subset is empty!")
            print(f"  Using full data instead...")
            
            return data, {
                'transform': transform,
                'width': data.shape[1],
                'height': data.shape[0],
                'center_lat': center_lat,
                'center_lon': center_lon,
                'size_km': size_km,
                'adjusted': True
            }
        
        # Calculate new transform for the subset
        from rasterio.windows import transform as window_transform, Window
        window = Window(col_start, row_start, col_end - col_start, row_end - row_start)
        subset_transform = window_transform(window, transform)
        
        metadata = {
            'transform': subset_transform,
            'width': subset.shape[1],
            'height': subset.shape[0],
            'center_lat': center_lat,
            'center_lon': center_lon,
            'size_km': size_km,
            'adjusted': False
        }
        
        # Report if size was adjusted due to clipping
        actual_size_km = min(
            (row_end - row_start) * 30 / 1000,
            (col_end - col_start) * 30 / 1000
        )
        
        if abs(actual_size_km - size_km) > 1:  # More than 1km difference
            print(f"\n  Note: Requested {size_km}km × {size_km}km, got ~{actual_size_km:.1f}km × {actual_size_km:.1f}km")
            print(f"  (clipped to data bounds)")
        else:
            print(f"Extracted {size_km}km × {size_km}km region: {subset.shape}")
        
        return subset, metadata
    def extract_year_mask(self, wsf_data: np.ndarray, year: int) -> np.ndarray:
        """Extract binary mask of built areas up to and including a specific year."""
        if self.use_numba:
            return extract_year_mask_numba(wsf_data, year)
        else:
            mask = ((wsf_data > 0) & (wsf_data <= year)).astype(np.uint8)
            return mask
    
    def find_largest_connected_component(self, binary_mask: np.ndarray) -> Tuple[np.ndarray, int]:
        """Find the largest connected component in a binary mask."""
        if binary_mask.sum() == 0:
            return binary_mask, 0
        
        labeled_array, num_features = ndimage.label(binary_mask)
        
        if num_features == 0:
            return np.zeros_like(binary_mask), 0
        
        component_sizes = np.bincount(labeled_array.ravel())[1:]
        largest_label = component_sizes.argmax() + 1
        
        lcc_mask = (labeled_array == largest_label).astype(np.uint8)
        lcc_size = component_sizes[largest_label - 1]
        
        return lcc_mask, int(lcc_size)
    
    def calculate_lcc_centroid(self, lcc_mask: np.ndarray) -> Tuple[float, float]:
        """Calculate the centroid (center of mass) of the LCC."""
        if lcc_mask.sum() == 0:
            return None, None
        
        rows, cols = np.where(lcc_mask == 1)
        row_centroid = rows.mean()
        col_centroid = cols.mean()
        return row_centroid, col_centroid
    
    def calculate_mean_radius(self, lcc_mask: np.ndarray, 
                             centroid_row: float, 
                             centroid_col: float) -> float:
        """Calculate the mean radius of LCC from its centroid."""
        if lcc_mask.sum() == 0 or centroid_row is None:
            return 0.0
        
        rows, cols = np.where(lcc_mask == 1)
        distances = np.sqrt((rows - centroid_row)**2 + (cols - centroid_col)**2)
        return float(distances.mean())


# ============================================================================
# CLUSTER EVOLUTION TRACKER (OPTIMIZED + PERSISTENT IDS)
# ============================================================================

class ClusterEvolutionTracker:
    """
    Track urban cluster evolution over time with optimization and persistent IDs.
    
    NEW: Persistent cluster ID system
    - LCC is always ID 0
    - Initial clusters get IDs 1, 2, 3, ..., n_clusters-1
    - New clusters appearing later get IDs n_clusters, n_clusters+1, etc.
    - Disappeared cluster IDs are NEVER reused
    - Absorbed clusters are tracked in the 'absorbed_clusters' column
    
    Expected runtimes (31 years, with Numba):
    - Small images (1000×1000): ~30-60s
    - Medium images (3000×3000): ~60-90s
    - Large images (5000×5000): ~80-120s
    """
    
    def __init__(self, analyzer: BuiltAreaAnalyzer, n_clusters: int = 10, 
                 radius_factor: Optional[float] = None):
        """
        Initialize cluster evolution tracker.
        
        Args:
            analyzer: BuiltAreaAnalyzer instance
            n_clusters: Number of clusters to track initially (including LCC)
            radius_factor: Optional radius multiplier for search region.
                          If provided, only clusters within (LCC_mean_radius × radius_factor)
                          from LCC centroid are considered. None = no radius filtering (default).
                          Example: radius_factor=4.0 means search within 4× LCC mean radius.
        """
        self.analyzer = analyzer
        self.n_clusters = n_clusters
        self.radius_factor = radius_factor
        self.years = list(range(1985, 2016))
        self.pixel_area_km2 = (30 * 30) / 1e6
        self.pixel_size_km = 0.03
    
    def find_clusters_by_size_optimized(self, binary_mask: np.ndarray,
                                       n_clusters: int) -> Tuple[List, np.ndarray, np.ndarray, np.ndarray]:
        """
        Optimized cluster finding - returns labeled array and centroids in one pass.
        Returns: (clusters_list, labeled_array, label_ids, centroids)
        """
        if binary_mask.sum() == 0:
            return [], None, np.array([], dtype=np.int32), np.zeros((0, 2))
        
        labeled_array, num_features = ndimage.label(binary_mask)
        
        if num_features == 0:
            return [], labeled_array, np.array([], dtype=np.int32), np.zeros((0, 2))
        
        component_sizes = np.bincount(labeled_array.ravel())[1:]
        
        if len(component_sizes) == 0:
            return [], labeled_array, np.array([], dtype=np.int32), np.zeros((0, 2))
        
        sorted_indices = np.argsort(component_sizes)[::-1]
        n_to_extract = min(n_clusters, len(sorted_indices))
        
        label_ids = (sorted_indices[:n_to_extract] + 1).astype(np.int32)
        
        if NUMBA_AVAILABLE:
            centroids = calculate_centroids_batch_numba(labeled_array, label_ids)
        else:
            centroids = np.zeros((n_to_extract, 2))
            for i, label_id in enumerate(label_ids):
                rows, cols = np.where(labeled_array == label_id)
                if len(rows) > 0:
                    centroids[i] = [rows.mean(), cols.mean()]
                else:
                    centroids[i] = [-1, -1]
        
        clusters = []
        for i, idx in enumerate(sorted_indices[:n_to_extract]):
            label_id = int(label_ids[i])
            size = int(component_sizes[idx])
            centroid = tuple(centroids[i])
            clusters.append((i, label_id, size, centroid))
        
        return clusters, labeled_array, label_ids, centroids
    
    def track_evolution(self, wsf_data: np.ndarray) -> pd.DataFrame:
        """
        Track cluster evolution with full optimization and persistent IDs.
        
        Key features:
        1. Persistent cluster IDs that are never reused
        2. New clusters get sequential IDs (e.g., if starting with 0-9, new clusters become 10, 11, 12...)
        3. When a cluster disappears/merges, its ID is retired
        4. Cluster matching preserves identity across years
        
        Key optimizations:
        1. Reuse labeled arrays across operations
        2. Batch centroid calculation
        3. Numba-accelerated overlap detection
        4. Vectorized distance calculations
        5. Optional radius-based filtering
        """
        results = []
        
        # Persistent ID tracking
        next_cluster_id = 1  # Start at 1 (LCC is always 0)
        prev_cluster_persistent_ids = {}  # Maps match index to persistent ID
        
        prev_clusters = None
        prev_labeled = None
        prev_label_ids = None
        prev_centroids = None
        prev_lcc_mask = None
        prev_lcc_centroid = None
        
        mode = 'OPTIMIZED (Numba)' if NUMBA_AVAILABLE else 'STANDARD (no Numba)'
        radius_info = f" with radius_factor={self.radius_factor}" if self.radius_factor else ""
        print(f"\n{'='*70}")
        print(f"Tracking cluster evolution (1985-2015) - {mode}{radius_info}")
        print(f"With persistent cluster IDs (new clusters get IDs {self.n_clusters}+)")
        print(f"{'='*70}\n")
        
        total_start = time.time()
        # for year_idx,year in enumerate([1985,2015]):
        for year_idx, year in enumerate(self.years):
            year_start = time.time()
            
            built_mask = self.analyzer.extract_year_mask(wsf_data, year)
            
            if built_mask.sum() == 0:
                print(f"{year}: No built areas, skipping")
                continue
            
            lcc_mask, lcc_size = self.analyzer.find_largest_connected_component(built_mask)
            
            if lcc_size == 0:
                print(f"{year}: No LCC, skipping")
                continue
            
            lcc_rows, lcc_cols = np.where(lcc_mask == 1)
            lcc_centroid = (lcc_rows.mean(), lcc_cols.mean())
            lcc_area_km2 = lcc_size * self.pixel_area_km2
            
            # Calculate LCC mean radius if using radius filtering
            lcc_mean_radius_pixels = None
            search_radius_km = None
            if self.radius_factor is not None:
                lcc_mean_radius_pixels = self.analyzer.calculate_mean_radius(
                    lcc_mask, lcc_centroid[0], lcc_centroid[1]
                )
                search_radius_pixels = lcc_mean_radius_pixels * self.radius_factor
                search_radius_km = search_radius_pixels * self.pixel_size_km
            
            # Detect absorbed clusters (for LCC absorption tracking)
            absorbed_cluster_ids = []
            if prev_clusters is not None and prev_lcc_mask is not None and NUMBA_AVAILABLE:
                merged_indices = detect_mergers_fast_numba(
                    lcc_mask.astype(np.uint8), 
                    prev_lcc_mask.astype(np.uint8),
                    prev_labeled,
                    prev_label_ids,
                    overlap_threshold=0.5
                )
                # Convert to persistent IDs
                for idx in merged_indices:
                    if idx in prev_cluster_persistent_ids:
                        absorbed_cluster_ids.append(prev_cluster_persistent_ids[idx])
            
            # Record LCC
            results.append({
                'year': year,
                'cluster_id': 0,  # LCC is always ID 0
                'area_km2': round(lcc_area_km2, 3),
                'radial_distance_km': 0.0,
                'absorbed_clusters': ','.join(map(str, sorted(absorbed_cluster_ids))) if absorbed_cluster_ids else ''
            })
            
            non_lcc_mask = built_mask & (~lcc_mask)
            
            # Apply radius filtering if specified
            if self.radius_factor is not None and lcc_mean_radius_pixels is not None:
                # Create circular mask for filtering
                if NUMBA_AVAILABLE:
                    search_circle_mask = create_circular_mask_numba(
                        wsf_data.shape[0], wsf_data.shape[1],
                        lcc_centroid[0], lcc_centroid[1],
                        search_radius_pixels
                    )
                else:
                    rows, cols = np.ogrid[0:wsf_data.shape[0], 0:wsf_data.shape[1]]
                    distance_from_centroid = np.sqrt(
                        (rows - lcc_centroid[0])**2 + (cols - lcc_centroid[1])**2
                    )
                    search_circle_mask = (distance_from_centroid <= search_radius_pixels).astype(np.uint8)
                
                # Filter non-LCC mask to only include areas within search radius
                non_lcc_mask = non_lcc_mask & search_circle_mask
            
            other_clusters, labeled_array, label_ids, centroids = \
                self.find_clusters_by_size_optimized(non_lcc_mask, self.n_clusters - 1)
            
            # Match clusters to previous year
            if prev_clusters is not None and NUMBA_AVAILABLE and len(other_clusters) > 0 and len(prev_clusters) > 0:
                matches = match_clusters_fast_numba(
                    labeled_array, prev_labeled,
                    label_ids, prev_label_ids,
                    overlap_threshold=0.3
                )
            else:
                matches = np.full(len(other_clusters), -1, dtype=np.int32)
            
            # Assign persistent IDs to current clusters
            current_cluster_persistent_ids = {}
            
            if len(other_clusters) > 0:
                # Calculate distances
                if NUMBA_AVAILABLE:
                    distances = calculate_radial_distances_numba(centroids, lcc_centroid)
                    distances_km = distances * self.pixel_size_km
                else:
                    distances_km = []
                    for centroid in centroids:
                        if centroid[0] >= 0:
                            dist = np.sqrt((centroid[0] - lcc_centroid[0])**2 + 
                                         (centroid[1] - lcc_centroid[1])**2)
                            distances_km.append(dist * self.pixel_size_km)
                        else:
                            distances_km.append(np.nan)
                    distances_km = np.array(distances_km)
                
                # Assign IDs based on matching
                for idx, (_, _, size, _) in enumerate(other_clusters):
                    match_idx = matches[idx]
                    
                    if match_idx >= 0 and match_idx in prev_cluster_persistent_ids:
                        # Matched to previous cluster - preserve ID
                        persistent_id = prev_cluster_persistent_ids[match_idx]
                    else:
                        # New cluster - assign next available ID
                        persistent_id = next_cluster_id
                        next_cluster_id += 1
                    
                    current_cluster_persistent_ids[idx] = persistent_id
                    
                    area_km2 = size * self.pixel_area_km2
                    
                    results.append({
                        'year': year,
                        'cluster_id': persistent_id,
                        'area_km2': round(area_km2, 3),
                        'radial_distance_km': round(float(distances_km[idx]), 3) if distances_km[idx] >= 0 else np.nan,
                        'absorbed_clusters': ''
                    })
            
            prev_clusters = other_clusters
            prev_labeled = labeled_array
            prev_label_ids = label_ids
            prev_centroids = centroids
            prev_lcc_mask = lcc_mask
            prev_lcc_centroid = lcc_centroid
            prev_cluster_persistent_ids = current_cluster_persistent_ids
            
            year_time = time.time() - year_start
            
            if (year_idx + 1) % 5 == 0 or year_idx == 0:
                avg_time = (time.time() - total_start) / (year_idx + 1)
                remaining = avg_time * (len(self.years) - year_idx - 1)
                status_msg = f"{year}: {year_time:.2f}s | Clusters: {len(other_clusters)} | Next ID: {next_cluster_id}"
                if self.radius_factor is not None and search_radius_km is not None:
                    status_msg += f" | Search radius: {search_radius_km:.1f}km"
                print(status_msg)
        
        total_time = time.time() - total_start
        print(f"\n{'='*70}")
        print(f"✓ Complete! Total time: {total_time:.2f}s ({total_time/len(self.years):.2f}s/year avg)")
        print(f"  Total unique cluster IDs assigned: {next_cluster_id}")
        if self.radius_factor is not None:
            print(f"  Radius filtering: {self.radius_factor}× LCC mean radius")
        print(f"{'='*70}\n")
        
        return pd.DataFrame(results)
    
    def export_evolution_csv(self, wsf_data: np.ndarray, output_path: str) -> str:
        """Export evolution tracking to CSV."""
        df = self.track_evolution(wsf_data)
        
        output_path = Path(output_path)
        df.to_csv(output_path, index=False)
        
        print(f"✓ Saved: {output_path}")
        print(f"  Total records: {len(df):,}")
        print(f"  File size: {output_path.stat().st_size / 1024:.2f} KB")
        
        return str(output_path)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def geocode_city(city_name: str) -> Tuple[float, float]:
    """Geocode a city name to coordinates."""
    geolocator = Nominatim(user_agent="wsf_evolution_analyzer")
    
    try:
        location = geolocator.geocode(city_name, timeout=10)
        
        if location:
            print(f"✓ Found: {location.address}")
            print(f"  Coordinates: {location.latitude:.4f}°, {location.longitude:.4f}°")
            return location.latitude, location.longitude
        else:
            raise ValueError(f"Could not geocode city: {city_name}")
            
    except Exception as e:
        raise ValueError(f"Geocoding failed: {e}")


def print_system_info():
    """Print system information and optimization status."""
    print("\n" + "="*70)
    print("URBAN ANALYSIS LIBRARY - System Information")
    print("="*70)
    print(f"Numba available: {NUMBA_AVAILABLE}")
    if NUMBA_AVAILABLE:
        print("✓ Optimized functions enabled (5-10× speedup)")
    else:
        print("⚠️  Standard functions only (install numba for optimization)")
    print("="*70 + "\n")


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def extract_clusters_optimized(binary_mask: np.ndarray, 
                               n_clusters: int) -> Tuple[List, np.ndarray, int]:
    """
    Optimized cluster extraction - returns labeled array for efficient reuse.
    
    Args:
        binary_mask: Binary mask of areas to cluster
        n_clusters: Number of clusters to extract
    
    Returns:
        (clusters_list, labeled_array, num_features)
        clusters_list: List of (index, label_id, size) tuples
    """
    if binary_mask.sum() == 0:
        return [], None, 0
    
    # Label connected components
    labeled_array, num_features = ndimage.label(binary_mask)
    
    if num_features == 0:
        return [], labeled_array, 0
    
    # Get sizes and sort
    component_sizes = np.bincount(labeled_array.ravel())[1:]
    
    if len(component_sizes) == 0:
        return [], labeled_array, 0
    
    sorted_indices = np.argsort(component_sizes)[::-1]
    n_to_extract = min(n_clusters, len(sorted_indices))
    
    # Build clusters list
    clusters = []
    for i in range(n_to_extract):
        idx = sorted_indices[i]
        label_id = idx + 1
        size = int(component_sizes[idx])
        clusters.append((i, label_id, size))
    
    return clusters, labeled_array, num_features
def translate_mask_to_center_lcc(mask: np.ndarray, lcc_label: int = 1) -> Tuple[np.ndarray, Dict]:
    """
    Translate all urbanized areas to align the LCC centroid with the mask center.
    
    This is useful for preparing masks for continued growth simulations, as it centers
    the LCC in the domain and maximizes available space for expansion in all directions.
    
    Parameters
    ----------
    mask : np.ndarray
        Labeled mask where LCC has label `lcc_label`
    lcc_label : int, default=1
        Label value of the LCC in the mask
        
    Returns
    -------
    translated_mask : np.ndarray
        New mask with all areas translated so LCC is centered
    translation_info : dict
        Dictionary containing:
        - 'lcc_centroid_before': (row, col) of LCC centroid before translation
        - 'mask_center': (row, col) of mask center
        - 'translation_vector': (delta_row, delta_col) applied
        - 'lcc_centroid_after': (row, col) of LCC centroid after translation
        - 'pixels_lost': number of pixels that went out of bounds
        
    Notes
    -----
    - Pixels that would be translated outside the mask boundaries are lost (clipped)
    - This is intentional to avoid border effects in subsequent simulations
    - All labeled areas (not just LCC) are translated together
    """
    # Find LCC centroid
    lcc_coords = np.argwhere(mask == lcc_label)
    
    if len(lcc_coords) == 0:
        raise ValueError(f"No pixels found with label {lcc_label} (LCC)")
    
    lcc_centroid_row = np.mean(lcc_coords[:, 0])
    lcc_centroid_col = np.mean(lcc_coords[:, 1])
    
    # Calculate mask center
    mask_center_row = (mask.shape[0] - 1) / 2.0
    mask_center_col = (mask.shape[1] - 1) / 2.0
    
    # Calculate translation vector (how much to shift)
    delta_row = int(np.round(mask_center_row - lcc_centroid_row))
    delta_col = int(np.round(mask_center_col - lcc_centroid_col))
    
    print(f"  Translation vector: ({delta_row}, {delta_col})")
    print(f"  LCC centroid before: ({lcc_centroid_row:.1f}, {lcc_centroid_col:.1f})")
    print(f"  Mask center: ({mask_center_row:.1f}, {mask_center_col:.1f})")
    
    # Create new mask
    translated_mask = np.zeros_like(mask)
    
    # Get all urbanized pixels (non-zero labels)
    urbanized_coords = np.argwhere(mask > 0)
    original_count = len(urbanized_coords)
    
    # Translate each pixel
    pixels_kept = 0
    for coord in urbanized_coords:
        old_row, old_col = coord
        new_row = old_row + delta_row
        new_col = old_col + delta_col
        
        # Check if new position is within bounds
        if (0 <= new_row < mask.shape[0]) and (0 <= new_col < mask.shape[1]):
            translated_mask[new_row, new_col] = mask[old_row, old_col]
            pixels_kept += 1
    
    pixels_lost = original_count - pixels_kept
    
    # Verify new LCC centroid
    new_lcc_coords = np.argwhere(translated_mask == lcc_label)
    new_lcc_centroid_row = np.mean(new_lcc_coords[:, 0])
    new_lcc_centroid_col = np.mean(new_lcc_coords[:, 1])
    
    print(f"  LCC centroid after: ({new_lcc_centroid_row:.1f}, {new_lcc_centroid_col:.1f})")
    print(f"  Pixels kept: {pixels_kept:,} / {original_count:,}")
    print(f"  Pixels lost (out of bounds): {pixels_lost:,}")
    
    # Prepare info dictionary
    translation_info = {
        'lcc_centroid_before': (lcc_centroid_row, lcc_centroid_col),
        'mask_center': (mask_center_row, mask_center_col),
        'translation_vector': (delta_row, delta_col),
        'lcc_centroid_after': (new_lcc_centroid_row, new_lcc_centroid_col),
        'pixels_lost': pixels_lost,
        'pixels_kept': pixels_kept
    }
    
    return translated_mask, translation_info




def extract_lcc_region(wsf_data: np.ndarray, year: int, region_size: int):
    """
    Extract a square region around the LCC centroid.
    
    This is a shared helper function used by both mask extraction functions
    to ensure consistent region extraction.
    
    Parameters
    ----------
    wsf_data : np.ndarray
        WSF Evolution data
    year : int
        Year to extract
    region_size : int
        Size of square region in pixels
        
    Returns
    -------
    region_mask : np.ndarray
        Extracted region (region_size × region_size)
    extraction_info : dict
        Information about the extraction
    """
    # Extract binary mask
    if NUMBA_AVAILABLE:
        urbanized_mask = extract_year_mask_numba(wsf_data, year)
    else:
        urbanized_mask = (wsf_data > 0) & (wsf_data <= year)
        urbanized_mask = urbanized_mask.astype(np.uint8)
    
    # Find LCC
    labeled_array, num_features = ndimage.label(urbanized_mask)
    
    if num_features == 0:
        print("  ⚠️  No urbanized areas found!")
        return np.zeros((region_size, region_size), dtype=np.uint8), {
            'lcc_found': False,
            'lcc_centroid': None,
            'region_bounds': None
        }
    
    # Get LCC
    component_sizes = ndimage.sum(urbanized_mask, labeled_array, range(1, num_features + 1))
    lcc_label = np.argmax(component_sizes) + 1
    
    # Calculate LCC centroid
    lcc_mask = (labeled_array == lcc_label).astype(np.uint8)
    lcc_coords = np.argwhere(lcc_mask)
    centroid_row = int(np.mean(lcc_coords[:, 0]))
    centroid_col = int(np.mean(lcc_coords[:, 1]))
    
    # Define square region bounds centered on LCC
    half_size = region_size // 2
    row_min = max(0, centroid_row - half_size)
    row_max = min(urbanized_mask.shape[0], centroid_row + half_size)
    col_min = max(0, centroid_col - half_size)
    col_max = min(urbanized_mask.shape[1], centroid_col + half_size)
    
    actual_height = row_max - row_min
    actual_width = col_max - col_min
    
    # Extract region
    region_mask = urbanized_mask[row_min:row_max, col_min:col_max].copy()
    
    # Pad to exact square if needed (hit boundary)
    if actual_height != region_size or actual_width != region_size:
        padded_mask = np.zeros((region_size, region_size), dtype=np.uint8)
        padded_mask[:actual_height, :actual_width] = region_mask
        region_mask = padded_mask
    
    extraction_info = {
        'lcc_found': True,
        'lcc_centroid': (centroid_row, centroid_col),
        'region_bounds': (row_min, row_max, col_min, col_max),
        'padded': (actual_height != region_size or actual_width != region_size)
    }
    
    return region_mask, extraction_info


def extract_lcc_and_n_clusters_mask(
    wsf_data: np.ndarray,
    year: int,
    n_clusters: int = 5,
    region_size: int = 1000,
    output_csv: Optional[str] = None,
    coarse_grain_factor: int = 1,
    center_lcc: bool = False
) -> np.ndarray:
    """
    Extract city mask with LCC and n other clusters.
    
    **HOMOGENIZED VERSION**: Now uses region_size (pixels) instead of search radius.
    Produces output of consistent size with extract_constraint_mask().
    
    Parameters
    ----------
    wsf_data : np.ndarray
        WSF Evolution data array
    year : int
        Year to extract
    n_clusters : int, default=5
        Number of clusters to extract (including LCC)
        Labels: 1 (LCC), 2, 3, ..., n_clusters
    region_size : int, default=1000
        Size of square region in pixels around LCC
        Final output will be region_size × region_size (before coarse graining)
    output_csv : str, optional
        Path to save the mask as CSV
    coarse_grain_factor : int, default=1
        Coarse graining factor (1=no coarse graining)
    center_lcc : bool, default=False
        If True, translate all urbanized areas to center LCC in domain
        
    Returns
    -------
    np.ndarray
        Labeled mask array where LCC=1, second cluster=2, etc.
        Shape: (region_size, region_size) or smaller if coarse-grained
        
    Example
    -------
    >>> # Extract 1000×1000 region with top 10 clusters
    >>> mask = extract_lcc_and_n_clusters_mask(
    ...     wsf_data, year=2015, 
    ...     n_clusters=10,
    ...     region_size=1000,
    ...     center_lcc=True,
    ...     output_csv='clusters_2015.csv'
    ... )
    """
    start_time = time.time()
    print(f"\n{'='*60}")
    print(f"Extracting LCC and Top {n_clusters-1} Clusters for {year}")
    print(f"Region size: {region_size}×{region_size} pixels (~{region_size*0.03:.1f}km)")
    print(f"{'='*60}")
    
    # Extract region using shared function
    print(f"  Extracting region around LCC...")
    region_mask, extraction_info = extract_lcc_region(wsf_data, year, region_size)
    
    if not extraction_info['lcc_found']:
        print("  ⚠️  No LCC found, returning empty mask")
        if output_csv:
            df = pd.DataFrame(np.zeros((region_size, region_size), dtype=np.uint32))
            df.to_csv(output_csv, index=False, header=False)
        return np.zeros((region_size, region_size), dtype=np.uint32)
    
    print(f"  LCC centroid: {extraction_info['lcc_centroid']}")
    
    urbanized_pixels = region_mask.sum()
    print(f"  Urbanized pixels in region: {urbanized_pixels:,}")
    
    # Label connected components in the region
    print(f"  Labeling connected components...")
    labeled_array, num_features = ndimage.label(region_mask)
    print(f"  Found {num_features:,} connected components")
    
    if num_features == 0:
        print("  ⚠️  No components found")
        output_mask = np.zeros((region_size, region_size), dtype=np.uint32)
    else:
        # Get component sizes
        label_ids = np.arange(1, num_features + 1)
        sizes = ndimage.sum(region_mask, labeled_array, label_ids)
        
        # Sort by size
        sorted_indices = np.argsort(sizes)[::-1]
        sorted_label_ids = label_ids[sorted_indices]
        sorted_sizes = sizes[sorted_indices]
        
        # Create output mask
        output_mask = np.zeros((region_size, region_size), dtype=np.uint32)
        
        # Determine how many to extract
        n_to_extract = min(n_clusters, num_features)
        
        # Hard limit at 100 clusters
        if n_to_extract > 100:
            print(f"  ⚠️  Limiting to 100 clusters (requested {n_to_extract})")
            n_to_extract = 100
        
        print(f"\n  Extracting top {n_to_extract} clusters:")
        print(f"  {'Rank':<8} {'Label':<12} {'Size (pixels)':<15} {'Size (km²)':<12}")
        print(f"  {'-'*50}")
        
        pixel_area_km2 = 0.03 * 0.03
        
        # Assign labels: 1 for LCC, 2 for second, etc.
        for rank in range(n_to_extract):
            original_label = sorted_label_ids[rank]
            new_label = rank + 1
            size_pixels = int(sorted_sizes[rank])
            size_km2 = size_pixels * pixel_area_km2
            
            output_mask[labeled_array == original_label] = new_label
            
            cluster_type = "LCC" if rank == 0 else f"Cluster {rank}"
            
            # Only print first 10 for brevity
            if rank < 10:
                print(f"  {new_label:<8} {cluster_type:<12} {size_pixels:<15,} {size_km2:<12.2f}")
        
        if n_to_extract > 10:
            print(f"  ... and {n_to_extract - 10} more clusters")
        
        total_labeled = (output_mask > 0).sum()
        print(f"\n  Total labeled pixels: {total_labeled:,}")
    
    # Apply LCC centering if requested
    if center_lcc and (output_mask > 0).sum() > 0:
        print(f"\n  Centering LCC in domain...")
        trans_start = time.time()
        
        output_mask, trans_info = translate_mask_to_center_lcc(output_mask, lcc_label=1)
        
        trans_time = time.time() - trans_start
        print(f"  Translation completed in {trans_time:.2f}s")
        print(f"  Pixels kept: {trans_info['pixels_kept']:,}")
    
    print(f"  Pre-coarse-grain shape: {output_mask.shape}")
    
    # Apply coarse graining if requested
    if coarse_grain_factor > 1:
        print(f"\n  Applying {coarse_grain_factor}x coarse graining...")
        cg_start = time.time()
        
        if NUMBA_AVAILABLE:
            coarse_mask = coarse_grain_mask_numba(output_mask, coarse_grain_factor)
        else:
            new_height = output_mask.shape[0] // coarse_grain_factor
            new_width = output_mask.shape[1] // coarse_grain_factor
            coarse_mask = np.zeros((new_height, new_width), dtype=np.uint32)
            
            for i in range(new_height):
                for j in range(new_width):
                    block = output_mask[
                        i * coarse_grain_factor:(i + 1) * coarse_grain_factor,
                        j * coarse_grain_factor:(j + 1) * coarse_grain_factor
                    ]
                    if block.size > 0:
                        counts = np.bincount(block.ravel())
                        coarse_mask[i, j] = np.argmax(counts)
        
        cg_time = time.time() - cg_start
        
        original_size = output_mask.shape[0] * output_mask.shape[1]
        coarse_size = coarse_mask.shape[0] * coarse_mask.shape[1]
        reduction_factor = original_size / coarse_size
        
        print(f"  Coarse-grained shape: {coarse_mask.shape}")
        print(f"  Size reduction: {reduction_factor:.1f}x smaller")
        print(f"  Coarse graining time: {cg_time:.2f}s")
        
        output_mask = coarse_mask
    
    # Save to CSV if requested
    if output_csv:
        print(f"\n  Saving mask to CSV: {output_csv}")
        save_start = time.time()
        
        df = pd.DataFrame(output_mask)
        df.to_csv(output_csv, index=False, header=False)
        
        save_time = time.time() - save_start
        print(f"  ✓ Saved in {save_time:.2f}s")
        
        file_size_mb = Path(output_csv).stat().st_size / (1024 * 1024)
        print(f"  File size: {file_size_mb:.2f} MB")
    
    total_time = time.time() - start_time
    print(f"\n  Total time: {total_time:.2f}s")
    print(f"  Final shape: {output_mask.shape}")
    print(f"{'='*60}\n")
    
    return output_mask


def extract_constraint_mask(
    wsf_data: np.ndarray,
    year: int = 2015,
    region_size: int = 1000,
    output_csv: Optional[str] = None,
    coarse_grain_factor: int = 1,
    center_lcc: bool = False
) -> np.ndarray:
    """
    Extract constraint mask showing non-urbanized areas.
    
    **HOMOGENIZED VERSION**: Simplified interface to match extract_lcc_and_n_clusters_mask().
    Always extracts a region around LCC of size region_size × region_size.
    
    Parameters
    ----------
    wsf_data : np.ndarray
        WSF Evolution data array
    year : int, default=2015
        Year to extract constraint mask for
    region_size : int, default=1000
        Size of square region in pixels around LCC
        Final output will be region_size × region_size (before coarse graining)
    output_csv : str, optional
        Path to save the constraint mask as CSV
    coarse_grain_factor : int, default=1
        Coarse graining factor (1=no coarse graining)
    center_lcc : bool, default=False
        If True, translate all urbanized areas to center LCC in domain
        
    Returns
    -------
    np.ndarray
        Binary constraint mask where 1=available, 0=urbanized
        Shape: (region_size, region_size) or smaller if coarse-grained
        
    Example
    -------
    >>> # Extract constraint mask matching cluster mask dimensions
    >>> constraint = extract_constraint_mask(
    ...     wsf_data, year=2015,
    ...     region_size=1000,
    ...     center_lcc=True,
    ...     output_csv='constraint_2015.csv'
    ... )
    """
    start_time = time.time()
    print(f"\n{'='*60}")
    print(f"Extracting Constraint Mask (Non-Urbanized) for {year}")
    print(f"Region size: {region_size}×{region_size} pixels (~{region_size*0.03:.1f}km)")
    print(f"{'='*60}")
    
    # Extract region using shared function
    print(f"  Extracting region around LCC...")
    region_mask, extraction_info = extract_lcc_region(wsf_data, year, region_size)
    
    if not extraction_info['lcc_found']:
        print("  ⚠️  No LCC found, returning all-available mask")
        constraint_mask = np.ones((region_size, region_size), dtype=np.uint8)
    else:
        print(f"  LCC centroid: {extraction_info['lcc_centroid']}")
        
        urbanized_pixels = region_mask.sum()
        available_pixels = region_size * region_size - urbanized_pixels
        
        print(f"  Urbanized pixels: {urbanized_pixels:,}")
        print(f"  Available pixels: {available_pixels:,} ({available_pixels/(region_size**2)*100:.1f}%)")
        
        # Apply LCC centering if requested
        if center_lcc and urbanized_pixels > 0:
            print(f"\n  Centering LCC in domain...")
            trans_start = time.time()
            
            # Re-label the region to find LCC
            region_labeled, region_features = ndimage.label(region_mask)
            
            if region_features > 0:
                region_sizes = ndimage.sum(region_mask, region_labeled, 
                                          range(1, region_features + 1))
                region_lcc_label = np.argmax(region_sizes) + 1
                
                # Create labeled mask with LCC=1
                temp_labeled = np.zeros_like(region_mask, dtype=np.uint32)
                temp_labeled[region_labeled == region_lcc_label] = 1
                temp_labeled[region_mask & (region_labeled != region_lcc_label)] = 2
                
                # Translate to center
                centered_mask, trans_info = translate_mask_to_center_lcc(
                    temp_labeled, lcc_label=1
                )
                
                # Convert back to binary urbanized mask
                region_mask = (centered_mask > 0).astype(np.uint8)
                
                trans_time = time.time() - trans_start
                print(f"  Translation completed in {trans_time:.2f}s")
                print(f"  Pixels kept: {trans_info['pixels_kept']:,}")
        
        # Create constraint mask (INVERT: non-urbanized = 1)
        print(f"\n  Creating constraint mask...")
        constraint_mask = (1 - region_mask).astype(np.uint8)
        
        available_after = constraint_mask.sum()
        constrained_after = (constraint_mask == 0).sum()
        
        print(f"  Final breakdown:")
        print(f"    Available (1): {available_after:,} pixels")
        print(f"    Constrained (0): {constrained_after:,} pixels")
    
    print(f"  Pre-coarse-grain shape: {constraint_mask.shape}")
    
    # Apply coarse graining if requested
    if coarse_grain_factor > 1:
        print(f"\n  Applying {coarse_grain_factor}x coarse graining...")
        cg_start = time.time()
        
        if NUMBA_AVAILABLE:
            coarse_mask = coarse_grain_mask_numba(constraint_mask, coarse_grain_factor)
        else:
            new_height = constraint_mask.shape[0] // coarse_grain_factor
            new_width = constraint_mask.shape[1] // coarse_grain_factor
            coarse_mask = np.zeros((new_height, new_width), dtype=np.uint8)
            
            for i in range(new_height):
                for j in range(new_width):
                    block = constraint_mask[
                        i * coarse_grain_factor:(i + 1) * coarse_grain_factor,
                        j * coarse_grain_factor:(j + 1) * coarse_grain_factor
                    ]
                    if block.size > 0:
                        # Majority voting for constraints
                        coarse_mask[i, j] = 1 if block.mean() >= 0.5 else 0
        
        cg_time = time.time() - cg_start
        
        original_size = constraint_mask.shape[0] * constraint_mask.shape[1]
        coarse_size = coarse_mask.shape[0] * coarse_mask.shape[1]
        reduction_factor = original_size / coarse_size
        
        print(f"  Coarse-grained shape: {coarse_mask.shape}")
        print(f"  Size reduction: {reduction_factor:.1f}x smaller")
        print(f"  Coarse graining time: {cg_time:.2f}s")
        
        constraint_mask = coarse_mask
    
    # Save to CSV if requested
    if output_csv:
        print(f"\n  Saving constraint mask to CSV: {output_csv}")
        save_start = time.time()
        
        df = pd.DataFrame(constraint_mask)
        df.to_csv(output_csv, index=False, header=False)
        
        save_time = time.time() - save_start
        print(f"  ✓ Saved in {save_time:.2f}s")
        
        file_size_mb = Path(output_csv).stat().st_size / (1024 * 1024)
        print(f"  File size: {file_size_mb:.2f} MB")
    
    total_time = time.time() - start_time
    print(f"\n  Total time: {total_time:.2f}s")
    print(f"  Final shape: {constraint_mask.shape}")
    print(f"{'='*60}\n")
    
    return constraint_mask


def visualize_clusters_optimized(wsf_data: np.ndarray, 
                                 analyzer: BuiltAreaAnalyzer,
                                 year: int, 
                                 radius_factor: float,
                                 n_clusters: int,
                                 output_path: Optional[str] = None,
                                 figsize: Tuple[int, int] = (12, 12), 
                                 dpi: int = 150,
                                 show_circle: bool = True, 
                                 crop_factor: Optional[float] = None) -> Dict:
    """
    Optimized cluster visualization with significant speedup.
    
    Improvements:
    - Uses labeled array directly (avoids creating N separate masks)
    - Numba-accelerated RGB creation
    - Vectorized operations where possible
    - More efficient memory usage
    
    Styling:
    - Black background
    - Gray for urbanized areas (only within search circle)
    - Colored clusters (orange, cyan, yellow, magenta, green, etc.)
    - Red for LCC (Largest Connected Component)
    
    Args:
        wsf_data: WSF Evolution array
        analyzer: BuiltAreaAnalyzer instance
        year: Year to analyze
        radius_factor: Search radius multiplier
        n_clusters: Number of clusters to visualize
        output_path: Save path (None to display)
        figsize: Figure size tuple
        dpi: Resolution
        show_circle: Draw search circle
        crop_factor: Crop multiplier around LCC (None for full image)
                    Crop is centered on LCC centroid
    
    Returns:
        Statistics dictionary with cluster info and timing
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    
    print(f"Visualizing clusters ({NUMBA_AVAILABLE and 'OPTIMIZED' or 'STANDARD'} mode)...")
    start_time = time.time()
    
    # Extract mask for the year
    mask = analyzer.extract_year_mask(wsf_data, year)
    
    # Get LCC
    lcc_mask, lcc_size = analyzer.find_largest_connected_component(mask)
    
    if lcc_size == 0:
        print("No LCC found")
        return {'success': False, 'error': 'No LCC found'}
    
    # Calculate LCC centroid and radius
    centroid_row, centroid_col = analyzer.calculate_lcc_centroid(lcc_mask)
    mean_radius = analyzer.calculate_mean_radius(lcc_mask, centroid_row, centroid_col)
    circle_radius = mean_radius * radius_factor
    
    # Create circular mask
    if NUMBA_AVAILABLE:
        circle_mask = create_circular_mask_numba(
            wsf_data.shape[0], wsf_data.shape[1],
            centroid_row, centroid_col, circle_radius
        )
    else:
        rows, cols = np.ogrid[0:wsf_data.shape[0], 0:wsf_data.shape[1]]
        distance_from_centroid = np.sqrt((rows - centroid_row)**2 + (cols - centroid_col)**2)
        circle_mask = (distance_from_centroid <= circle_radius).astype(np.uint8)
    
    # Get built areas in circle
    built_in_circle = mask & circle_mask
    
    # Get LCC in circle
    lcc_in_circle = lcc_mask & circle_mask
    
    # Get non-LCC areas
    non_lcc_in_circle = built_in_circle & (~lcc_mask)
    
    # Extract clusters (optimized)
    clusters, labeled_array, num_features = extract_clusters_optimized(
        non_lcc_in_circle, n_clusters - 1
    )
    
    extract_time = time.time() - start_time
    print(f"  Cluster extraction: {extract_time:.3f}s")
    
    # Prepare cluster data
    pixel_area_km2 = (30 * 30) / 1e6
    
    cluster_data = {
        'success': True,
        'year': year,
        'centroid_row': round(centroid_row, 2),
        'centroid_col': round(centroid_col, 2),
        'circle_radius_km': round(circle_radius * 0.03, 3),
        'radius_factor': radius_factor,
        'lcc': {
            'cluster_id': 0,
            'size_pixels': int(lcc_in_circle.sum()),
            'size_km2': round(lcc_in_circle.sum() * pixel_area_km2, 3)
        },
        'other_clusters': [
            {
                'cluster_id': i + 1,
                'size_pixels': size,
                'size_km2': round(size * pixel_area_km2, 3)
            }
            for i, label_id, size in clusters
        ],
        'total_clusters': 1 + len(clusters)
    }
    
    # Handle cropping - CENTER ON LCC CENTROID
    crop_bounds = None
    if crop_factor is not None:
        crop_radius = int(circle_radius * crop_factor)
        
        # Center crop on LCC centroid (not search circle centroid)
        lcc_centroid_row, lcc_centroid_col = analyzer.calculate_lcc_centroid(lcc_mask)
        
        row_min = max(0, int(lcc_centroid_row - crop_radius))
        row_max = min(wsf_data.shape[0], int(lcc_centroid_row + crop_radius))
        col_min = max(0, int(lcc_centroid_col - crop_radius))
        col_max = min(wsf_data.shape[1], int(lcc_centroid_col + crop_radius))
        
        crop_bounds = {
            'row_min': row_min, 'row_max': row_max,
            'col_min': col_min, 'col_max': col_max,
            'crop_radius_km': round(crop_radius * 0.03, 3),
            'crop_center_row': round(lcc_centroid_row, 2),
            'crop_center_col': round(lcc_centroid_col, 2)
        }
        
        # Crop all arrays
        mask_display = mask[row_min:row_max, col_min:col_max]
        circle_mask_display = circle_mask[row_min:row_max, col_min:col_max]
        lcc_display = lcc_in_circle[row_min:row_max, col_min:col_max]
        labeled_display = labeled_array[row_min:row_max, col_min:col_max] if labeled_array is not None else None
        
        # Filter mask to only show areas within circle
        mask_display = mask_display & circle_mask_display
        
        # Adjust display coordinates relative to crop
        centroid_row_display = centroid_row - row_min
        centroid_col_display = centroid_col - col_min
    else:
        # Filter mask to only show areas within circle
        mask_display = mask & circle_mask
        lcc_display = lcc_in_circle
        labeled_display = labeled_array
        centroid_row_display = centroid_row
        centroid_col_display = centroid_col
    
    # Create RGB visualization (OPTIMIZED)
    # BLACK BACKGROUND (0,0,0), GRAY URBANIZED (128,128,128), COLORED CLUSTERS, RED LCC
    rgb_start = time.time()
    
    if NUMBA_AVAILABLE and labeled_display is not None and len(clusters) > 0:
        # Use optimized labeled array method (fastest)
        # Base colors for first 9 clusters
        base_cluster_colors = [
            [255, 165, 0],   # Orange
            [0, 255, 255],   # Cyan
            [255, 255, 0],   # Yellow
            [255, 0, 255],   # Magenta
            [0, 255, 0],     # Green
            [255, 192, 203], # Pink
            [255, 128, 0],   # Dark Orange
            [128, 0, 255],   # Purple
            [0, 128, 255],   # Light Blue
        ]
        
        n_other = len(clusters)
        
        # If we have more than 9 clusters, generate additional colors using HSV colormap
        if n_other > len(base_cluster_colors):
            import matplotlib.pyplot as plt
            # Keep the base colors and add more from HSV
            additional_colors = []
            for i in range(len(base_cluster_colors), n_other):
                # Use HSV colormap to generate distinct colors
                hsv_color = plt.cm.hsv(i / n_other)
                # Convert from 0-1 range to 0-255 range (RGB only, ignore alpha)
                rgb_color = [int(hsv_color[0] * 255), int(hsv_color[1] * 255), int(hsv_color[2] * 255)]
                additional_colors.append(rgb_color)
            
            cluster_colors_list = base_cluster_colors + additional_colors
        else:
            cluster_colors_list = base_cluster_colors[:n_other]
        
        cluster_colors_array = np.array(cluster_colors_list, dtype=np.uint8)
        
        # Create label ID array
        cluster_label_ids = np.array([label_id for _, label_id, _ in clusters], dtype=np.int32)
        
        # Start with BLACK background (0,0,0) - initialize to zeros
        rgb = np.zeros((*labeled_display.shape, 3), dtype=np.uint8)
        
        # Add GRAY for all urbanized areas (only within circle now)
        rgb[mask_display == 1] = [128, 128, 128]
        
        # Apply colored clusters on top
        for idx, (_, label_id, _) in enumerate(clusters):
            if idx < len(cluster_colors_array):
                cluster_mask = labeled_display == label_id
                rgb[cluster_mask] = cluster_colors_array[idx]
        
        # Add LCC in RED (highest priority - on top)
        rgb[lcc_display == 1] = [255, 0, 0]
        
    else:
        # Fallback to standard numpy indexing
        # Start with BLACK background (0,0,0) - initialize to zeros
        rgb = np.zeros((*mask_display.shape, 3), dtype=np.uint8)
        
        # Add GRAY for all urbanized areas (only within circle now)
        rgb[mask_display == 1] = [128, 128, 128]
        
        # Color clusters
        if labeled_display is not None and len(clusters) > 0:
            # Base colors for first 9 clusters
            base_cluster_colors = [
                [255, 165, 0],   # Orange
                [0, 255, 255],   # Cyan
                [255, 255, 0],   # Yellow
                [255, 0, 255],   # Magenta
                [0, 255, 0],     # Green
                [255, 192, 203], # Pink
                [255, 128, 0],   # Dark Orange
                [128, 0, 255],   # Purple
                [0, 128, 255],   # Light Blue
            ]
            
            n_other = len(clusters)
            
            # If we have more than 9 clusters, generate additional colors using HSV colormap
            if n_other > len(base_cluster_colors):
                import matplotlib.pyplot as plt
                # Keep the base colors and add more from HSV
                cluster_colors = base_cluster_colors.copy()
                for i in range(len(base_cluster_colors), n_other):
                    # Use HSV colormap to generate distinct colors
                    hsv_color = plt.cm.hsv(i / n_other)
                    # Convert from 0-1 range to 0-255 range (RGB only, ignore alpha)
                    rgb_color = [int(hsv_color[0] * 255), int(hsv_color[1] * 255), int(hsv_color[2] * 255)]
                    cluster_colors.append(rgb_color)
            else:
                cluster_colors = base_cluster_colors[:n_other]
            
            for idx, (_, label_id, _) in enumerate(clusters):
                if idx < len(cluster_colors):
                    cluster_mask = labeled_display == label_id
                    rgb[cluster_mask] = cluster_colors[idx]
        
        # LCC in RED on top (highest priority)
        rgb[lcc_display == 1] = [255, 0, 0]
    
    rgb_time = time.time() - rgb_start
    print(f"  RGB creation: {rgb_time:.3f}s")
    
    # Create figure with black background
    plot_start = time.time()
    
    fig, ax = plt.subplots(figsize=figsize, facecolor='black')
    ax.set_facecolor('black')
    ax.imshow(rgb)
    
    # Draw circle (white dashed)
    if show_circle:
        circle = plt.Circle(
            (centroid_col_display, centroid_row_display),
            circle_radius if crop_factor is None else circle_radius / crop_factor,
            fill=False, edgecolor='white', linewidth=2,
            linestyle='--'
        )
        ax.add_patch(circle)
        ax.plot(centroid_col_display, centroid_row_display,
               'w+', markersize=15, markeredgewidth=2)
    
    # Title (white text)
    n_other = len(clusters)
    total_other_area = sum(c['size_km2'] for c in cluster_data['other_clusters'])
    
    title_text = (
        f'Top {cluster_data["total_clusters"]} Clusters - {year}\n'
        f'LCC (red): {cluster_data["lcc"]["size_km2"]:.2f} km² | '
        f'Others: {n_other} clusters, {total_other_area:.2f} km²'
    )
    
    ax.set_title(title_text, color='white', fontsize=11, pad=15)
    ax.axis('off')
    
    # Legend with black background
    legend_elements = [
        Patch(facecolor='red', label=f'LCC ({cluster_data["lcc"]["size_km2"]:.2f} km²)')
    ]
    
    # Define base colors for legend (matching visualization colors)
    base_legend_colors = [
        'orange', 'cyan', 'yellow', 'magenta', 'green',
        'pink', 'darkorange', 'purple', 'lightblue'
    ]
    
    n_other = len(cluster_data['other_clusters'])
    
    # Generate additional colors if we have more than 9 clusters
    if n_other > len(base_legend_colors):
        import matplotlib.pyplot as plt
        legend_colors_list = base_legend_colors.copy()
        for i in range(len(base_legend_colors), n_other):
            # Use HSV colormap to generate distinct colors
            hsv_color = plt.cm.hsv(i / n_other)
            legend_colors_list.append(hsv_color)
    else:
        legend_colors_list = base_legend_colors
    
    # Show up to 10 clusters in legend (instead of just 5)
    max_legend_clusters = min(10, n_other)
    
    for i in range(max_legend_clusters):
        cluster = cluster_data['other_clusters'][i]
        color = legend_colors_list[i] if i < len(legend_colors_list) else 'grey'
        legend_elements.append(
            Patch(facecolor=color, 
                 label=f'Cluster {i+1} ({cluster["size_km2"]:.2f} km²)')
        )
    
    if n_other > max_legend_clusters:
        legend_elements.append(
            Patch(facecolor='grey', 
                 label=f'+ {n_other - max_legend_clusters} more')
        )
    
    legend_elements.append(Patch(facecolor='grey', label='Other built areas'))
    
    ax.legend(handles=legend_elements, loc='upper right',
             facecolor='black', edgecolor='white',
             labelcolor='white', fontsize=8)
    
    plt.tight_layout()
    
    plot_time = time.time() - plot_start
    print(f"  Matplotlib: {plot_time:.3f}s")
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                   facecolor='black', edgecolor='none')
        print(f"  ✓ Saved: {output_path}")
    else:
        plt.show()
    
    plt.close(fig)
    
    total_time = time.time() - start_time
    print(f"  Total visualization: {total_time:.3f}s")
    
    # Add timing info
    cluster_data['timing'] = {
        'extraction': round(extract_time, 3),
        'rgb_creation': round(rgb_time, 3),
        'plotting': round(plot_time, 3),
        'total': round(total_time, 3)
    }
    
    if crop_bounds:
        cluster_data['crop_info'] = crop_bounds
    
    cluster_data['image_shape'] = wsf_data.shape
    cluster_data['display_shape'] = mask_display.shape
    
    return cluster_data



def calculate_lcc_density_metrics(
    wsf_data: np.ndarray,
    year: int,
    analyzer: Optional[BuiltAreaAnalyzer] = None
) -> Dict:
    """
    Calculate the ratio of non-urbanized pixels in the LCC region using three methods.
    Method 1 (Bounding Box): Uses the strict rectangular perimeter (bounding box) of the LCC.
    Method 2 (Convex Hull): Uses the convex hull (smallest convex polygon) containing the LCC.
    Method 3 (Filled Holes): Fills internal holes in the LCC to consider only non-urbanized regions inside the LCC.
    The ratio represents "porosity" or "fragmentation" - how much empty space exists
    within the urban area's boundary.
    Parameters
    ----------
    wsf_data : np.ndarray
        WSF Evolution data array
    year : int
        Year to analyze
    analyzer : BuiltAreaAnalyzer, optional
        If provided, uses this analyzer. Otherwise creates a new one.
    Returns
    -------
    dict
        Dictionary with metrics for all methods:
        - 'year': year analyzed
        - 'lcc_pixels': total LCC pixels (urbanized)
        - 'lcc_area_km2': LCC area in km²
        Bounding Box metrics:
        - 'bbox_total_pixels': total pixels in bounding box
        - 'bbox_urbanized_pixels': urbanized pixels in bbox
        - 'bbox_non_urbanized_pixels': non-urbanized pixels in bbox
        - 'bbox_non_urbanized_ratio': ratio of non-urbanized pixels (0-1)
        - 'bbox_bounds': (row_min, row_max, col_min, col_max)
        Convex Hull metrics:
        - 'convex_hull_total_pixels': total pixels in convex hull
        - 'convex_hull_urbanized_pixels': urbanized pixels in hull
        - 'convex_hull_non_urbanized_pixels': non-urbanized pixels in hull
        - 'convex_hull_non_urbanized_ratio': ratio of non-urbanized pixels (0-1)
        - 'convex_hull_vertices': number of vertices in hull
        Filled Holes metrics:
        - 'filled_total_pixels': total pixels in filled LCC
        - 'filled_urbanized_pixels': urbanized pixels in filled
        - 'filled_non_urbanized_pixels': non-urbanized pixels in filled
        - 'filled_non_urbanized_ratio': ratio of non-urbanized pixels (0-1)
    Example
    -------
    >>> metrics = calculate_lcc_density_metrics(wsf_data, year=2015)
    >>> print(f"Bbox porosity: {metrics['bbox_non_urbanized_ratio']:.2%}")
    >>> print(f"Hull porosity: {metrics['convex_hull_non_urbanized_ratio']:.2%}")
    >>> print(f"Filled porosity: {metrics['filled_non_urbanized_ratio']:.2%}")
    """
    from scipy.spatial import ConvexHull
    from skimage.draw import polygon
    from scipy.ndimage import binary_fill_holes
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
    import numpy as np
    # Create analyzer if not provided
    if analyzer is None:
        analyzer = BuiltAreaAnalyzer()
    print(f"\n{'='*60}")
    print(f"Calculating LCC Density Metrics for {year}")
    print(f"{'='*60}")
    # Extract urbanized mask for the year
    print(" Extracting urbanized areas...")
    urbanized_mask = analyzer.extract_year_mask(wsf_data, year)
    if urbanized_mask.sum() == 0:
        print(" ⚠️ No urbanized areas found!")
        return {
            'year': year,
            'lcc_pixels': 0,
            'lcc_area_km2': 0.0,
            'bbox_non_urbanized_ratio': None,
            'convex_hull_non_urbanized_ratio': None,
            'filled_non_urbanized_ratio': None,
            'error': 'No urbanized areas'
        }
    # Find LCC
    print(" Finding largest connected component...")
    lcc_mask, lcc_size = analyzer.find_largest_connected_component(urbanized_mask)
    if lcc_size == 0:
        print(" ⚠️ No LCC found!")
        return {
            'year': year,
            'lcc_pixels': 0,
            'lcc_area_km2': 0.0,
            'bbox_non_urbanized_ratio': None,
            'convex_hull_non_urbanized_ratio': None,
            'filled_non_urbanized_ratio': None,
            'error': 'No LCC found'
        }
    pixel_area_km2 = 0.03 * 0.03 # 30m × 30m
    lcc_area_km2 = lcc_size * pixel_area_km2
    print(f" LCC size: {lcc_size:,} pixels ({lcc_area_km2:.2f} km²)")
    # Get LCC coordinates
    lcc_coords = np.argwhere(lcc_mask == 1)
    # ========================================================================
    # METHOD 1: BOUNDING BOX (Strict Perimeter)
    # ========================================================================
    print("\n Method 1: Bounding Box Analysis")
    row_min = lcc_coords[:, 0].min()
    row_max = lcc_coords[:, 0].max()
    col_min = lcc_coords[:, 1].min()
    col_max = lcc_coords[:, 1].max()
    bbox_height = row_max - row_min + 1
    bbox_width = col_max - col_min + 1
    bbox_total_pixels = bbox_height * bbox_width
    print(f" Bounding box: {bbox_height} × {bbox_width} pixels")
    print(f" Total bbox pixels: {bbox_total_pixels:,}")
    # Extract bbox region
    bbox_region = lcc_mask[row_min:row_max+1, col_min:col_max+1]
    bbox_urbanized = bbox_region.sum()
    bbox_non_urbanized = bbox_total_pixels - bbox_urbanized
    bbox_ratio = bbox_non_urbanized / bbox_total_pixels if bbox_total_pixels > 0 else 0.0
    print(f" Urbanized pixels: {bbox_urbanized:,}")
    print(f" Non-urbanized pixels: {bbox_non_urbanized:,}")
    print(f" Non-urbanized ratio: {bbox_ratio:.4f} ({bbox_ratio*100:.2f}%)")
    # ========================================================================
    # METHOD 2: CONVEX HULL
    # ========================================================================
    print("\n Method 2: Convex Hull Analysis")
    # Compute convex hull of LCC coordinates
    # Note: ConvexHull expects (x, y) not (row, col), so we swap
    points_xy = lcc_coords[:, [1, 0]] # Swap to (col, row) = (x, y)
    try:
        hull = ConvexHull(points_xy)
        hull_vertices = points_xy[hull.vertices]
        print(f" Convex hull vertices: {len(hull_vertices)}")
        # Create binary mask of convex hull region
        hull_mask = np.zeros_like(lcc_mask, dtype=np.uint8)
        # Use polygon to fill the convex hull
        # polygon expects (row, col) coordinates
        rr, cc = polygon(hull_vertices[:, 1], hull_vertices[:, 0], lcc_mask.shape)
        hull_mask[rr, cc] = 1
        hull_total_pixels = hull_mask.sum()
        # Count urbanized pixels within hull
        hull_urbanized = (hull_mask & lcc_mask).sum()
        hull_non_urbanized = hull_total_pixels - hull_urbanized
        hull_ratio = hull_non_urbanized / hull_total_pixels if hull_total_pixels > 0 else 0.0
        print(f" Total hull pixels: {hull_total_pixels:,}")
        print(f" Urbanized pixels: {hull_urbanized:,}")
        print(f" Non-urbanized pixels: {hull_non_urbanized:,}")
        print(f" Non-urbanized ratio: {hull_ratio:.4f} ({hull_ratio*100:.2f}%)")
        hull_success = True
        hull_error = None
    except Exception as e:
        print(f" ⚠️ Convex hull computation failed: {e}")
        hull_total_pixels = 0
        hull_urbanized = 0
        hull_non_urbanized = 0
        hull_ratio = None
        hull_vertices = []
        hull_success = False
        hull_error = str(e)
    # ========================================================================
    # METHOD 3: FILLED HOLES (Strict Front Considering Internal Non-Urbanized)
    # ========================================================================
    print("\n Method 3: Filled Holes Analysis")
    try:
        filled_mask = binary_fill_holes(lcc_mask)
        filled_total_pixels = filled_mask.sum()
        filled_urbanized = lcc_size
        filled_non_urbanized = filled_total_pixels - filled_urbanized
        filled_ratio = filled_non_urbanized / filled_total_pixels if filled_total_pixels > 0 else 0.0
        print(f" Total filled pixels: {filled_total_pixels:,}")
        print(f" Urbanized pixels: {filled_urbanized:,}")
        print(f" Non-urbanized pixels: {filled_non_urbanized:,}")
        print(f" Non-urbanized ratio: {filled_ratio:.4f} ({filled_ratio*100:.2f}%)")
        filled_success = True
        filled_error = None
    except Exception as e:
        print(f" ⚠️ Filled holes computation failed: {e}")
        filled_total_pixels = 0
        filled_urbanized = 0
        filled_non_urbanized = 0
        filled_ratio = None
        filled_success = False
        filled_error = str(e)
    # ========================================================================
    # COMPARISON
    # ========================================================================
    # print(f"\n Comparison:")
    # print(f" Bounding box non-urbanized: {bbox_ratio*100:.2f}%")
    # if hull_success:
    #     print(f" Convex hull non-urbanized: {hull_ratio*100:.2f}%")
    # if filled_success:
    #     print(f" Filled holes non-urbanized: {filled_ratio*100:.2f}%")
    # if hull_success:
    #     difference_hull_bbox = abs(bbox_ratio - hull_ratio) * 100
    #     print(f" Difference (bbox vs hull): {difference_hull_bbox:.2f} percentage points")
    # if filled_success:
    #     difference_filled_bbox = abs(bbox_ratio - filled_ratio) * 100
    #     print(f" Difference (bbox vs filled): {difference_filled_bbox:.2f} percentage points")
    #     if hull_success:
    #         difference_filled_hull = abs(hull_ratio - filled_ratio) * 100
    #         print(f" Difference (hull vs filled): {difference_filled_hull:.2f} percentage points")
    # # Interpretation
    # if hull_success and filled_success:
    #     if filled_ratio < hull_ratio < bbox_ratio:
    #         print(f" → Filled is most compact, then hull, then bbox")
    #     # Add more if needed
    # print(f"{'='*60}\n")
    # # ========================================================================
    # # VISUALIZATION (Optimized with cropping)
    # # ========================================================================
    # print("Visualizing delimitations...")
    # buffer = 10  # Margin around the bounding box for visualization
    # # Define visualization crop bounds
    # vis_row_min = max(0, row_min - buffer)
    # vis_row_max = min(lcc_mask.shape[0] - 1, row_max + buffer)
    # vis_col_min = max(0, col_min - buffer)
    # vis_col_max = min(lcc_mask.shape[1] - 1, col_max + buffer)
    # # Crop lcc_mask
    # cropped_lcc = lcc_mask[vis_row_min:vis_row_max+1, vis_col_min:vis_col_max+1]
    # # Relative coordinates for bbox
    # rel_row_min = row_min - vis_row_min
    # rel_row_max = row_max - vis_row_min
    # rel_col_min = col_min - vis_col_min
    # rel_col_max = col_max - vis_col_min
    # # Create cropped bbox_mask
    # cropped_bbox_mask = np.zeros_like(cropped_lcc, dtype=bool)
    # cropped_bbox_mask[max(0, rel_row_min):min(cropped_lcc.shape[0], rel_row_max + 1),
    #                   max(0, rel_col_min):min(cropped_lcc.shape[1], rel_col_max + 1)] = True
    # # Cropped non-urbanized for bbox
    # cropped_non_urban_bbox = cropped_bbox_mask & ~cropped_lcc.astype(bool)
    # num_subplots = 1
    # if hull_success:
    #     num_subplots += 1
    # if filled_success:
    #     num_subplots += 1
    # fig, axs = plt.subplots(1, num_subplots, figsize=(10 * num_subplots, 10))
    # if num_subplots == 1:
    #     axs = [axs]  # Make it iterable
    # subplot_idx = 0
    # # Bounding Box Plot
    # ax_bbox = axs[subplot_idx]
    # ax_bbox.imshow(cropped_lcc, cmap='gray', interpolation='none')
    # ax_bbox.imshow(cropped_non_urban_bbox, cmap='Reds', alpha=0.3, interpolation='none')
    # # Add bounding box rectangle (relative)
    # rect = Rectangle((rel_col_min, rel_row_min), bbox_width, bbox_height, edgecolor='blue', facecolor='none', linewidth=2)
    # ax_bbox.add_patch(rect)
    # ax_bbox.set_title('Bounding Box Delimitation')
    # subplot_idx += 1
    # if hull_success:
    #     # Crop hull_mask
    #     cropped_hull_mask = hull_mask[vis_row_min:vis_row_max+1, vis_col_min:vis_col_max+1]
    #     # Cropped non-urbanized for hull
    #     cropped_non_urban_hull = cropped_hull_mask.astype(bool) & ~cropped_lcc.astype(bool)
    #     # Relative hull vertices
    #     hull_vertices_rel = hull_vertices.copy()
    #     hull_vertices_rel[:, 0] -= vis_col_min  # x (col)
    #     hull_vertices_rel[:, 1] -= vis_row_min  # y (row)
    #     # Convex Hull Plot
    #     ax_hull = axs[subplot_idx]
    #     ax_hull.imshow(cropped_lcc, cmap='gray', interpolation='none')
    #     ax_hull.imshow(cropped_non_urban_hull, cmap='Reds', alpha=0.3, interpolation='none')
    #     # Add convex hull polygon (relative)
    #     poly = Polygon(hull_vertices_rel, edgecolor='blue', facecolor='none', linewidth=2)
    #     ax_hull.add_patch(poly)
    #     ax_hull.set_title('Convex Hull Delimitation')
    #     subplot_idx += 1
    # if filled_success:
    #     # Crop filled_mask
    #     cropped_filled_mask = filled_mask[vis_row_min:vis_row_max+1, vis_col_min:vis_col_max+1]
    #     # Cropped non-urbanized for filled
    #     cropped_non_urban_filled = cropped_filled_mask.astype(bool) & ~cropped_lcc.astype(bool)
    #     # Filled Holes Plot
    #     ax_filled = axs[subplot_idx]
    #     ax_filled.imshow(cropped_lcc, cmap='gray', interpolation='none')
    #     ax_filled.imshow(cropped_non_urban_filled, cmap='Reds', alpha=0.3, interpolation='none')
    #     # Add contour for filled boundary
    #     ax_filled.contour(cropped_filled_mask, levels=[0.5], colors='blue', linewidths=2)
    #     ax_filled.set_title('Filled Holes Delimitation')
    # plt.tight_layout()
    # plt.show()
    # Return comprehensive metrics
    result = {
        'year': year,
        'lcc_pixels': int(lcc_size),
        'lcc_area_km2': round(lcc_area_km2, 3),
        # Bounding Box
        'bbox_total_pixels': int(bbox_total_pixels),
        'bbox_urbanized_pixels': int(bbox_urbanized),
        'bbox_non_urbanized_pixels': int(bbox_non_urbanized),
        'bbox_non_urbanized_ratio': round(float(bbox_ratio), 6),
        'bbox_bounds': (int(row_min), int(row_max), int(col_min), int(col_max)),
        'bbox_size_pixels': (int(bbox_height), int(bbox_width)),
        # Convex Hull
        'convex_hull_success': hull_success,
        'convex_hull_total_pixels': int(hull_total_pixels),
        'convex_hull_urbanized_pixels': int(hull_urbanized),
        'convex_hull_non_urbanized_pixels': int(hull_non_urbanized),
        'convex_hull_non_urbanized_ratio': round(float(hull_ratio), 6) if hull_success else None,
        'convex_hull_vertices': len(hull_vertices) if hull_success else 0,
        'convex_hull_error': hull_error,
        # Filled Holes
        'filled_success': filled_success,
        'filled_total_pixels': int(filled_total_pixels),
        'filled_urbanized_pixels': int(filled_urbanized),
        'filled_non_urbanized_pixels': int(filled_non_urbanized),
        'filled_non_urbanized_ratio': round(float(filled_ratio), 6) if filled_success else None,
        'filled_error': filled_error
    }
    return result