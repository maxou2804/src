import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import os
from adjustText import adjust_text



# Try to import adjustText for better label placement
try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False
    print("adjustText not installed. Using custom label placement. Install with: pip install adjustText")
def add_city_labels_with_adjusttext(ax, x_data, y_data, cities, fontsize=9):
    """Add city labels using adjustText library for automatic overlap avoidance"""
    texts = []
    for x, y, city in zip(x_data, y_data, cities):
        texts.append(ax.annotate(city, xy=(x, y), fontsize=fontsize))
    
    # Adjust text positions to minimize overlaps
    adjust_text(texts, 
                x=x_data, y=y_data,
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5),
                ha='center', va='center',
                force_text=(0.3, 0.5),  # Force to push texts apart
                expand_text=(1.2, 1.5),  # Expand text bounding boxes
                expand_points=(1.2, 1.2))  # Expand point bounding boxes

def add_city_labels_custom(ax, x_data, y_data, cities, fontsize=9):
    """Custom label placement with collision detection to minimize overlap"""
    
    # Get axis limits for scaling
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    
    # Store placed labels for collision detection
    placed_labels = []
    
    # Define possible offset directions (8 directions + center)
    offset_directions = [
        (0.02, 0.02),    # top-right
        (-0.02, 0.02),   # top-left
        (0.02, -0.02),   # bottom-right
        (-0.02, -0.02),  # bottom-left
        (0.03, 0),       # right
        (-0.03, 0),      # left
        (0, 0.03),       # top
        (0, -0.03),      # bottom
        (0.04, 0.01),    # far top-right
        (-0.04, 0.01),   # far top-left
        (0.04, -0.01),   # far bottom-right
        (-0.04, -0.01),  # far bottom-left
    ]
    
    def get_text_bbox(x, y, text, fontsize):
        """Estimate bounding box for text (simplified)"""
        # Rough estimation: each character is about 0.01 * x_range wide
        width = len(text) * 0.008 * x_range
        height = 0.02 * y_range
        return (x - width/2, y - height/2, x + width/2, y + height/2)
    
    def check_overlap(bbox1, bbox2):
        """Check if two bounding boxes overlap"""
        return not (bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or 
                   bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3])
    
    # Sort cities by their y-position to process from bottom to top
    sorted_indices = np.argsort(y_data)
    
    for idx in sorted_indices:
        x, y, city = x_data[idx], y_data[idx], cities[idx]
        
        best_offset = None
        min_overlaps = float('inf')
        
        # Try each offset direction
        for offset_x, offset_y in offset_directions:
            # Calculate actual offset based on data range
            actual_offset_x = offset_x * x_range
            actual_offset_y = offset_y * y_range
            
            # Calculate text position
            text_x = x + actual_offset_x
            text_y = y + actual_offset_y
            
            # Get bounding box for this position
            text_bbox = get_text_bbox(text_x, text_y, city, fontsize)
            
            # Count overlaps with already placed labels
            overlaps = 0
            for placed_bbox in placed_labels:
                if check_overlap(text_bbox, placed_bbox):
                    overlaps += 1
            
            # Update best position if this has fewer overlaps
            if overlaps < min_overlaps:
                min_overlaps = overlaps
                best_offset = (actual_offset_x, actual_offset_y, text_x, text_y, text_bbox)
                if overlaps == 0:
                    break  # Found perfect position
        
        # Place label at best position
        if best_offset:
            offset_x, offset_y, text_x, text_y, text_bbox = best_offset
            
            # Determine text alignment based on offset direction
            if offset_x > 0:
                ha = 'left'
            elif offset_x < 0:
                ha = 'right'
            else:
                ha = 'center'
            
            if offset_y > 0:
                va = 'bottom'
            elif offset_y < 0:
                va = 'top'
            else:
                va = 'center'
            
            # Add the annotation
            ax.annotate(city, xy=(x, y), 
                       xytext=(text_x, text_y),
                       fontsize=fontsize, ha=ha, va=va,
                       arrowprops=dict(arrowstyle='-', lw=0.5, color='gray', alpha=0.3))
            
            # Store the bounding box
            placed_labels.append(text_bbox)

# Choose which labeling function to use
if HAS_ADJUST_TEXT:
    add_city_labels = add_city_labels_with_adjusttext
    print("Using adjustText for optimal label placement")
else:
    add_city_labels = add_city_labels_custom
    print("Using custom label placement algorithm")








# Beta coefficient data for cities
data=pd.DataFrame({'City': ['Ningbo','Chengdu Deyang', 'Beijing Lafang','Changzhou','Bengalore','Kolkata','Paris','Bangkok','Cairo','Guatemala City','Johannesburg','London','Mexico City','Nairobi','Santiago','Sao Paulo','Tehran','Las Vegas','Atlanta'],
      'alpha':[0.56, 0.53, 0.54, 0.54, 0.55, 0.52, 0.52 ,0.53, 0.53, 0.52, 0.58, 0.54, 0.55, 0.56, 0.58, 0.51, 0.55, 0.55, 0.56],
      'beta': [0.44, 0.68, 0.41, 0.37, 0.83, 0.34, 0.56, 1.01, 0.37, 0.37, 0.07, 0.01, 0.04, 0.28, 0.62, 0.89, 0.10, 0.41, 0.04],
       '1/z': [0.58, 0.68, 0.58, 0.56, 0.72, 0.74, 0.76, 0.80, 0.54, 0.33, 0.27, 0.21, 0.25, 0.27, 0.74, 0.41, 0.54, 0.37, 0.52]})

data = data.sort_values('City')
directory = 'C:\\Users\\trique\\Downloads\\MASTER_THESIS\\data_vizualization\\render_report\\csv_outputs'

output_dir='C:\\Users\\trique\\Downloads\\MASTER_THESIS\\data_vizualization\\render_report\\growth_correlation'

# Set style
sns.set_style("whitegrid")

print("="*80)
print("COMBINED GROWTH RATE ANALYSIS - ALL CITIES")
print("="*80)

# Store all growth data from all cities
all_cities_growth_data = []
all_cities_trajectories = []
city_statistics = []
cluster0_statistics = []

# Process each city
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        filepath = os.path.join(directory, filename)
        city_name = filename.replace('output_', '').replace('.csv', '')
        
        print(f"\nProcessing {city_name}...")
        
        # Read the CSV file
        df = pd.read_csv(filepath)
        
        # ========================================================================
        # CLUSTER 0 (LARGEST COMPONENT) GROWTH RATE
        # ========================================================================
        df_cluster0 = df[df['cluster_id'] == 0].sort_values('year')
        
        if len(df_cluster0) >= 2:
            years_c0 = df_cluster0['year'].values
            areas_c0 = df_cluster0['area_km2'].values
            
            # Calculate growth rates for consecutive years
            cluster0_growth_rates = []
            
            for i in range(len(years_c0) - 1):
                if years_c0[i+1] - years_c0[i] == 1:
                    growth_rate = (areas_c0[i+1] - areas_c0[i]) / areas_c0[i]
                    cluster0_growth_rates.append(growth_rate)
            
            if len(cluster0_growth_rates) > 0:
                cluster0_statistics.append({
                    'city': city_name,
                    'cluster0_mean_growth_rate': np.mean(cluster0_growth_rates),
                    'cluster0_median_growth_rate': np.median(cluster0_growth_rates),
                    'cluster0_std_growth_rate': np.std(cluster0_growth_rates),
                    'cluster0_initial_area': areas_c0[0],
                    'cluster0_final_area': areas_c0[-1],
                    'n_growth_observations': len(cluster0_growth_rates)
                })
                print(f"  Cluster 0 mean growth rate: {np.mean(cluster0_growth_rates):.4f} ({np.mean(cluster0_growth_rates)*100:.2f}%)")
        
        # ========================================================================
        # SATELLITE CLUSTERS (EXCLUDING CLUSTER 0)
        # ========================================================================
        df_growth = df[df['cluster_id'] != 0].copy()
        df_growth = df_growth.sort_values(['cluster_id', 'year'])
        
        # Get unique cluster IDs
        cluster_ids = df_growth['cluster_id'].unique()
        
        city_growth_data = []
        
        for cluster_id in cluster_ids:
            # Get all years for this cluster
            cluster_df = df_growth[df_growth['cluster_id'] == cluster_id].sort_values('year')
            
            if len(cluster_df) < 2:
                continue
            
            years = cluster_df['year'].values
            areas = cluster_df['area_km2'].values
            
            # Calculate growth rates for consecutive years
            cluster_growth_rates = []
            cluster_areas_for_growth = []
            
            for i in range(len(years) - 1):
                # Check if years are consecutive
                if years[i+1] - years[i] == 1:
                    # Calculate growth rate
                    growth_rate = (areas[i+1] - areas[i]) / areas[i]
                    
                    # Check if cluster was absorbed
                    next_year_data = df[(df['year'] == years[i+1]) & (df['cluster_id'] == cluster_id)]
                    
                    if len(next_year_data) > 0:
                        absorbed = next_year_data['absorbed_clusters'].values[0]
                        if pd.isna(absorbed) or absorbed == '':
                            cluster_growth_rates.append(growth_rate)
                            cluster_areas_for_growth.append(areas[i])
                        else:
                            break
            
            # Calculate average growth rate for this cluster
            if len(cluster_growth_rates) > 0:
                avg_growth_rate = ((cluster_areas_for_growth[-1]/cluster_areas_for_growth[0])**(1/len(cluster_areas_for_growth))-1)*100
                avg_area = np.mean(cluster_areas_for_growth)
                
                cluster_data = {
                    'city': city_name,
                    'cluster_id': cluster_id,
                    'avg_growth_rate': avg_growth_rate,
                    'avg_area': avg_area,
                    'n_observations': len(cluster_growth_rates),
                    'first_year': years[0],
                    'last_tracked_year': years[len(cluster_growth_rates)],
                    'initial_area': areas[0]
                }
                
                city_growth_data.append(cluster_data)
                all_cities_growth_data.append(cluster_data)
                
                # Add to trajectories
                for i in range(len(cluster_growth_rates)):
                    all_cities_trajectories.append({
                        'city': city_name,
                        'cluster_id': cluster_id,
                        'year': years[i],
                        'area': cluster_areas_for_growth[i],
                        'growth_rate': cluster_growth_rates[i]
                    })
        
        # Calculate city statistics for satellite clusters
        if len(city_growth_data) > 0:
            city_df = pd.DataFrame(city_growth_data)
            city_statistics.append({
                'city': city_name,
                'n_clusters': len(city_df),
                'mean_growth_rate': city_df['avg_growth_rate'].mean(),
                'median_growth_rate': city_df['avg_growth_rate'].median(),
                'std_growth_rate': city_df['avg_growth_rate'].std(),
                'mean_area': city_df['avg_area'].mean(),
                'median_area': city_df['avg_area'].median()
            })
            
            print(f"  {len(city_df)} satellite clusters analyzed")
            print(f"  Satellites mean growth rate: {city_df['avg_growth_rate'].mean():.4f} ({city_df['avg_growth_rate'].mean()*100:.2f}%)")

# Create DataFrames
all_growth_df = pd.DataFrame(all_cities_growth_data)
all_trajectories_df = pd.DataFrame(all_cities_trajectories)
city_stats_df = pd.DataFrame(city_statistics)
cluster0_stats_df = pd.DataFrame(cluster0_statistics)

print("\n" + "="*80)
print(f"TOTAL: {len(all_growth_df)} satellite clusters analyzed across {len(city_stats_df)} cities")
print(f"Overall satellite mean growth rate: {all_growth_df['avg_growth_rate'].mean():.4f} ({all_growth_df['avg_growth_rate'].mean()*100:.2f}%)")
print(f"Cluster 0 data available for {len(cluster0_stats_df)} cities")
print(f"Overall cluster 0 mean growth rate: {cluster0_stats_df['cluster0_mean_growth_rate'].mean():.4f} ({cluster0_stats_df['cluster0_mean_growth_rate'].mean()*100:.2f}%)")
print("="*80)

print("\n" + "="*80)
print("Generating Area vs Growth Rate visualization...")
print("="*80)


print("\n" + "="*80)
print("CORRELATION ANALYSIS: GROWTH RATES vs BETA COEFFICIENT")
print("="*80)

# Normalize city names for matching
city_stats_df['City'] = city_stats_df['city'].str.replace('_', ' ').str.title()
cluster0_stats_df['City'] = cluster0_stats_df['city'].str.replace('_', ' ').str.title()
data['City'] = data['City'].str.strip()

# Merge satellite clusters statistics with beta data
merged_satellites = city_stats_df.merge(data[['City', 'beta', 'alpha', '1/z']], 
                                        on='City', how='inner')

# Merge cluster 0 statistics with beta data
merged_cluster0 = cluster0_stats_df.merge(data[['City', 'beta', 'alpha', '1/z']], 
                                          on='City', how='inner')

print(f"\nCities matched for satellite analysis: {len(merged_satellites)}")
print(f"Cities matched for cluster 0 analysis: {len(merged_cluster0)}")

# Create figure with 2 subplots for area vs growth rate
fig_area, axes_area = plt.subplots(1, 2, figsize=(16, 6))

# ============================================================================
# Plot 1: Urban Clusters Area vs Growth Rate
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import stats

# Assuming merged_satellites dataframe is already created with columns:
# 'City', 'mean_area', 'mean_growth_rate'

# Function to add city labels with minimal overlap (from your example)
def add_city_labels(ax, x_data, y_data, cities, fontsize=10):
    """Add city labels to points with basic offset to reduce overlap"""
    for i, (x, y, city) in enumerate(zip(x_data, y_data, cities)):
        # Simple offset pattern to reduce overlap
        if i % 4 == 0:
            ha, va = 'left', 'bottom'
            offset_x, offset_y = 0.01, 0.01
        elif i % 4 == 1:
            ha, va = 'right', 'top'
            offset_x, offset_y = -0.01, -0.01
        elif i % 4 == 2:
            ha, va = 'left', 'top'
            offset_x, offset_y = 0.01, -0.01
        else:
            ha, va = 'right', 'bottom'
            offset_x, offset_y = -0.01, 0.01
        
        # Normalize offsets based on data range
        x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        
        ax.annotate(city, xy=(x, y), 
                   xytext=(x + offset_x * x_range, y + offset_y * y_range),
                   fontsize=fontsize, ha=ha, va=va,
                   arrowprops=dict(arrowstyle='-', lw=0.5, color='gray', alpha=0.5))

# Main plotting code
if len(merged_satellites) >= 3:
    # Create figure with specified size
    plt.figure(figsize=(14, 8))    
    # Create a color map for cities using tab20 (same as your example)
    n_cities = len(merged_satellites)
    colors = cm.tab20(np.linspace(0, 1, n_cities))
    
    # Create a consistent color dictionary for each city
    city_colors = {city: colors[i] for i, city in enumerate(merged_satellites['City'])}
    
    # Plot points with different colors for each city
    for city in merged_satellites['City']:
        city_data = merged_satellites[merged_satellites['City'] == city]
        plt.scatter(city_data['mean_area'], 
                   city_data['mean_growth_rate'], 
                   color=city_colors[city], 
                   s=100, 
                   label=city, 
                   edgecolors='black', 
                   linewidth=0.5)
    
    # Fit and plot regression line (dashed black line like in your example)
    z = np.polyfit(merged_satellites['mean_area'], merged_satellites['mean_growth_rate'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(merged_satellites['mean_area'].min(), 
                        merged_satellites['mean_area'].max(), 100)
    plt.plot(x_line, p(x_line), '--', color='black', 
            alpha=0.7, linewidth=2, 
            label=f'Linear fit: y={z[0]:.2f}x + {z[1]:.2f}')
    
    # Add city labels using the same function as your example
    add_city_labels(plt.gca(), merged_satellites['mean_area'], 
                   merged_satellites['mean_growth_rate'], 
                   merged_satellites['City'])
    
    # Calculate correlation statistics
    pearson_r, pearson_p = stats.pearsonr(merged_satellites['mean_area'], 
                                          merged_satellites['mean_growth_rate'])
    spearman_r, spearman_p = stats.spearmanr(merged_satellites['mean_area'], 
                                             merged_satellites['mean_growth_rate'])
    
    # Add correlation statistics text box (matching your example style)
    textstr = f'Pearson r = {pearson_r:.2f} (p = {pearson_p:.2f})\n' + \
              f'Spearman ρ = {spearman_r:.2f} (p = {spearman_p:.2f})'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    # Set labels and title
    plt.gca().set_xlabel('Mean Area (km²)', fontsize=12)
    plt.gca().set_ylabel('Mean Growth Rate', fontsize=12)
    plt.gca().set_title('Urban Clusters: Mean Area vs Mean Growth Rate', 
                 fontsize=14, fontweight='bold')
    
    # Add grid (matching your example)
    plt.grid(True, alpha=0.3)
    
    # Legend can be added or commented out based on preference
    # If you want the legend like in some of your plots:
    # ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    # Or without legend for cleaner look (as in some of your examples):
    # Comment out the legend
    
    # Use tight layout
    plt.tight_layout()
    
    # Save figure with high DPI
    output_path_area = f'{output_dir}\\satellite_growth_vs_area.png'
    plt.savefig(output_path_area, dpi=300, bbox_inches='tight')
    plt.show()



    
print("\n" + "="*80)
print("Generating satellite growth rate vs normalized distance plot...")
print("="*80)






# ============================================================================
# SATELLITE GROWTH RATE vs NORMALIZED RADIAL DISTANCE ANALYSIS
# ============================================================================
# Add this code after the urban area evolution plots

print("\n" + "="*80)
print("Analyzing satellite growth rate vs normalized radial distance...")
print("="*80)

# First, get the cluster 0 area in 2014 for each city for normalization
cluster0_area_2015 = {}

for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        filepath = os.path.join(directory, filename)
        city_name = filename.replace('output_', '').replace('.csv', '')
        
        # Read the CSV file
        df = pd.read_csv(filepath)
        
        # Get cluster 0 area in 2014
        df_2015_c0 = df[(df['year'] == 2015) & (df['cluster_id'] == 0)]
        if len(df_2015_c0) > 0:
            # Use square root of area as characteristic length scale
            cluster0_area_2015[city_name] = np.sqrt(df_2015_c0['area_km2'].values[0])
        else:
            # If 2014 not available, use first available year
            df_c0 = df[df['cluster_id'] == 0].sort_values('year')
            if len(df_c0) > 0:
                cluster0_area_2015[city_name] = np.sqrt(df_c0.iloc[0]['area_km2'])
                print(f"  Warning: Using year {df_c0.iloc[0]['year']} instead of 2014 for {city_name}")

print(f"\nCluster 0 normalization factors (sqrt of area) calculated for {len(cluster0_area_2015)} cities")

# Merge growth rate data with radial distance
# We'll use the all_growth_df which has avg_growth_rate for each satellite cluster
print(f"\nSatellite clusters with growth data: {len(all_growth_df)}")

# Now add normalized distance to the growth data
growth_distance_data = []

for idx, growth_row in all_growth_df.iterrows():
    city_name = growth_row['city']
    cluster_id = growth_row['cluster_id']
    
    # Get normalization factor for this city
    if city_name not in cluster0_area_2015:
        continue
    
    norm_factor = cluster0_area_2015[city_name]
    
    # Read the city's data to get radial distance
    filepath = os.path.join(directory, f'output_{city_name}.csv')
    if not os.path.exists(filepath):
        continue
    
    df_city = pd.read_csv(filepath)
    
    # Check if radial_distance_km column exists
    if 'radial_distance_km' not in df_city.columns:
        print(f"  Warning: 'radial_distance_km' not found in {city_name}, skipping...")
        continue
    
    # Get the cluster's data
    cluster_data = df_city[(df_city['cluster_id'] == cluster_id) & 
                           (df_city['cluster_id'] != 0)].copy()
    
    if len(cluster_data) == 0:
        continue
    
    # Use the average radial distance across all years for this cluster
    avg_radial_distance = cluster_data['radial_distance_km'].mean()
    
    if pd.notna(avg_radial_distance):
        # Normalize distance by sqrt(cluster0_area)
        normalized_distance = avg_radial_distance / norm_factor
        
        growth_distance_data.append({
            'city': city_name,
            'cluster_id': cluster_id,
            'avg_growth_rate': growth_row['avg_growth_rate'],
            'avg_area': growth_row['avg_area'],
            'avg_radial_distance_km': avg_radial_distance,
            'normalized_distance': normalized_distance,
            'cluster0_area_2015_sqrt': norm_factor,
            'n_observations': growth_row['n_observations']
        })

growth_distance_df = pd.DataFrame(growth_distance_data)

print(f"\nCollected {len(growth_distance_df)} satellite clusters with both growth rate and distance data")
print(f"Cities included: {len(growth_distance_df['city'].unique())}")

# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================


if len(growth_distance_df) > 2:
    print("\n" + "-"*80)
    print("CORRELATION ANALYSIS: SATELLITE GROWTH RATE vs NORMALIZED DISTANCE")
    print("-"*80)
    
    # Remove any infinite or NaN values
    growth_distance_clean = growth_distance_df[
        np.isfinite(growth_distance_df['avg_growth_rate']) & 
        np.isfinite(growth_distance_df['normalized_distance'])
    ].copy()
    
    print(f"Clean data points: {len(growth_distance_clean)}")
    
    if len(growth_distance_clean) > 2:
        # Calculate correlations
        spearman_r, spearman_p = stats.spearmanr(
            growth_distance_clean['normalized_distance'], 
            growth_distance_clean['avg_growth_rate']
        )
        pearson_r, pearson_p = stats.pearsonr(
            growth_distance_clean['normalized_distance'], 
            growth_distance_clean['avg_growth_rate']
        )
        
        print(f"Urban Clusters Growth Rate vs Normalized Distance:")
        print(f"  Spearman rho = {spearman_r:.4f}, p = {spearman_p:.6f}")
        print(f"  Pearson r = {pearson_r:.4f}, p = {pearson_p:.6f}")
    
    
    
    # ========================================================================
    # ADDITIONAL ANALYSIS: BY CITY
    # ========================================================================
    print("\n" + "-"*80)
    print("CITY-SPECIFIC CORRELATIONS")
    print("-"*80)
    
    city_correlations = []
        # Get unique cities for color mapping
    unique_cities_rad = growth_distance_clean['city'].unique()
    colors_rad = plt.cm.tab20(np.linspace(0, 1, len(unique_cities_rad)))
    city_color_map_rad = dict(zip(unique_cities_rad, colors_rad))
    for city in unique_cities_rad:
        city_data = growth_distance_clean[growth_distance_clean['city'] == city]
        
        if len(city_data) >= 3:
            try:
                pearson_r_city, pearson_p_city = stats.pearsonr(
                    city_data['normalized_distance'], 
                    city_data['avg_growth_rate']
                )
                spearman_r_city, spearman_p_city = stats.spearmanr(
                    city_data['normalized_distance'], 
                    city_data['avg_growth_rate']
                )
                
                city_label = city.replace('_', ' ').title()
                
                city_correlations.append({
                    'city': city_label,
                    'n_satellites': len(city_data),
                    'pearson_r': pearson_r_city,
                    'pearson_p': pearson_p_city,
                    'spearman_r': spearman_r_city,
                    'spearman_p': spearman_p_city
                })
                
                sig_marker = ''
                if pearson_p_city < 0.001:
                    sig_marker = '***'
                elif pearson_p_city < 0.01:
                    sig_marker = '**'
                elif pearson_p_city < 0.05:
                    sig_marker = '*'
                
                print(f"\n{city_label} (n={len(city_data)}):")
                print(f"  Pearson r = {pearson_r_city:.4f} (p = {pearson_p_city:.4f}) {sig_marker}")
                print(f"  Spearman ρ = {spearman_r_city:.4f} (p = {spearman_p_city:.4f})")
                
            except:
                print(f"\n{city.replace('_', ' ').title()}: Insufficient data or calculation error")
    
    # Save city-specific correlations
    if len(city_correlations) > 0:
        city_corr_df = pd.DataFrame(city_correlations)
        city_corr_df.to_csv(f'{output_dir}\\city_specific_growth_distance_correlations.csv', 
                           index=False)
        print("\n✓ City-specific growth-distance correlations saved")
    

# ============================================================================
# VISUALIZATION: ALL CITIES COMBINED
# ============================================================================

print("\n" + "="*80)
print("Generating satellite growth rate vs normalized distance plot...")
print("="*80)

if len(growth_distance_clean) > 2:
    # Create figure
    plt.figure(figsize=(16, 10))
    
    # Get unique cities for color mapping
    unique_cities_rad = growth_distance_clean['city'].unique()
    colors_rad = plt.cm.tab20(np.linspace(0, 1, len(unique_cities_rad)))
    city_color_map_rad = dict(zip(unique_cities_rad, colors_rad))
    
    # Plot each city's satellites with different colors
    for city in unique_cities_rad:
        city_data = growth_distance_clean[growth_distance_clean['city'] == city]
        city_label = city.replace('_', ' ').title()
        
        plt.scatter(city_data['normalized_distance'], 
                   city_data['avg_growth_rate'],
                   s=80, alpha=0.6, 
                   color=city_color_map_rad[city],
                   edgecolors='black', linewidth=0.5,
                   label=city_label)
    
    # Fit and plot overall regression line
    z = np.polyfit(growth_distance_clean['normalized_distance'], 
                   growth_distance_clean['avg_growth_rate'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(growth_distance_clean['normalized_distance'].min(), 
                        growth_distance_clean['normalized_distance'].max(), 100)
    plt.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=3, 
            label='Overall linear trend', zorder=100)
    
    # Calculate correlation statistics
    pearson_r, pearson_p = stats.pearsonr(growth_distance_clean['normalized_distance'], 
                                           growth_distance_clean['avg_growth_rate'])
    spearman_r, spearman_p = stats.spearmanr(growth_distance_clean['normalized_distance'], 
                                              growth_distance_clean['avg_growth_rate'])
    
    # Create statistics text box
    stats_text = f'Correlation Analysis\n'
    stats_text += f'{"="*35}\n\n'
    stats_text += f'N = {len(growth_distance_clean)} satellites\n'
    stats_text += f'Cities = {len(unique_cities_rad)}\n\n'
    stats_text += f'Pearson Correlation:\n'
    stats_text += f'  r = {pearson_r:.2f}\n'
    stats_text += f'  p = {pearson_p:.2f}\n'
    if pearson_p < 0.001:
        stats_text += f'  Sig: *** (p < 0.001)\n\n'
    elif pearson_p < 0.01:
        stats_text += f'  Sig: ** (p < 0.01)\n\n'
    elif pearson_p < 0.05:
        stats_text += f'  Sig: * (p < 0.05)\n\n'
    else:
        stats_text += f'  Not significant\n\n'
    
    stats_text += f'Spearman Correlation:\n'
    stats_text += f'  ρ = {spearman_r:.2f}\n'
    stats_text += f'  p = {spearman_p:.2f}\n'
    if spearman_p < 0.001:
        stats_text += f'  Sig: ***'
    elif spearman_p < 0.01:
        stats_text += f'  Sig: **'
    elif spearman_p < 0.05:
        stats_text += f'  Sig: *'
    else:
        stats_text += f'  Not significant'
    
    plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, 
                     edgecolor='black', linewidth=2),
            family='monospace')
    
   
    
   
    
    # Add zero growth reference line
    plt.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    
    plt.gca().set_xlabel('Normalized Radial Distance\n(Distance / √Cluster0_Area_2014)', 
                  fontsize=14, fontweight='bold')
    plt.gca().set_ylabel('Urban Cluster Average Growth Rate', fontsize=14, fontweight='bold')
    plt.gca().set_title('Urban Cluster Growth Rate vs Distance from Urban Core\n(All Clusters from All Cities)', 
                 fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
             fontsize=9, framealpha=0.9, ncol=1)
    
    plt.tight_layout()
    
    # Save the figure
    output_path_radial = f'{output_dir}\\satellite_growth_vs_normalized_distance.png'
    plt.savefig(output_path_radial, dpi=300, bbox_inches='tight')
    print(f"\n✓ Satellite growth rate vs normalized distance plot saved to: {output_path_radial}")
    
    plt.show()



    if len(growth_distance_clean) > 2:
    # -------------------------------------------------
    # 1. Aggregate to ONE point per city
    # -------------------------------------------------
        city_agg = (
        growth_distance_clean
        .groupby('city')
        .agg(
            normalized_distance=('normalized_distance', 'mean'),
            avg_growth_rate=('avg_growth_rate', 'mean')
        )
        .reset_index()
    )

    # -------------------------------------------------
    # 2. Create figure (same size as before)
    # -------------------------------------------------
    plt.figure(figsize=(16, 10))

    # -------------------------------------------------
    # 3. Colour mapping (one colour per city)
    # -------------------------------------------------
    unique_cities = city_agg['city'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_cities)))
    city_color_map = dict(zip(unique_cities, colors))

    # -------------------------------------------------
    # 4. Plot ONE point per city
    # -------------------------------------------------
    for city in unique_cities:
        row = city_agg[city_agg['city'] == city].iloc[0]
        city_label = city.replace('_', ' ').title()

        plt.scatter(row['normalized_distance'],
                    row['avg_growth_rate'],
                    s=120,                     # a bit larger – only one point per city
                    alpha=0.8,
                    color=city_color_map[city],
                    edgecolors='black', linewidth=0.8,
                    label=city_label)
        
       
    


    # -------------------------------------------------
    # 5. Overall regression line (now on the aggregated data)
    # -------------------------------------------------
    z = np.polyfit(city_agg['normalized_distance'],
                   city_agg['avg_growth_rate'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(city_agg['normalized_distance'].min(),
                         city_agg['normalized_distance'].max(), 100)
    plt.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=3,
             label='Overall linear trend', zorder=100)

    # -------------------------------------------------
    # 6. Correlation statistics (on aggregated data)
    # -------------------------------------------------
    stats_text=''
    pearson_r, pearson_p = stats.pearsonr(city_agg['normalized_distance'],
                                           city_agg['avg_growth_rate'])
    spearman_r, spearman_p = stats.spearmanr(city_agg['normalized_distance'],
                                            city_agg['avg_growth_rate'])

    stats_text += f'Pearson r = {pearson_r:.4f} (p = {pearson_p:.4f})\n'
    stats_text += f'Spearman ρ  = {spearman_r:.4f} (p = {spearman_p:.4f})\n'
    # if pearson_p < 0.001:
    #     stats_text += f'  Sig: *** (p < 0.001)\n\n'
    # elif pearson_p < 0.01:
    #     stats_text += f'  Sig: ** (p < 0.01)\n\n'
    # elif pearson_p < 0.05:
    #     stats_text += f'  Sig: * (p < 0.05)\n\n'
    # else:
    #     stats_text += f'  Not significant\n\n'


    # if spearman_p < 0.001:
    #     stats_text += f'  Sig: ***'
    # elif spearman_p < 0.01:
    #     stats_text += f'  Sig: **'
    # elif spearman_p < 0.05:
    #     stats_text += f'  Sig: *'
    # else:
    #     stats_text += f'  Not significant'
    add_city_labels(plt.gca(),city_agg['normalized_distance'],city_agg['avg_growth_rate'],data['City'])
    plt.text(0.98, 0.98, stats_text,
             transform=plt.gca().transAxes,
             fontsize=11, verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat',
                       alpha=0.9, edgecolor='black', linewidth=2),
             family='monospace')

    # -------------------------------------------------
    # 7. Cosmetic elements (unchanged)
    # -------------------------------------------------
  
    plt.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)

    plt.gca().set_xlabel(r"$\bar{r}_{clusters}$",
                         fontsize=14, fontweight='bold')
    plt.gca().set_ylabel('Urban Cluster Average Growth Rate',
                         fontsize=14, fontweight='bold')
    plt.gca().set_title('Urban Cluster Growth Rate vs Distance from Urban Core\n'
                        '(One averaged point per city)',
                        fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')

    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
            #    fontsize=9, framealpha=0.9, ncol=1)

    plt.tight_layout()

    # -------------------------------------------------
    # 8. Save & show
    # -------------------------------------------------
    output_path_radial = f'{output_dir}\\city_averaged_growth_vs_normalized_distance.png'
    plt.savefig(output_path_radial, dpi=300, bbox_inches='tight')
    print(f"\n✓ City-averaged growth rate vs normalized distance plot saved to: {output_path_radial}")

    plt.show()