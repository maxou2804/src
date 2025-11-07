import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import os

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
                avg_growth_rate = np.mean(cluster_growth_rates)
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

# ============================================================================
# CORRELATION ANALYSIS: GROWTH RATES vs BETA COEFFICIENT
# ============================================================================
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

# ============================================================================
# CALCULATE CORRELATIONS
# ============================================================================

results_text = []
results_text.append("="*80)
results_text.append("CORRELATION ANALYSIS: GROWTH RATES vs SCALING COEFFICIENTS")
results_text.append("="*80)
results_text.append("")
results_text.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
results_text.append(f"Number of cities analyzed (Urban clusters): {len(merged_satellites)}")
results_text.append(f"Number of cities analyzed (LCC): {len(merged_cluster0)}")
results_text.append("")

# ============================================================================
# 1. SATELLITE CLUSTERS vs BETA
# ============================================================================
if len(merged_satellites) >= 3:
    print("\n" + "-"*80)
    print("SATELLITE CLUSTERS ANALYSIS")
    print("-"*80)
    
    spearman_sat, spearman_sat_p = stats.spearmanr(merged_satellites['mean_growth_rate'], 
                                                     merged_satellites['beta'])
    pearson_sat, pearson_sat_p = stats.pearsonr(merged_satellites['mean_growth_rate'], 
                                                  merged_satellites['beta'])
    
    print(f"\nSatellite Growth Rate vs Beta:")
    print(f"  Spearman rho = {spearman_sat:.4f}, p = {spearman_sat_p:.6f}")
    print(f"  Pearson r = {pearson_sat:.4f}, p = {pearson_sat_p:.6f}")
    
    results_text.append("="*80)
    results_text.append("1. URBAN CLUSTERS: MEAN GROWTH RATE vs BETA COEFFICIENT")
    results_text.append("="*80)
    results_text.append("")
    results_text.append("Spearman Correlation:")
    results_text.append(f"   rho = {spearman_sat:.6f}")
    results_text.append(f"  p-value = {spearman_sat_p:.6f}")
    results_text.append(f"  Significance: {'***' if spearman_sat_p < 0.001 else '**' if spearman_sat_p < 0.01 else '*' if spearman_sat_p < 0.05 else 'Not significant (p >= 0.05)'}")
    results_text.append("")
    results_text.append("Pearson Correlation:")
    results_text.append(f"  r = {pearson_sat:.6f}")
    results_text.append(f"  p-value = {pearson_sat_p:.6f}")
    results_text.append(f"  Significance: {'***' if pearson_sat_p < 0.001 else '**' if pearson_sat_p < 0.01 else '*' if pearson_sat_p < 0.05 else 'Not significant (p >= 0.05)'}")
    results_text.append("")
    results_text.append("Interpretation:")
    if pearson_sat_p < 0.05:
        if pearson_sat > 0:
            results_text.append(f"  There is a significant POSITIVE correlation (r={pearson_sat:.4f}).")
            results_text.append(f"  Cities with higher beta coefficients have faster-growing satellite clusters.")
        else:
            results_text.append(f"  There is a significant NEGATIVE correlation (r={pearson_sat:.4f}).")
            results_text.append(f"  Cities with higher beta coefficients have slower-growing satellite clusters.")
    else:
        results_text.append(f"  No significant linear correlation found (p={pearson_sat_p:.4f}).")
    results_text.append("")

# ============================================================================
# 2. CLUSTER 0 vs BETA
# ============================================================================
if len(merged_cluster0) >= 3:
    print("\n" + "-"*80)
    print("LCC ANALYSIS")
    print("-"*80)
    
    spearman_c0, spearman_c0_p = stats.spearmanr(merged_cluster0['cluster0_mean_growth_rate'], 
                                                   merged_cluster0['beta'])
    pearson_c0, pearson_c0_p = stats.pearsonr(merged_cluster0['cluster0_mean_growth_rate'], 
                                                merged_cluster0['beta'])
    
    print(f"\nCluster 0 Growth Rate vs Beta:")
    print(f"  Spearman rho = {spearman_c0:.4f}, p = {spearman_c0_p:.6f}")
    print(f"  Pearson r = {pearson_c0:.4f}, p = {pearson_c0_p:.6f}")
    
    results_text.append("="*80)
    results_text.append("2. LCC: GROWTH RATE vs BETA COEFFICIENT")
    results_text.append("="*80)
    results_text.append("")
    results_text.append("Spearman Correlation:")
    results_text.append(f"  rho  = {spearman_c0:.6f}")
    results_text.append(f"  p-value = {spearman_c0_p:.6f}")
    results_text.append(f"  Significance: {'***' if spearman_c0_p < 0.001 else '**' if spearman_c0_p < 0.01 else '*' if spearman_c0_p < 0.05 else 'Not significant (p >= 0.05)'}")
    results_text.append("")
    results_text.append("Pearson Correlation:")
    results_text.append(f"  r = {pearson_c0:.6f}")
    results_text.append(f"  p-value = {pearson_c0_p:.6f}")
    results_text.append(f"  Significance: {'***' if pearson_c0_p < 0.001 else '**' if pearson_c0_p < 0.01 else '*' if pearson_c0_p < 0.05 else 'Not significant (p >= 0.05)'}")
    results_text.append("")
    results_text.append("Interpretation:")
    if pearson_c0_p < 0.05:
        if pearson_c0 > 0:
            results_text.append(f"  There is a significant POSITIVE correlation (r={pearson_c0:.4f}).")
            results_text.append(f"  Cities with higher beta (more dispersed) have faster-growing main urban cores.")
        else:
            results_text.append(f"  There is a significant NEGATIVE correlation (r={pearson_c0:.4f}).")
            results_text.append(f"  Cities with higher beta (more dispersed) have slower-growing main urban cores.")
    else:
        results_text.append(f"  No significant linear correlation found (p={pearson_c0_p:.4f}).")
    results_text.append("")
    
    # Calculate correlations with alpha and 1/z for cluster 0
    spearman_c0_alpha, spearman_c0_alpha_p = stats.spearmanr(merged_cluster0['cluster0_mean_growth_rate'], 
                                                               merged_cluster0['alpha'])
    pearson_c0_alpha, pearson_c0_alpha_p = stats.pearsonr(merged_cluster0['cluster0_mean_growth_rate'], 
                                                            merged_cluster0['alpha'])
    
    spearman_c0_z, spearman_c0_z_p = stats.spearmanr(merged_cluster0['cluster0_mean_growth_rate'], 
                                                       merged_cluster0['1/z'])
    pearson_c0_z, pearson_c0_z_p = stats.pearsonr(merged_cluster0['cluster0_mean_growth_rate'], 
                                                    merged_cluster0['1/z'])

# ============================================================================
# 3. ADDITIONAL CORRELATIONS FOR SATELLITES
# ============================================================================
if len(merged_satellites) >= 3:
    spearman_sat_alpha, spearman_sat_alpha_p = stats.spearmanr(merged_satellites['mean_growth_rate'], 
                                                                 merged_satellites['alpha'])
    pearson_sat_alpha, pearson_sat_alpha_p = stats.pearsonr(merged_satellites['mean_growth_rate'], 
                                                              merged_satellites['alpha'])
    
    spearman_sat_z, spearman_sat_z_p = stats.spearmanr(merged_satellites['mean_growth_rate'], 
                                                         merged_satellites['1/z'])
    pearson_sat_z, pearson_sat_z_p = stats.pearsonr(merged_satellites['mean_growth_rate'], 
                                                      merged_satellites['1/z'])
    
    results_text.append("="*80)
    results_text.append("3. SATELLITE CLUSTERS: ADDITIONAL CORRELATIONS")
    results_text.append("="*80)
    results_text.append("")
    results_text.append("Satellites vs Alpha:")
    results_text.append(f"  Spearman rho = {spearman_sat_alpha:.6f}, p = {spearman_sat_alpha_p:.6f}")
    results_text.append(f"  Pearson r = {pearson_sat_alpha:.6f}, p = {pearson_sat_alpha_p:.6f}")
    results_text.append("")
    results_text.append("Satellites vs 1/z:")
    results_text.append(f"  Spearman rho = {spearman_sat_z:.6f}, p = {spearman_sat_z_p:.6f}")
    results_text.append(f"  Pearson r = {pearson_sat_z:.6f}, p = {pearson_sat_z_p:.6f}")
    results_text.append("")

if len(merged_cluster0) >= 3:
    results_text.append("="*80)
    results_text.append("4. CLUSTER 0: ADDITIONAL CORRELATIONS")
    results_text.append("="*80)
    results_text.append("")
    results_text.append("Cluster 0 vs Alpha:")
    results_text.append(f"  Spearman rho = {spearman_c0_alpha:.6f}, p = {spearman_c0_alpha_p:.6f}")
    results_text.append(f"  Pearson r = {pearson_c0_alpha:.6f}, p = {pearson_c0_alpha_p:.6f}")
    results_text.append("")
    results_text.append("Cluster 0 vs 1/z:")
    results_text.append(f"  Spearman rho = {spearman_c0_z:.6f}, p = {spearman_c0_z_p:.6f}")
    results_text.append(f"  Pearson r = {pearson_c0_z:.6f}, p = {pearson_c0_z_p:.6f}")
    results_text.append("")
# ============================================================================
# ADDITIONAL CORRELATION ANALYSIS: AREA vs GROWTH RATE
# ============================================================================
# Add this code after the "3. ADDITIONAL CORRELATIONS FOR SATELLITES" section
# and before the "DATA SUMMARY" section

print("\n" + "-"*80)
print("AREA vs GROWTH RATE CORRELATION ANALYSIS")
print("-"*80)

# ============================================================================
# 5. SATELLITE CLUSTERS: AREA vs GROWTH RATE
# ============================================================================
if len(merged_satellites) >= 3:
    spearman_sat_area_growth, spearman_sat_area_growth_p = stats.spearmanr(
        merged_satellites['mean_area'], 
        merged_satellites['mean_growth_rate']
    )
    pearson_sat_area_growth, pearson_sat_area_growth_p = stats.pearsonr(
        merged_satellites['mean_area'], 
        merged_satellites['mean_growth_rate']
    )
    
    print(f"\nSatellite Area vs Growth Rate:")
    print(f"  Spearman rho = {spearman_sat_area_growth:.4f}, p = {spearman_sat_area_growth_p:.6f}")
    print(f"  Pearson r = {pearson_sat_area_growth:.4f}, p = {pearson_sat_area_growth_p:.6f}")
    
    results_text.append("="*80)
    results_text.append("5. SATELLITE CLUSTERS: AREA vs GROWTH RATE")
    results_text.append("="*80)
    results_text.append("")
    results_text.append("Spearman Correlation:")
    results_text.append(f"  rho = {spearman_sat_area_growth:.6f}")
    results_text.append(f"  p-value = {spearman_sat_area_growth_p:.6f}")
    results_text.append(f"  Significance: {'***' if spearman_sat_area_growth_p < 0.001 else '**' if spearman_sat_area_growth_p < 0.01 else '*' if spearman_sat_area_growth_p < 0.05 else 'Not significant (p >= 0.05)'}")
    results_text.append("")
    results_text.append("Pearson Correlation:")
    results_text.append(f"  r = {pearson_sat_area_growth:.6f}")
    results_text.append(f"  p-value = {pearson_sat_area_growth_p:.6f}")
    results_text.append(f"  Significance: {'***' if pearson_sat_area_growth_p < 0.001 else '**' if pearson_sat_area_growth_p < 0.01 else '*' if pearson_sat_area_growth_p < 0.05 else 'Not significant (p >= 0.05)'}")
    results_text.append("")
    results_text.append("Interpretation:")
    if pearson_sat_area_growth_p < 0.05:
        if pearson_sat_area_growth > 0:
            results_text.append(f"  There is a significant POSITIVE correlation (r={pearson_sat_area_growth:.4f}).")
            results_text.append(f"  Larger satellite clusters tend to grow faster.")
        else:
            results_text.append(f"  There is a significant NEGATIVE correlation (r={pearson_sat_area_growth:.4f}).")
            results_text.append(f"  Larger satellite clusters tend to grow slower.")
    else:
        results_text.append(f"  No significant linear correlation found (p={pearson_sat_area_growth_p:.4f}).")
    results_text.append("")

# ============================================================================
# 6. CLUSTER 0: AREA vs GROWTH RATE
# ============================================================================
if len(merged_cluster0) >= 3:
    # Use initial area, final area, or average area - let's use initial area
    spearman_c0_area_growth, spearman_c0_area_growth_p = stats.spearmanr(
        merged_cluster0['cluster0_initial_area'], 
        merged_cluster0['cluster0_mean_growth_rate']
    )
    pearson_c0_area_growth, pearson_c0_area_growth_p = stats.pearsonr(
        merged_cluster0['cluster0_initial_area'], 
        merged_cluster0['cluster0_mean_growth_rate']
    )
    
    print(f"\nCluster 0 Initial Area vs Growth Rate:")
    print(f"  Spearman rho = {spearman_c0_area_growth:.4f}, p = {spearman_c0_area_growth_p:.6f}")
    print(f"  Pearson r = {pearson_c0_area_growth:.4f}, p = {pearson_c0_area_growth_p:.6f}")
    
    results_text.append("="*80)
    results_text.append("6.LCC: INITIAL AREA vs GROWTH RATE")
    results_text.append("="*80)
    results_text.append("")
    results_text.append("Spearman Correlation:")
    results_text.append(f"  rho = {spearman_c0_area_growth:.6f}")
    results_text.append(f"  p-value = {spearman_c0_area_growth_p:.6f}")
    results_text.append(f"  Significance: {'***' if spearman_c0_area_growth_p < 0.001 else '**' if spearman_c0_area_growth_p < 0.01 else '*' if spearman_c0_area_growth_p < 0.05 else 'Not significant (p >= 0.05)'}")
    results_text.append("")
    results_text.append("Pearson Correlation:")
    results_text.append(f"  r = {pearson_c0_area_growth:.6f}")
    results_text.append(f"  p-value = {pearson_c0_area_growth_p:.6f}")
    results_text.append(f"  Significance: {'***' if pearson_c0_area_growth_p < 0.001 else '**' if pearson_c0_area_growth_p < 0.01 else '*' if pearson_c0_area_growth_p < 0.05 else 'Not significant (p >= 0.05)'}")
    results_text.append("")
    results_text.append("Interpretation:")
    if pearson_c0_area_growth_p < 0.05:
        if pearson_c0_area_growth > 0:
            results_text.append(f"  There is a significant POSITIVE correlation (r={pearson_c0_area_growth:.4f}).")
            results_text.append(f"  Cities with larger main urban cores tend to have faster core growth rates.")
        else:
            results_text.append(f"  There is a significant NEGATIVE correlation (r={pearson_c0_area_growth:.4f}).")
            results_text.append(f"  Cities with larger main urban cores tend to have slower core growth rates.")
    else:
        results_text.append(f"  No significant linear correlation found (p={pearson_c0_area_growth_p:.4f}).")
    results_text.append("")
    
    # Also check with final area
    spearman_c0_final_area_growth, spearman_c0_final_area_growth_p = stats.spearmanr(
        merged_cluster0['cluster0_final_area'], 
        merged_cluster0['cluster0_mean_growth_rate']
    )
    pearson_c0_final_area_growth, pearson_c0_final_area_growth_p = stats.pearsonr(
        merged_cluster0['cluster0_final_area'], 
        merged_cluster0['cluster0_mean_growth_rate']
    )
    
    print(f"\nCluster 0 Final Area vs Growth Rate:")
    print(f"  Spearman rho = {spearman_c0_final_area_growth:.4f}, p = {spearman_c0_final_area_growth_p:.6f}")
    print(f"  Pearson r = {pearson_c0_final_area_growth:.4f}, p = {pearson_c0_final_area_growth_p:.6f}")
    
    results_text.append("="*80)
    results_text.append("7.LCC: FINAL AREA vs GROWTH RATE")
    results_text.append("="*80)
    results_text.append("")
    results_text.append("Spearman Correlation:")
    results_text.append(f"  rho = {spearman_c0_final_area_growth:.6f}")
    results_text.append(f"  p-value = {spearman_c0_final_area_growth_p:.6f}")
    results_text.append(f"  Significance: {'***' if spearman_c0_final_area_growth_p < 0.001 else '**' if spearman_c0_final_area_growth_p < 0.01 else '*' if spearman_c0_final_area_growth_p < 0.05 else 'Not significant (p >= 0.05)'}")
    results_text.append("")
    results_text.append("Pearson Correlation:")
    results_text.append(f"  r = {pearson_c0_final_area_growth:.6f}")
    results_text.append(f"  p-value = {pearson_c0_final_area_growth_p:.6f}")
    results_text.append(f"  Significance: {'***' if pearson_c0_final_area_growth_p < 0.001 else '**' if pearson_c0_final_area_growth_p < 0.01 else '*' if pearson_c0_final_area_growth_p < 0.05 else 'Not significant (p >= 0.05)'}")
    results_text.append("")
    results_text.append("Interpretation:")
    if pearson_c0_final_area_growth_p < 0.05:
        if pearson_c0_final_area_growth > 0:
            results_text.append(f"  There is a significant POSITIVE correlation (r={pearson_c0_final_area_growth:.4f}).")
            results_text.append(f"  Cities with larger final main urban core areas had faster growth rates.")
        else:
            results_text.append(f"  There is a significant NEGATIVE correlation (r={pearson_c0_final_area_growth:.4f}).")
            results_text.append(f"  Cities with larger final main urban core areas had slower growth rates.")
    else:
        results_text.append(f"  No significant linear correlation found (p={pearson_c0_final_area_growth_p:.4f}).")
    results_text.append("")

# ============================================================================
# DATA SUMMARY
# ============================================================================
results_text.append("="*80)
results_text.append("DATA SUMMARY")
results_text.append("="*80)
results_text.append("")

# Merge all data for comprehensive table
full_merge = merged_satellites.merge(merged_cluster0[['City', 'cluster0_mean_growth_rate']], 
                                     on='City', how='outer')

results_text.append("City-wise data:")
results_text.append("-" * 100)
results_text.append(f"{'City':<20} {'Urban clusters Growth':>18} {'LCC Growth':>18} {'Beta':>10} {'Alpha':>10} {'1/z':>10}")
results_text.append("-" * 100)

for _, row in full_merge.sort_values('mean_growth_rate', ascending=False).iterrows():
    sat_growth = row['mean_growth_rate'] if pd.notna(row['mean_growth_rate']) else float('nan')
    c0_growth = row['cluster0_mean_growth_rate'] if pd.notna(row['cluster0_mean_growth_rate']) else float('nan')
    results_text.append(f"{row['City']:<20} {sat_growth:>18.6f} {c0_growth:>18.6f} "
                       f"{row['beta']:>10.4f} {row['alpha']:>10.4f} {row['1/z']:>10.4f}")

results_text.append("")
results_text.append("="*80)
results_text.append("NOTES")
results_text.append("="*80)
results_text.append("")
results_text.append("Significance levels:")
results_text.append("  *** : p < 0.001 (highly significant)")
results_text.append("  **  : p < 0.01  (very significant)")
results_text.append("  *   : p < 0.05  (significant)")
results_text.append("")
results_text.append("Spearman correlation: Non-parametric, rank-based correlation")
results_text.append("Pearson correlation: Parametric, assumes linear relationship")
results_text.append("")
results_text.append("Cluster 0: The largest connected component (main urban core)")
results_text.append("Urban clusters: All other clusters excluding cluster 0")

# Save to file
with open(f'{output_dir}\\correlation_growth_vs_beta_complete.txt', 'w') as f:
    f.write('\n'.join(results_text))


# ============================================================================
# VISUALIZATION: GROWTH RATES vs BETA COEFFICIENT
# ============================================================================
# Add this code after the correlation analysis section, before the data saving

print("\n" + "="*80)
print("Generating Growth Rate vs Beta visualization...")
print("="*80)

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ============================================================================
# Plot 1: Satellite Clusters Growth Rate vs Beta
# ============================================================================
if len(merged_satellites) >= 3:
    ax1 = axes[0]
    
    # Scatter plot
    ax1.scatter(merged_satellites['beta'], 
                merged_satellites['mean_growth_rate'], 
                s=100, alpha=0.6, c='steelblue', edgecolors='black', linewidth=1)
    
    # Add city labels
    for idx, row in merged_satellites.iterrows():
        ax1.annotate(row['City'], 
                    (row['beta'], row['mean_growth_rate']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.7)
    
    # Fit and plot regression line
    z = np.polyfit(merged_satellites['beta'], merged_satellites['mean_growth_rate'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(merged_satellites['beta'].min(), merged_satellites['beta'].max(), 100)
    ax1.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label=f'Linear fit')
    
    # Add correlation statistics to plot
    pearson_r, pearson_p = stats.pearsonr(merged_satellites['mean_growth_rate'], 
                                           merged_satellites['beta'])
    spearman_r, spearman_p = stats.spearmanr(merged_satellites['mean_growth_rate'], 
                                              merged_satellites['beta'])
    
    stats_text = f'Pearson r = {pearson_r:.4f} (p = {pearson_p:.4f})\n'
    stats_text += f'Spearman rho = {spearman_r:.4f} (p = {spearman_p:.4f})'
    ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax1.set_xlabel('Beta Coefficient (β)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Growth Rate (Urban clusters)', fontsize=12, fontweight='bold')
    ax1.set_title('Urban Clusters: Average Growth Rate vs Beta Coefficient', 
                 fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend()

# ============================================================================
# Plot 2: Cluster 0 Growth Rate vs Beta
# ============================================================================
if len(merged_cluster0) >= 3:
    ax2 = axes[1]
    
    # Scatter plot
    ax2.scatter(merged_cluster0['beta'], 
                merged_cluster0['cluster0_mean_growth_rate'], 
                s=100, alpha=0.6, c='coral', edgecolors='black', linewidth=1)
    
    # Add city labels
    for idx, row in merged_cluster0.iterrows():
        ax2.annotate(row['City'], 
                    (row['beta'], row['cluster0_mean_growth_rate']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.7)
    
    # Fit and plot regression line
    z = np.polyfit(merged_cluster0['beta'], merged_cluster0['cluster0_mean_growth_rate'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(merged_cluster0['beta'].min(), merged_cluster0['beta'].max(), 100)
    ax2.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label=f'Linear fit')
    
    # Add correlation statistics to plot
    pearson_r, pearson_p = stats.pearsonr(merged_cluster0['cluster0_mean_growth_rate'], 
                                           merged_cluster0['beta'])
    spearman_r, spearman_p = stats.spearmanr(merged_cluster0['cluster0_mean_growth_rate'], 
                                              merged_cluster0['beta'])
    
    stats_text = f'Pearson r = {pearson_r:.4f} (p = {pearson_p:.4f})\n'
    stats_text += f'Spearman rho = {spearman_r:.4f} (p = {spearman_p:.4f})'
    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax2.set_xlabel('Beta Coefficient (β)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Growth Rate (LCC)', fontsize=12, fontweight='bold')
    ax2.set_title('LCC: Growth Rate vs Beta Coefficient', 
                 fontsize=14, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend()

plt.tight_layout()

# Save the figure
output_path = f'{output_dir}\\growth_rate_vs_beta_correlation.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Growth rate vs beta visualization saved to: {output_path}")

plt.show()

# ============================================================================
# ENHANCED VISUALIZATION: ADD AREA vs GROWTH RATE PLOTS
# ============================================================================

print("\n" + "="*80)
print("Generating Area vs Growth Rate visualization...")
print("="*80)

# Create figure with 2 subplots for area vs growth rate
fig_area, axes_area = plt.subplots(1, 2, figsize=(16, 6))

# ============================================================================
# Plot 1: Satellite Clusters Area vs Growth Rate
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
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Create a color map for cities using tab20 (same as your example)
    n_cities = len(merged_satellites)
    colors = cm.tab20(np.linspace(0, 1, n_cities))
    
    # Create a consistent color dictionary for each city
    city_colors = {city: colors[i] for i, city in enumerate(merged_satellites['City'])}
    
    # Plot points with different colors for each city
    for city in merged_satellites['City']:
        city_data = merged_satellites[merged_satellites['City'] == city]
        ax1.scatter(city_data['mean_area'], 
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
    ax1.plot(x_line, p(x_line), '--', color='black', 
            alpha=0.7, linewidth=2, 
            label=f'Linear fit: y={z[0]:.2f}x + {z[1]:.2f}')
    
    # Add city labels using the same function as your example
    add_city_labels(ax1, merged_satellites['mean_area'], 
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
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    # Set labels and title
    ax1.set_xlabel('Mean Area (km²)', fontsize=12)
    ax1.set_ylabel('Mean Growth Rate', fontsize=12)
    ax1.set_title('Urban Clusters: Mean Area vs Mean Growth Rate', 
                 fontsize=14, fontweight='bold')
    
    # Add grid (matching your example)
    ax1.grid(True, alpha=0.3)
    
    # Legend can be added or commented out based on preference
    # If you want the legend like in some of your plots:
    # ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    # Or without legend for cleaner look (as in some of your examples):
    # Comment out the legend
    
    # Use tight layout
    plt.tight_layout()
    
    # Save figure with high DPI
    plt.savefig('urban_clusters_area_vs_growth_styled.png', dpi=300, bbox_inches='tight')
    plt.show()

# Alternative version without individual city legend for cleaner visualization
if len(merged_satellites) >= 3:
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Create color map
    n_cities = len(merged_satellites)
    colors = cm.tab20(np.linspace(0, 1, n_cities))
    city_colors = {city: colors[i] for i, city in enumerate(merged_satellites['City'])}
    
    # Plot all points with different colors (no individual labels)
    for i, (idx, row) in enumerate(merged_satellites.iterrows()):
        ax1.scatter(row['mean_area'], 
                   row['mean_growth_rate'], 
                   color=colors[i], 
                   s=100, 
                   edgecolors='black', 
                   linewidth=0.5)
    
    # Fit and plot regression line
    z = np.polyfit(merged_satellites['mean_area'], merged_satellites['mean_growth_rate'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(merged_satellites['mean_area'].min(), 
                        merged_satellites['mean_area'].max(), 100)
    ax1.plot(x_line, p(x_line), '--', color='black', 
            alpha=0.7, linewidth=2)
    
    # Add city labels
    add_city_labels(ax1, merged_satellites['mean_area'], 
                   merged_satellites['mean_growth_rate'], 
                   merged_satellites['City'], fontsize=10)
    
    # Calculate and display correlation statistics
    pearson_r, pearson_p = stats.pearsonr(merged_satellites['mean_area'], 
                                          merged_satellites['mean_growth_rate'])
    spearman_r, spearman_p = stats.spearmanr(merged_satellites['mean_area'], 
                                             merged_satellites['mean_growth_rate'])
    
    textstr = f'Pearson r = {pearson_r:.2f} (p = {pearson_p:.2f})\n' + \
              f'Spearman ρ = {spearman_r:.2f} (p = {spearman_p:.2f})'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    # Set labels and title
    ax1.set_xlabel('Mean Area (km²)', fontsize=12)
    ax1.set_ylabel('Mean Growth Rate', fontsize=12)
    ax1.set_title('Urban Clusters: Mean Area vs Mean Growth Rate', 
                 fontsize=14, fontweight='bold')
    
    # Add grid
    ax1.grid(True, alpha=0.3)
    
    # Add linear fit equation to the plot
    equation_text = f'Linear fit: y = {z[0]:.4f}x + {z[1]:.4f}'
    ax1.text(0.98, 0.02, equation_text, transform=ax1.transAxes, fontsize=10,
            horizontalalignment='right', verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('urban_clusters_area_vs_growth_clean.png', dpi=300, bbox_inches='tight')
    plt.show()

# Print correlation statistics to console
if len(merged_satellites) >= 3:
    print("\n" + "="*80)
    print("URBAN CLUSTERS: AREA VS GROWTH RATE CORRELATION ANALYSIS")
    print("="*80)
    
    pearson_r, pearson_p = stats.pearsonr(merged_satellites['mean_area'], 
                                          merged_satellites['mean_growth_rate'])
    spearman_r, spearman_p = stats.spearmanr(merged_satellites['mean_area'], 
                                             merged_satellites['mean_growth_rate'])
    
    print("\nPEARSON CORRELATION:")
    print(f"  Correlation coefficient (r): {pearson_r:.4f}")
    print(f"  P-value: {pearson_p:.4f}")
    
    print("\nSPEARMAN CORRELATION:")
    print(f"  Correlation coefficient (ρ): {spearman_r:.4f}")
    print(f"  P-value: {spearman_p:.4f}")
    
    # Interpret correlation
    abs_r = abs(spearman_r)
    if abs_r < 0.20:
        strength = "Very weak"
    elif abs_r < 0.40:
        strength = "Weak"
    elif abs_r < 0.60:
        strength = "Moderate"
    elif abs_r < 0.80:
        strength = "Strong"
    else:
        strength = "Very strong"
    
    direction = "positive" if spearman_r > 0 else "negative"
    
    if spearman_p < 0.001:
        significance = "highly significant (p < 0.001)"
    elif spearman_p < 0.01:
        significance = "very significant (p < 0.01)"
    elif spearman_p < 0.05:
        significance = "significant (p < 0.05)"
    else:
        significance = "not significant (p ≥ 0.05)"
    
    print(f"\nInterpretation: {strength} {direction} correlation, {significance}")
    print("="*80)
# ============================================================================
# Plot 2: Cluster 0 Initial Area vs Growth Rate
# ============================================================================
if len(merged_cluster0) >= 3:
    ax2 = axes_area[1]
    
    # Scatter plot
    ax2.scatter(merged_cluster0['cluster0_initial_area'], 
                merged_cluster0['cluster0_mean_growth_rate'], 
                s=100, alpha=0.6, c='coral', edgecolors='black', linewidth=1)
    
    # Add city labels
    for idx, row in merged_cluster0.iterrows():
        ax2.annotate(row['City'], 
                    (row['cluster0_initial_area'], row['cluster0_mean_growth_rate']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.7)
    
    # Fit and plot regression line
    z = np.polyfit(merged_cluster0['cluster0_initial_area'], merged_cluster0['cluster0_mean_growth_rate'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(merged_cluster0['cluster0_initial_area'].min(), 
                        merged_cluster0['cluster0_initial_area'].max(), 100)
    ax2.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label=f'Linear fit')
    
    # Add correlation statistics to plot
    pearson_r, pearson_p = stats.pearsonr(merged_cluster0['cluster0_initial_area'], 
                                           merged_cluster0['cluster0_mean_growth_rate'])
    spearman_r, spearman_p = stats.spearmanr(merged_cluster0['cluster0_initial_area'], 
                                              merged_cluster0['cluster0_mean_growth_rate'])
    
    stats_text = f'Pearson r = {pearson_r:.2f} (p = {pearson_p:.2f})\n'
    stats_text += f'Spearman rho = {spearman_r:.2f} (p = {spearman_p:.2f})'
    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax2.set_xlabel('Initial Area (km²)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Growth Rate (LCC)', fontsize=12, fontweight='bold')
    ax2.set_title('LCC : Initial Area vs Growth Rate', 
                 fontsize=14, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend()

plt.tight_layout()

# Save the figure
output_path_area = f'{output_dir}\\area_vs_growth_rate_correlation.png'
plt.savefig(output_path_area, dpi=300, bbox_inches='tight')
print(f"\n✓ Area vs growth rate visualization saved to: {output_path_area}")

plt.show()



print("\n✓ Complete correlation analysis saved to: correlation_growth_vs_beta_complete.txt")

# Save merged datasets
full_merge.to_csv(f'{output_dir}\\city_stats_complete_with_coefficients.csv', 
                 index=False)
print("✓ Complete city statistics with coefficients saved")

cluster0_stats_df.to_csv(f'{output_dir}\\cluster0_statistics.csv', 
                        index=False)
print("✓ Cluster 0 statistics saved")

# [REST OF THE PLOTTING CODE REMAINS THE SAME AS BEFORE...]
# (The comprehensive 9-panel plot generation code continues here)
# For brevity, I'm showing the key additions. The full plotting code would be included.

print("\n" + "="*80)
print("Generating comprehensive combined plot...")
# [Previous plotting code continues...]

all_growth_df.to_csv(f'{output_dir}\\combined_growth_data_all_cities.csv', 
                    index=False)
all_trajectories_df.to_csv(f'{output_dir}\\combined_trajectories_all_cities.csv', 
                           index=False)
city_stats_df.to_csv(f'{output_dir}\\city_statistics.csv', 
                    index=False)


# ============================================================================
# CORRELATION PLOT: SATELLITE GROWTH vs CLUSTER 0 GROWTH
# ============================================================================
# Add this code after the previous comprehensive plot

print("\n" + "="*80)
print("Generating correlation plot: Satellite growth vs Cluster 0 growth...")
print("="*80)

# Merge satellite and cluster 0 statistics by city
merged_both_growth = city_stats_df.merge(
    cluster0_stats_df[['city', 'cluster0_mean_growth_rate']], 
    on='city', 
    how='inner'
)

print(f"Cities with both satellite and cluster 0 data: {len(merged_both_growth)}")

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# Get unique cities for color mapping
unique_cities = merged_both_growth['city'].unique()
colors = plt.cm.tab20(np.linspace(0, 1, len(unique_cities)))
city_color_map = dict(zip(unique_cities, colors))

# Plot each city with its own color
for idx, row in merged_both_growth.iterrows():
    city_name = row['city']
    city_label = city_name.replace('_', ' ').title()
    
    ax.scatter(row['mean_growth_rate'], 
               row['cluster0_mean_growth_rate'],
               s=200, alpha=0.7, 
               color=city_color_map[city_name],
               edgecolors='black', linewidth=2,
               label=city_label)
    
    # Add city label next to point
    ax.annotate(city_label, 
                (row['mean_growth_rate'], row['cluster0_mean_growth_rate']),
                xytext=(8, 8), textcoords='offset points',
                fontsize=9, alpha=0.8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor=city_color_map[city_name], 
                         alpha=0.3, edgecolor='none'))

# Fit and plot regression line
if len(merged_both_growth) > 1:
    z = np.polyfit(merged_both_growth['mean_growth_rate'], 
                   merged_both_growth['cluster0_mean_growth_rate'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(merged_both_growth['mean_growth_rate'].min(), 
                        merged_both_growth['mean_growth_rate'].max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=3, 
            label='Linear regression', zorder=100)
    
    # Calculate correlation statistics
    pearson_r, pearson_p = stats.pearsonr(merged_both_growth['mean_growth_rate'], 
                                           merged_both_growth['cluster0_mean_growth_rate'])
    spearman_r, spearman_p = stats.spearmanr(merged_both_growth['mean_growth_rate'], 
                                              merged_both_growth['cluster0_mean_growth_rate'])
    
    # Create statistics text box
    stats_text = f'Correlation Analysis\n'
    stats_text += f'{"="*30}\n\n'
    stats_text += f'N = {len(merged_both_growth)} cities\n\n'
    stats_text += f'Pearson Correlation:\n'
    stats_text += f'  r = {pearson_r:.4f}\n'
    stats_text += f'  p-value = {pearson_p:.6f}\n'
    if pearson_p < 0.001:
        stats_text += f'  Significance: *** (p < 0.001)\n\n'
    elif pearson_p < 0.01:
        stats_text += f'  Significance: ** (p < 0.01)\n\n'
    elif pearson_p < 0.05:
        stats_text += f'  Significance: * (p < 0.05)\n\n'
    else:
        stats_text += f'  Not significant (p ≥ 0.05)\n\n'
    
    stats_text += f'Spearman Correlation:\n'
    stats_text += f'  ρ = {spearman_r:.4f}\n'
    stats_text += f'  p-value = {spearman_p:.6f}\n'
    if spearman_p < 0.001:
        stats_text += f'  Significance: ***'
    elif spearman_p < 0.01:
        stats_text += f'  Significance: **'
    elif spearman_p < 0.05:
        stats_text += f'  Significance: *'
    else:
        stats_text += f'  Not significant'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black', linewidth=2),
            family='monospace')
    
   


# Add reference lines
ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
ax.axvline(x=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)

# Add diagonal line (y=x) to show where satellite growth = core growth
x_range = [min(merged_both_growth['mean_growth_rate'].min(), 
               merged_both_growth['cluster0_mean_growth_rate'].min()),
           max(merged_both_growth['mean_growth_rate'].max(), 
               merged_both_growth['cluster0_mean_growth_rate'].max())]
ax.plot(x_range, x_range, 'k:', alpha=0.3, linewidth=2, label='y = x (equal growth)')

ax.set_xlabel('Urban Clusters Mean Growth Rate', fontsize=14, fontweight='bold')
ax.set_ylabel('LCC Mean Growth Rate', fontsize=14, fontweight='bold')
ax.set_title('Relationship Between Satellite and Core Growth Rates\n(Each Point Represents One City)', 
             fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--')

# Remove the automatic legend (since we have annotations)
# ax.legend(loc='upper left', fontsize=8, framealpha=0.9)

plt.tight_layout()

# Save the figure
output_path_growth_correlation = f'{output_dir}\\satellite_vs_core_growth_correlation.png'
plt.savefig(output_path_growth_correlation, dpi=300, bbox_inches='tight')
print(f"\n✓ Satellite vs Core growth correlation plot saved to: {output_path_growth_correlation}")

plt.show()

# ============================================================================
# ADD TO TEXT FILE RESULTS
# ============================================================================
print("\n" + "="*80)
print("CORRELATION: URBAN CLUSTERS GROWTH vs CLUSTER 0 GROWTH")
print("="*80)

if len(merged_both_growth) > 1:
    pearson_r, pearson_p = stats.pearsonr(merged_both_growth['mean_growth_rate'], 
                                           merged_both_growth['cluster0_mean_growth_rate'])
    spearman_r, spearman_p = stats.spearmanr(merged_both_growth['mean_growth_rate'], 
                                              merged_both_growth['cluster0_mean_growth_rate'])
    
    print(f"\nNumber of cities: {len(merged_both_growth)}")
    print(f"Pearson r = {pearson_r:.4f}, p = {pearson_p:.6f}")
    print(f"Spearman ρ = {spearman_r:.4f}, p = {spearman_p:.6f}")
    
    if pearson_p < 0.05:
        direction = "POSITIVE" if pearson_r > 0 else "NEGATIVE"
        print(f"\n→ Significant {direction} correlation!")
        if pearson_r > 0:
            print("  Cities with faster-growing satellites also have faster-growing urban cores.")
        else:
            print("  Cities with faster-growing satellites have slower-growing urban cores.")
    else:
        print("\n→ No significant correlation")
        print("  Satellite and core growth rates are independent across cities.")
    
    # Append to results text file
    additional_results = []
    additional_results.append("")
    additional_results.append("="*80)
    additional_results.append("8. URBAN CLUSTERS GROWTH RATE vs  LCC GROWTH RATE")
    additional_results.append("="*80)
    additional_results.append("")
    additional_results.append(f"Number of cities analyzed: {len(merged_both_growth)}")
    additional_results.append("")
    additional_results.append("Spearman Correlation:")
    additional_results.append(f"  rho = {spearman_r:.6f}")
    additional_results.append(f"  p-value = {spearman_p:.6f}")
    additional_results.append(f"  Significance: {'***' if spearman_p < 0.001 else '**' if spearman_p < 0.01 else '*' if spearman_p < 0.05 else 'Not significant (p >= 0.05)'}")
    additional_results.append("")
    additional_results.append("Pearson Correlation:")
    additional_results.append(f"  r = {pearson_r:.6f}")
    additional_results.append(f"  p-value = {pearson_p:.6f}")
    additional_results.append(f"  Significance: {'***' if pearson_p < 0.001 else '**' if pearson_p < 0.01 else '*' if pearson_p < 0.05 else 'Not significant (p >= 0.05)'}")
    additional_results.append("")
    additional_results.append("Interpretation:")
    if pearson_p < 0.05:
        if pearson_r > 0:
            additional_results.append(f"  There is a significant POSITIVE correlation (r={pearson_r:.4f}).")
            additional_results.append("  Cities where satellite clusters grow faster also tend to have")
            additional_results.append("  faster-growing main urban cores (largest component).")
            additional_results.append("  This suggests coordinated growth dynamics across the urban system.")
        else:
            additional_results.append(f"  There is a significant NEGATIVE correlation (r={pearson_r:.4f}).")
            additional_results.append("  Cities where satellite clusters grow faster tend to have")
            additional_results.append("  slower-growing main urban cores (largest component).")
            additional_results.append("  This suggests a trade-off in growth between satellites and core.")
    else:
        additional_results.append(f"  No significant correlation found (p={pearson_p:.4f}).")
        additional_results.append("  Satellite growth and core growth rates appear to be independent.")
        additional_results.append("  Different cities may have different growth strategies (core vs peripheral).")
    additional_results.append("")
    
    # Read existing results file and append
    results_file_path = 'C:\\Users\\trique\\Downloads\\MASTER_THESIS\\data_vizualization\\src_EDEN\\powerlaw\\correlation_growth_vs_beta_complete.txt'
    with open(results_file_path, 'r') as f:
        existing_content = f.read()
    
    with open(results_file_path, 'w') as f:
        f.write(existing_content)
        f.write('\n'.join(additional_results))
    
    print(f"\n✓ Correlation results appended to: correlation_growth_vs_beta_complete.txt")

# Save the merged data
merged_both_growth.to_csv(f'{output_dir}\\satellite_vs_core_growth_data.csv', 
                          index=False)
print("✓ Satellite vs Core growth data saved")


# ============================================================================
# URBAN AREA EVOLUTION OVER TIME - ALL CITIES
# ============================================================================
# Add this code after the previous visualizations

print("\n" + "="*80)
print("Generating urban area evolution plots for all cities...")
print("="*80)

# Prepare data for area evolution
area_evolution_data = []

for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        filepath = os.path.join(directory, filename)
        city_name = filename.replace('output_', '').replace('.csv', '')
        
        # Read the CSV file
        df = pd.read_csv(filepath)
        
        # Get unique years
        years = sorted(df['year'].unique())
        
        for year in years:
            df_year = df[df['year'] == year]
            
            # Cluster 0 area
            cluster0_area = df_year[df_year['cluster_id'] == 0]['area_km2'].sum()
            
            # Satellites total area (all clusters except 0)
            satellites_area = df_year[df_year['cluster_id'] != 0]['area_km2'].sum()
            
            # Total urban area
            total_area = cluster0_area + satellites_area
            
            area_evolution_data.append({
                'city': city_name,
                'year': year,
                'cluster0_area': cluster0_area,
                'satellites_area': satellites_area,
                'total_area': total_area
            })

area_evolution_df = pd.DataFrame(area_evolution_data)

# Get unique cities
cities = sorted(area_evolution_df['city'].unique())
n_cities = len(cities)

print(f"Processing {n_cities} cities...")

# Calculate grid dimensions for subplots
n_cols = 3
n_rows = int(np.ceil(n_cities / n_cols))

# ============================================================================
# CREATE COMPREHENSIVE MULTI-PANEL PLOT
# ============================================================================

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
axes = axes.flatten() if n_cities > 1 else [axes]

for idx, city in enumerate(cities):
    ax = axes[idx]
    city_data = area_evolution_df[area_evolution_df['city'] == city].sort_values('year')
    
    city_label = city.replace('_', ' ').title()
    
    # Plot cluster 0 (largest component)
    ax.plot(city_data['year'], city_data['cluster0_area'], 
            marker='o', linewidth=2.5, markersize=8, 
            color='#2E86AB', label='Largest Connected Component (Cluster 0)',
            alpha=0.8)
    
    # Plot satellites total
    ax.plot(city_data['year'], city_data['satellites_area'], 
            marker='s', linewidth=2.5, markersize=8, 
            color='#A23B72', label='All Urban Clusters (Combined)',
            alpha=0.8)
    
    # Plot total urban area
    ax.plot(city_data['year'], city_data['total_area'], 
            marker='D', linewidth=2, markersize=7, 
            color='#F18F01', label='Total Urban Area',
            alpha=0.6, linestyle='--')
    
    # Fill between to show composition
    ax.fill_between(city_data['year'], 0, city_data['cluster0_area'], 
                     alpha=0.2, color='#2E86AB', label='_nolegend_')
    ax.fill_between(city_data['year'], city_data['cluster0_area'], 
                     city_data['total_area'], 
                     alpha=0.2, color='#A23B72', label='_nolegend_')
    
    ax.set_xlabel('Year', fontsize=11, fontweight='bold')
    ax.set_ylabel('Area (km²)', fontsize=11, fontweight='bold')
    ax.set_title(f'{city_label}', fontsize=13, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    
    # Add percentage text
    final_year_data = city_data.iloc[-1]
    if final_year_data['total_area'] > 0:
        cluster0_pct = (final_year_data['cluster0_area'] / final_year_data['total_area']) * 100
        satellites_pct = (final_year_data['satellites_area'] / final_year_data['total_area']) * 100
        
        stats_text = f"Final Year ({int(final_year_data['year'])}):\n"
        stats_text += f"Core: {cluster0_pct:.1f}%\n"
        stats_text += f"Satellites: {satellites_pct:.1f}%"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# Hide empty subplots
for idx in range(n_cities, len(axes)):
    axes[idx].axis('off')

plt.suptitle('Urban Area Evolution Over Time: Core vs Urban Clusters\n(All Cities)', 
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])

# Save the figure
output_path_evolution =f'{output_dir}\\urban_area_evolution_all_cities.png'
plt.savefig(output_path_evolution, dpi=300, bbox_inches='tight')
print(f"\n✓ Urban area evolution plot saved to: {output_path_evolution}")

plt.show()

# ============================================================================
# CREATE INDIVIDUAL HIGH-RESOLUTION PLOTS FOR EACH CITY
# ============================================================================

print("\n" + "="*80)
print("Generating individual high-resolution plots for each city...")
print("="*80)

# Create output directory for individual plots
individual_plots_dir = f'{output_dir}\\individual_city_evolution'
os.makedirs(individual_plots_dir, exist_ok=True)

for city in cities:
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    city_data = area_evolution_df[area_evolution_df['city'] == city].sort_values('year')
    city_label = city.replace('_', ' ').title()
    
    # Plot cluster 0 (largest component)
    line1 = ax.plot(city_data['year'], city_data['cluster0_area'], 
                    marker='o', linewidth=3, markersize=10, 
                    color='#2E86AB', label='LCC',
                    alpha=0.8, zorder=3)
    
    # Plot satellites total
    line2 = ax.plot(city_data['year'], city_data['satellites_area'], 
                    marker='s', linewidth=3, markersize=10, 
                    color='#A23B72', label='All Urban Clusters (Combined)',
                    alpha=0.8, zorder=3)
    
    # Plot total urban area
    line3 = ax.plot(city_data['year'], city_data['total_area'], 
                    marker='D', linewidth=2.5, markersize=9, 
                    color='#F18F01', label='Total Urban Area',
                    alpha=0.7, linestyle='--', zorder=2)
    
    # Fill between to show composition
    ax.fill_between(city_data['year'], 0, city_data['cluster0_area'], 
                     alpha=0.25, color='#2E86AB', zorder=1)
    ax.fill_between(city_data['year'], city_data['cluster0_area'], 
                     city_data['total_area'], 
                     alpha=0.25, color='#A23B72', zorder=1)
    
    # Calculate growth statistics
    initial_total = city_data.iloc[0]['total_area']
    final_total = city_data.iloc[-1]['total_area']
    initial_cluster0 = city_data.iloc[0]['cluster0_area']
    final_cluster0 = city_data.iloc[-1]['cluster0_area']
    initial_satellites = city_data.iloc[0]['satellites_area']
    final_satellites = city_data.iloc[-1]['satellites_area']
    
    n_years = city_data.iloc[-1]['year'] - city_data.iloc[0]['year']
    
    if initial_total > 0 and n_years > 0:
        total_growth = ((final_total - initial_total) / initial_total) * 100
        total_growth_annual = ((final_total / initial_total) ** (1/n_years) - 1) * 100
        
        cluster0_growth = ((final_cluster0 - initial_cluster0) / initial_cluster0) * 100 if initial_cluster0 > 0 else 0
        satellites_growth = ((final_satellites - initial_satellites) / initial_satellites) * 100 if initial_satellites > 0 else 0
        
        cluster0_pct_final = (final_cluster0 / final_total) * 100
        satellites_pct_final = (final_satellites / final_total) * 100
        
        stats_text = f'GROWTH STATISTICS\n'
        stats_text += f'{"="*35}\n\n'
        stats_text += f'Period: {int(city_data.iloc[0]["year"])} - {int(city_data.iloc[-1]["year"])} ({n_years} years)\n\n'
        stats_text += f'Total Urban Growth:\n'
        stats_text += f'  {initial_total:.2f} → {final_total:.2f} km²\n'
        stats_text += f'  +{total_growth:.1f}% total\n'
        stats_text += f'  {total_growth_annual:.2f}% per year\n\n'
        stats_text += f'  LCC:\n'
        stats_text += f'  Growth: +{cluster0_growth:.1f}%\n'
        stats_text += f'  Final share: {cluster0_pct_final:.1f}%\n\n'
        stats_text += f'  Urban Clusters:\n'
        stats_text += f'  Growth: +{satellites_growth:.1f}%\n'
        stats_text += f'  Final share: {satellites_pct_final:.1f}%'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85, 
                         edgecolor='black', linewidth=1.5))
    
    ax.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax.set_ylabel('Area (km²)', fontsize=14, fontweight='bold')
    ax.set_title(f'Urban Area Evolution: {city_label}', 
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=12, framealpha=0.95, 
             edgecolor='black', fancybox=True, shadow=True)
    
    plt.tight_layout()
    
    # Save individual plot
    individual_output = os.path.join(individual_plots_dir, f'area_evolution_{city}.png')
    plt.savefig(individual_output, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ {city_label}")

print(f"\n✓ Individual city plots saved to: {individual_plots_dir}")

# ============================================================================
# SAVE AREA EVOLUTION DATA
# ============================================================================
area_evolution_df.to_csv(f'{output_dir}\\urban_area_evolution_data.csv', 
                        index=False)
print("\n✓ Urban area evolution data saved to: urban_area_evolution_data.csv")




# ============================================================================
# SATELLITE GROWTH RATE vs NORMALIZED RADIAL DISTANCE ANALYSIS
# ============================================================================
# Add this code after the urban area evolution plots

print("\n" + "="*80)
print("Analyzing satellite growth rate vs normalized radial distance...")
print("="*80)

# First, get the cluster 0 area in 2014 for each city for normalization
cluster0_area_2014 = {}

for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        filepath = os.path.join(directory, filename)
        city_name = filename.replace('output_', '').replace('.csv', '')
        
        # Read the CSV file
        df = pd.read_csv(filepath)
        
        # Get cluster 0 area in 2014
        df_2014_c0 = df[(df['year'] == 2014) & (df['cluster_id'] == 0)]
        if len(df_2014_c0) > 0:
            # Use square root of area as characteristic length scale
            cluster0_area_2014[city_name] = np.sqrt(df_2014_c0['area_km2'].values[0])
        else:
            # If 2014 not available, use first available year
            df_c0 = df[df['cluster_id'] == 0].sort_values('year')
            if len(df_c0) > 0:
                cluster0_area_2014[city_name] = np.sqrt(df_c0.iloc[0]['area_km2'])
                print(f"  Warning: Using year {df_c0.iloc[0]['year']} instead of 2014 for {city_name}")

print(f"\nCluster 0 normalization factors (sqrt of area) calculated for {len(cluster0_area_2014)} cities")

# Merge growth rate data with radial distance
# We'll use the all_growth_df which has avg_growth_rate for each satellite cluster
print(f"\nSatellite clusters with growth data: {len(all_growth_df)}")

# Now add normalized distance to the growth data
growth_distance_data = []

for idx, growth_row in all_growth_df.iterrows():
    city_name = growth_row['city']
    cluster_id = growth_row['cluster_id']
    
    # Get normalization factor for this city
    if city_name not in cluster0_area_2014:
        continue
    
    norm_factor = cluster0_area_2014[city_name]
    
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
            'cluster0_area_2014_sqrt': norm_factor,
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
        
        # Add to results text
        distance_results = []
        distance_results.append("")
        distance_results.append("="*80)
        distance_results.append("9. URBAN CLUSTERS CLUSTERS: GROWTH RATE vs NORMALIZED RADIAL DISTANCE")
        distance_results.append("="*80)
        distance_results.append("")
        distance_results.append(f"Number of satellite clusters: {len(growth_distance_clean)}")
        distance_results.append(f"Number of cities: {len(growth_distance_clean['city'].unique())}")
        distance_results.append("")
        distance_results.append("Distance Normalization:")
        distance_results.append("  Normalized Distance = radial_distance_km / sqrt(Cluster0_Area_2014)")
        distance_results.append("  This normalizes by the characteristic length scale of each city")
        distance_results.append("")
        distance_results.append("Spearman Correlation:")
        distance_results.append(f"  rho = {spearman_r:.6f}")
        distance_results.append(f"  p-value = {spearman_p:.6f}")
        distance_results.append(f"  Significance: {'***' if spearman_p < 0.001 else '**' if spearman_p < 0.01 else '*' if spearman_p < 0.05 else 'Not significant (p >= 0.05)'}")
        distance_results.append("")
        distance_results.append("Pearson Correlation:")
        distance_results.append(f"  r = {pearson_r:.6f}")
        distance_results.append(f"  p-value = {pearson_p:.6f}")
        distance_results.append(f"  Significance: {'***' if pearson_p < 0.001 else '**' if pearson_p < 0.01 else '*' if pearson_p < 0.05 else 'Not significant (p >= 0.05)'}")
        distance_results.append("")
        distance_results.append("Interpretation:")
        if pearson_p < 0.05:
            if pearson_r > 0:
                distance_results.append(f"  There is a significant POSITIVE correlation (r={pearson_r:.4f}).")
                distance_results.append("  Satellite clusters farther from the urban core tend to grow FASTER.")
                distance_results.append("  This suggests more rapid peripheral expansion.")
            else:
                distance_results.append(f"  There is a significant NEGATIVE correlation (r={pearson_r:.4f}).")
                distance_results.append("  Satellite clusters farther from the urban core tend to grow SLOWER.")
                distance_results.append("  This suggests growth concentrated near the main urban core.")
        else:
            distance_results.append(f"  No significant correlation found (p={pearson_p:.4f}).")
            distance_results.append("  Satellite cluster growth rate appears independent of distance from the core.")
        distance_results.append("")
        
        # Append to existing results file
        results_file_path = f'{output_dir}\\correlation_growth_vs_beta_complete.txt'
        with open(results_file_path, 'r') as f:
            existing_content = f.read()
        
        with open(results_file_path, 'w') as f:
            f.write(existing_content)
            f.write('\n'.join(distance_results))
        
        print(f"\n✓ Radial distance correlation appended to: correlation_growth_vs_beta_complete.txt")

# ============================================================================
# VISUALIZATION: ALL CITIES COMBINED
# ============================================================================

print("\n" + "="*80)
print("Generating satellite growth rate vs normalized distance plot...")
print("="*80)

if len(growth_distance_clean) > 2:
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Get unique cities for color mapping
    unique_cities_rad = growth_distance_clean['city'].unique()
    colors_rad = plt.cm.tab20(np.linspace(0, 1, len(unique_cities_rad)))
    city_color_map_rad = dict(zip(unique_cities_rad, colors_rad))
    
    # Plot each city's satellites with different colors
    for city in unique_cities_rad:
        city_data = growth_distance_clean[growth_distance_clean['city'] == city]
        city_label = city.replace('_', ' ').title()
        
        ax.scatter(city_data['normalized_distance'], 
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
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=3, 
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
    stats_text += f'  r = {pearson_r:.4f}\n'
    stats_text += f'  p = {pearson_p:.6f}\n'
    if pearson_p < 0.001:
        stats_text += f'  Sig: *** (p < 0.001)\n\n'
    elif pearson_p < 0.01:
        stats_text += f'  Sig: ** (p < 0.01)\n\n'
    elif pearson_p < 0.05:
        stats_text += f'  Sig: * (p < 0.05)\n\n'
    else:
        stats_text += f'  Not significant\n\n'
    
    stats_text += f'Spearman Correlation:\n'
    stats_text += f'  ρ = {spearman_r:.4f}\n'
    stats_text += f'  p = {spearman_p:.6f}\n'
    if spearman_p < 0.001:
        stats_text += f'  Sig: ***'
    elif spearman_p < 0.01:
        stats_text += f'  Sig: **'
    elif spearman_p < 0.05:
        stats_text += f'  Sig: *'
    else:
        stats_text += f'  Not significant'
    
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, 
                     edgecolor='black', linewidth=2),
            family='monospace')
    
   
    
   
    
    # Add zero growth reference line
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Normalized Radial Distance\n(Distance / √Cluster0_Area_2014)', 
                  fontsize=14, fontweight='bold')
    ax.set_ylabel('Urban Cluster Average Growth Rate', fontsize=14, fontweight='bold')
    ax.set_title('Urban Cluster Growth Rate vs Distance from Urban Core\n(All Clusters from All Cities)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
             fontsize=9, framealpha=0.9, ncol=1)
    
    plt.tight_layout()
    
    # Save the figure
    output_path_radial = f'{output_dir}\\satellite_growth_vs_normalized_distance.png'
    plt.savefig(output_path_radial, dpi=300, bbox_inches='tight')
    print(f"\n✓ Satellite growth rate vs normalized distance plot saved to: {output_path_radial}")
    
    plt.show()
    
    # ========================================================================
    # SAVE DATA
    # ========================================================================
    growth_distance_clean.to_csv(f'{output_dir}\\satellite_growth_distance_data.csv', 
                           index=False)
    print("✓ Satellite growth rate-distance data saved")
    
    # ========================================================================
    # ADDITIONAL ANALYSIS: BY CITY
    # ========================================================================
    print("\n" + "-"*80)
    print("CITY-SPECIFIC CORRELATIONS")
    print("-"*80)
    
    city_correlations = []
    
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
    
    # ========================================================================
    # BINNED ANALYSIS: Show average growth rate in distance bins
    # ========================================================================
    print("\n" + "-"*80)
    print("BINNED ANALYSIS: Average growth rate by distance bins")
    print("-"*80)
    
    # Create distance bins
    n_bins = 10
    growth_distance_clean['distance_bin'] = pd.qcut(
        growth_distance_clean['normalized_distance'], 
        q=n_bins, 
        labels=False, 
        duplicates='drop'
    )
    
    binned_stats = growth_distance_clean.groupby('distance_bin').agg({
        'normalized_distance': ['mean', 'min', 'max'],
        'avg_growth_rate': ['mean', 'std', 'count']
    }).reset_index()
    
    print("\nDistance Bin Statistics:")
    print(binned_stats.to_string())

print("\n" + "="*80)
print("Radial distance vs growth rate analysis complete!")
print("="*80)




# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*80)
print("URBAN AREA EVOLUTION SUMMARY")
print("="*80)

for city in cities:
    city_data = area_evolution_df[area_evolution_df['city'] == city].sort_values('year')
    city_label = city.replace('_', ' ').title()
    
    initial = city_data.iloc[0]
    final = city_data.iloc[-1]
    
    total_change = final['total_area'] - initial['total_area']
    core_change = final['cluster0_area'] - initial['cluster0_area']
    sat_change = final['satellites_area'] - initial['satellites_area']
    
    print(f"\n{city_label}:")
    print(f"  Years: {int(initial['year'])} - {int(final['year'])}")
    print(f"  Total area change: {initial['total_area']:.2f} → {final['total_area']:.2f} km² (+{total_change:.2f})")
    print(f"  Core change: {initial['cluster0_area']:.2f} → {final['cluster0_area']:.2f} km² (+{core_change:.2f})")
    print(f"  Satellites change: {initial['satellites_area']:.2f} → {final['satellites_area']:.2f} km² (+{sat_change:.2f})")
    print(f"  Final composition: {(final['cluster0_area']/final['total_area']*100):.1f}% core, {(final['satellites_area']/final['total_area']*100):.1f}% satellites")

print("\n" + "="*80)
print("Area evolution analysis complete!")
print("="*80)
print("="*80)
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)