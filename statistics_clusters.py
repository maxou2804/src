import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import powerlaw
from scipy import stats
from scipy.stats import pareto
from scipy.optimize import curve_fit
import os


directory = 'C:\\Users\\trique\\Downloads\\MASTER_THESIS\\data_vizualization\\src_EDEN\\outputs_clusters'

for filename in os.listdir(directory):

    if filename.endswith(".csv"):
        filepath = os.path.join(directory, filename)
        print(filename)

        # Set style for better-looking plots
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 10)

        # Read the CSV file
        df = pd.read_csv(filepath)

        # ============================================================================
        # GROWTH RATE ANALYSIS
        # ============================================================================
        print("\n" + "="*80)
        print("CLUSTER GROWTH RATE ANALYSIS")
        print("="*80)
        
        # Get all clusters except cluster 0
        df_growth = df[df['cluster_id'] != 0].copy()
        df_growth = df_growth.sort_values(['cluster_id', 'year'])
        
        # Initialize lists to store growth rate data
        growth_data = []
        
        # Get unique cluster IDs
        cluster_ids = df_growth['cluster_id'].unique()
        
        print(f"\nAnalyzing {len(cluster_ids)} unique clusters (excluding cluster 0)...")
        
        for cluster_id in cluster_ids:
            # Get all years for this cluster
            cluster_df = df_growth[df_growth['cluster_id'] == cluster_id].sort_values('year')
            
            if len(cluster_df) < 2:
                continue  # Need at least 2 years to calculate growth rate
            
            years = cluster_df['year'].values
            areas = cluster_df['area_km2'].values
            
            # Calculate growth rates for consecutive years
            cluster_growth_rates = []
            cluster_areas_for_growth = []
            
            for i in range(len(years) - 1):
                # Check if years are consecutive
                if years[i+1] - years[i] == 1:
                    # Calculate growth rate: (A(t+1) - A(t)) / A(t)
                    growth_rate = (areas[i+1] - areas[i]) / areas[i]
                    
                    # Check if cluster was absorbed in the next year
                    next_year_data = df[(df['year'] == years[i+1]) & (df['cluster_id'] == cluster_id)]
                    
                    # If cluster still exists and wasn't absorbed, record the growth
                    if len(next_year_data) > 0:
                        absorbed = next_year_data['absorbed_clusters'].values[0]
                        # Stop tracking if cluster was absorbed (has entries in absorbed_clusters)
                        if pd.isna(absorbed) or absorbed == '':
                            cluster_growth_rates.append(growth_rate)
                            cluster_areas_for_growth.append(areas[i])
                        else:
                            # Cluster was absorbed, stop tracking
                            break
            
            # Calculate average growth rate for this cluster
            if len(cluster_growth_rates) > 0:
                avg_growth_rate = np.mean(cluster_growth_rates)
                avg_area = np.mean(cluster_areas_for_growth)
                
                growth_data.append({
                    'cluster_id': cluster_id,
                    'avg_growth_rate': avg_growth_rate,
                    'avg_area': avg_area,
                    'n_observations': len(cluster_growth_rates),
                    'first_year': years[0],
                    'last_tracked_year': years[len(cluster_growth_rates)],
                    'initial_area': areas[0],
                    'all_growth_rates': cluster_growth_rates,
                    'all_areas': cluster_areas_for_growth
                })
        
        growth_df = pd.DataFrame(growth_data)
        
        if len(growth_df) == 0:
            print("No growth data available for analysis.")
            continue
        
        print(f"\nSuccessfully analyzed growth rates for {len(growth_df)} clusters")
        print(f"Growth rate statistics:")
        print(f"  Mean growth rate: {growth_df['avg_growth_rate'].mean():.4f} ({growth_df['avg_growth_rate'].mean()*100:.2f}%)")
        print(f"  Median growth rate: {growth_df['avg_growth_rate'].median():.4f} ({growth_df['avg_growth_rate'].median()*100:.2f}%)")
        print(f"  Std dev: {growth_df['avg_growth_rate'].std():.4f}")
        print(f"  Min: {growth_df['avg_growth_rate'].min():.4f}")
        print(f"  Max: {growth_df['avg_growth_rate'].max():.4f}")
        
        print(f"\nArea statistics:")
        print(f"  Mean area: {growth_df['avg_area'].mean():.3f} km²")
        print(f"  Median area: {growth_df['avg_area'].median():.3f} km²")
        
        # ============================================================================
        # CREATE GROWTH RATE VISUALIZATIONS
        # ============================================================================
        print("\nGenerating growth rate visualizations...")
        
        fig_growth, axes_growth = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Scatter plot of Area vs Growth Rate
        ax1 = axes_growth[0, 0]
        scatter = ax1.scatter(growth_df['avg_area'], growth_df['avg_growth_rate'], 
                            c=growth_df['n_observations'], cmap='viridis', 
                            alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Zero growth')
        ax1.set_xlabel('Average Cluster Area (km²)', fontsize=12)
        ax1.set_ylabel('Average Growth Rate', fontsize=12)
        ax1.set_title('Cluster Area vs Growth Rate', fontsize=14, fontweight='bold')
        ax1.set_xscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        cbar1 = plt.colorbar(scatter, ax=ax1)
        cbar1.set_label('# Observations', fontsize=10)
        
        # Plot 2: Log-log scatter with binned average
        ax2 = axes_growth[0, 1]
        ax2.scatter(growth_df['avg_area'], growth_df['avg_growth_rate'], 
                   alpha=0.4, s=30, color='blue', edgecolors='black', linewidth=0.3)
        
        # Create bins and calculate average growth rate per bin
        log_bins = np.logspace(np.log10(growth_df['avg_area'].min()), 
                               np.log10(growth_df['avg_area'].max()), 10)
        bin_centers = []
        bin_means = []
        bin_stds = []
        
        for i in range(len(log_bins)-1):
            mask = (growth_df['avg_area'] >= log_bins[i]) & (growth_df['avg_area'] < log_bins[i+1])
            if mask.sum() > 0:
                bin_centers.append(np.sqrt(log_bins[i] * log_bins[i+1]))
                bin_means.append(growth_df[mask]['avg_growth_rate'].mean())
                bin_stds.append(growth_df[mask]['avg_growth_rate'].std())
        
        ax2.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='o-', 
                    color='red', linewidth=2, markersize=8, capsize=5, 
                    label='Binned average ± std', markeredgecolor='black')
        ax2.axhline(y=0, color='gray', linestyle='--', linewidth=2, alpha=0.5)
        ax2.set_xlabel('Average Cluster Area (km²)', fontsize=12)
        ax2.set_ylabel('Average Growth Rate', fontsize=12)
        ax2.set_title('Area vs Growth Rate (Log scale with bins)', fontsize=14, fontweight='bold')
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: Distribution of growth rates
        ax3 = axes_growth[0, 2]
        ax3.hist(growth_df['avg_growth_rate'], bins=50, edgecolor='black', 
                alpha=0.7, color='coral')
        ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero growth')
        ax3.axvline(x=growth_df['avg_growth_rate'].median(), color='blue', 
                   linestyle=':', linewidth=2, label=f'Median = {growth_df["avg_growth_rate"].median():.3f}')
        ax3.set_xlabel('Average Growth Rate', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.set_title('Distribution of Growth Rates', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Growth rate by size category
        ax4 = axes_growth[1, 0]
        # Create size categories
        percentiles = [0, 25, 50, 75, 100]
        size_bins = np.percentile(growth_df['avg_area'], percentiles)
        categories = ['Small\n(0-25%)', 'Medium-Small\n(25-50%)', 
                     'Medium-Large\n(50-75%)', 'Large\n(75-100%)']
        
        growth_by_category = []
        for i in range(len(size_bins)-1):
            mask = (growth_df['avg_area'] >= size_bins[i]) & (growth_df['avg_area'] < size_bins[i+1])
            growth_by_category.append(growth_df[mask]['avg_growth_rate'].values)
        
        bp = ax4.boxplot(growth_by_category, labels=categories, patch_artist=True,
                        showmeans=True, meanline=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        ax4.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax4.set_xlabel('Size Category', fontsize=12)
        ax4.set_ylabel('Growth Rate', fontsize=12)
        ax4.set_title('Growth Rate by Cluster Size Category', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Plot 5: Time series of average growth rate
        ax5 = axes_growth[1, 1]
        # Calculate average growth rate by year (for all clusters active in that year)
        all_growth_by_year = []
        for cluster_data in growth_data:
            for i, year in enumerate(range(cluster_data['first_year'], 
                                          cluster_data['last_tracked_year'] + 1)):
                if i < len(cluster_data['all_growth_rates']):
                    all_growth_by_year.append({
                        'year': year,
                        'growth_rate': cluster_data['all_growth_rates'][i]
                    })
        
        if len(all_growth_by_year) > 0:
            yearly_growth = pd.DataFrame(all_growth_by_year)
            yearly_avg = yearly_growth.groupby('year')['growth_rate'].agg(['mean', 'std', 'count'])
            
            ax5.plot(yearly_avg.index, yearly_avg['mean'], 'o-', linewidth=2, 
                    markersize=6, color='darkgreen', label='Mean growth rate')
            ax5.fill_between(yearly_avg.index, 
                            yearly_avg['mean'] - yearly_avg['std'],
                            yearly_avg['mean'] + yearly_avg['std'],
                            alpha=0.3, color='green', label='±1 std dev')
            ax5.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5)
            ax5.set_xlabel('Year', fontsize=12)
            ax5.set_ylabel('Average Growth Rate', fontsize=12)
            ax5.set_title('Time Evolution of Growth Rates', fontsize=14, fontweight='bold')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # Plot 6: Correlation analysis
        ax6 = axes_growth[1, 2]
        # Separate positive and negative growth
        positive_growth = growth_df[growth_df['avg_growth_rate'] > 0]
        negative_growth = growth_df[growth_df['avg_growth_rate'] < 0]
        
        ax6.scatter(positive_growth['avg_area'], positive_growth['avg_growth_rate'], 
                   alpha=0.5, s=40, color='green', label=f'Growing ({len(positive_growth)})', 
                   edgecolors='black', linewidth=0.5)
        ax6.scatter(negative_growth['avg_area'], negative_growth['avg_growth_rate'], 
                   alpha=0.5, s=40, color='red', label=f'Shrinking ({len(negative_growth)})', 
                   edgecolors='black', linewidth=0.5)
        ax6.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
        ax6.set_xlabel('Average Cluster Area (km²)', fontsize=12)
        ax6.set_ylabel('Average Growth Rate', fontsize=12)
        ax6.set_title('Growing vs Shrinking Clusters', fontsize=14, fontweight='bold')
        ax6.set_xscale('log')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'C:\\Users\\trique\\Downloads\\MASTER_THESIS\\data_vizualization\\src_EDEN\\powerlaw\\growth_rate_analysis_{filename}.png', 
                   dpi=300, bbox_inches='tight')
        print("✓ Growth rate analysis plot saved")
        
        # ============================================================================
        # ADDITIONAL ANALYSIS: POWER-LAW FOR GROWTH RATES
        # ============================================================================
        print("\nAnalyzing relationship between area and growth rate...")
        
        # Calculate correlation
        corr_pearson = stats.pearsonr(np.log10(growth_df['avg_area']), growth_df['avg_growth_rate'])
        corr_spearman = stats.spearmanr(growth_df['avg_area'], growth_df['avg_growth_rate'])
        
        print(f"Correlation between log(area) and growth rate:")
        print(f"  Pearson r = {corr_pearson[0]:.4f}, p = {corr_pearson[1]:.4f}")
        print(f"  Spearman ρ = {corr_spearman[0]:.4f}, p = {corr_spearman[1]:.4f}")
        
        # Fit linear model: growth_rate = a * log(area) + b
        log_areas = np.log10(growth_df['avg_area'])
        growth_rates = growth_df['avg_growth_rate']
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_areas, growth_rates)
        
        print(f"\nLinear regression: growth_rate = {slope:.4f} * log10(area) + {intercept:.4f}")
        print(f"  R² = {r_value**2:.4f}")
        print(f"  p-value = {p_value:.4f}")
        
        if p_value < 0.05:
            if slope < 0:
                print("  → Larger clusters tend to grow SLOWER (negative scaling)")
            else:
                print("  → Larger clusters tend to grow FASTER (positive scaling)")
        else:
            print("  → No significant relationship between size and growth rate")
        
        # Save growth rate data
        growth_df.to_csv(f'C:\\Users\\trique\\Downloads\\MASTER_THESIS\\data_vizualization\\src_EDEN\\powerlaw\\growth_rate_data_{filename}.csv', 
                        index=False)
        print("✓ Growth rate data saved")
        
        # Save detailed growth trajectories
        trajectories = []
        for cluster_data in growth_data:
            for i in range(len(cluster_data['all_growth_rates'])):
                trajectories.append({
                    'cluster_id': cluster_data['cluster_id'],
                    'year': cluster_data['first_year'] + i,
                    'area': cluster_data['all_areas'][i],
                    'growth_rate': cluster_data['all_growth_rates'][i]
                })
        
        trajectories_df = pd.DataFrame(trajectories)
        trajectories_df.to_csv(f'C:\\Users\\trique\\Downloads\\MASTER_THESIS\\data_vizualization\\src_EDEN\\powerlaw\\growth_trajectories_{filename}.csv', 
                              index=False)
        print("✓ Growth trajectories saved")

        # ============================================================================
        # ORIGINAL POWER-LAW ANALYSIS (EXISTING CODE)
        # ============================================================================
        
        # Filter out cluster 0 (the largest cluster)
        df_filtered = df[df['cluster_id'] != 0].copy()

        # Extract cluster areas
        cluster_areas = df_filtered['area_km2'].values

        print("\n" + "="*80)
        print("POWER-LAW ANALYSIS OF CLUSTER AREAS")
        print("="*80)
        print(f"\nDataset: {len(cluster_areas)} cluster observations (excluding cluster 0)")
        print(f"Years: {df_filtered['year'].min()} to {df_filtered['year'].max()}")
        print(f"Area range: {cluster_areas.min():.3f} - {cluster_areas.max():.3f} km²")

        # ============================================================================
        # 1. FIT POWER-LAW DISTRIBUTION USING POWERLAW PACKAGE
        # ============================================================================
        print("\n" + "="*80)
        print("POWER-LAW FITTING RESULTS")
        print("="*80)

        # Fit power law to the data
        fit = powerlaw.Fit(cluster_areas, discrete=False, verbose=False)

        print(f"\n1. Power-Law Parameters:")
        print(f"   Alpha (exponent): {fit.alpha:.4f}")
        print(f"   Standard error: {fit.sigma:.4f}")
        print(f"   xmin (minimum value for fit): {fit.xmin:.4f} km²")
        print(f"   Number of data points used in fit: {np.sum(cluster_areas >= fit.xmin)}")

        # Calculate R-squared for power law fit
        x_fit = cluster_areas[cluster_areas >= fit.xmin]
        y_empirical, x_bins = np.histogram(x_fit, bins=30, density=True)
        x_centers = (x_bins[:-1] + x_bins[1:]) / 2

        # Filter out zero values for log-log regression
        valid_idx = y_empirical > 0
        if np.sum(valid_idx) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                np.log10(x_centers[valid_idx]), 
                np.log10(y_empirical[valid_idx])
            )
            print(f"   R² (log-log fit): {r_value**2:.4f}")
            print(f"   Slope from log-log regression: {slope:.4f}")

        # ============================================================================
        # 2. COMPARE WITH OTHER DISTRIBUTIONS
        # ============================================================================
        print(f"\n2. Distribution Comparison:")

        # Compare power law
        R_pl_ln, p_pl_ln = fit.distribution_compare('power_law', 'lognormal', normalized_ratio=True)
        print(f"   Power-law vs Lognormal:")
        print(f"     Log-likelihood ratio: {R_pl_ln:.4f}")
        print(f"     p-value: {p_pl_ln:.4f}")
        if p_pl_ln < 0.05:
            if R_pl_ln > 0:
                print(f"     → Power-law is significantly better (p < 0.05)")
            else:
                print(f"     → Lognormal is significantly better (p < 0.05)")
        else:
            print(f"     → No significant difference (p >= 0.05)")

        # Compare power law with exponential
        R_pl_exp, p_pl_exp = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)
        print(f"\n   Power-law vs Exponential:")
        print(f"     Log-likelihood ratio: {R_pl_exp:.4f}")
        print(f"     p-value: {p_pl_exp:.4f}")
        if p_pl_exp < 0.05:
            if R_pl_exp > 0:
                print(f"     → Power-law is significantly better (p < 0.05)")
            else:
                print(f"     → Exponential is significantly better (p < 0.05)")
        else:
            print(f"     → No significant difference (p >= 0.05)")

        # Fit lognormal for comparison
        fit_lognormal = powerlaw.Fit(cluster_areas, discrete=False, verbose=False)
        fit_lognormal.lognormal.mu
        fit_lognormal.lognormal.sigma

        print(f"\n3. Lognormal Parameters (for comparison):")
        print(f"   μ (mu): {fit_lognormal.lognormal.mu:.4f}")
        print(f"   σ (sigma): {fit_lognormal.lognormal.sigma:.4f}")

        # ============================================================================
        # 3. MANUAL POWER-LAW FIT (ALTERNATIVE METHOD)
        # ============================================================================
        print(f"\n4. Manual Power-Law Fit (log-log linear regression):")

        # Use all data or data above xmin
        x_data = cluster_areas[cluster_areas >= fit.xmin]
        x_sorted = np.sort(x_data)

        # Calculate empirical CCDF (Complementary Cumulative Distribution Function)
        ccdf = np.arange(len(x_sorted), 0, -1) / len(x_sorted)

        # Fit linear regression in log-log space
        valid_ccdf = ccdf > 0
        log_x = np.log10(x_sorted[valid_ccdf])
        log_ccdf = np.log10(ccdf[valid_ccdf])

        slope_manual, intercept_manual, r_val, p_val, stderr = stats.linregress(log_x, log_ccdf)
        alpha_manual = -slope_manual  # For CCDF, slope is negative of alpha

        print(f"   Alpha (from CCDF): {alpha_manual:.4f}")
        print(f"   R² (log-log CCDF): {r_val**2:.4f}")
        print(f"   Standard error: {stderr:.4f}")

        # ============================================================================
        # 4. CREATE VISUALIZATIONS
        # ============================================================================
        print("\n" + "="*80)
        print("GENERATING POWER-LAW VISUALIZATIONS...")
        print("="*80)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Plot 1: Log-log histogram
        ax1 = axes[0, 0]
        x_fit_range = cluster_areas[cluster_areas >= fit.xmin]
        fit.power_law.plot_pdf(ax=ax1, color='r', linestyle='--', linewidth=2, label=f'Power-law fit\n(α={fit.alpha:.3f})')
        fit.plot_pdf(ax=ax1, color='b', linewidth=2, marker='o', markersize=4, label='Empirical data')
        ax1.set_xlabel('Area (km²)', fontsize=12)
        ax1.set_ylabel('Probability Density P(x)', fontsize=12)
        ax1.set_title('Log-Log Plot: PDF of Cluster Areas', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, which='both')

        # Plot 2: Log-log CCDF
        ax2 = axes[0, 1]
        fit.power_law.plot_ccdf(ax=ax2, color='r', linestyle='--', linewidth=2, label=f'Power-law fit\n(α={fit.alpha:.3f})')
        fit.plot_ccdf(ax=ax2, color='b', linewidth=2, marker='o', markersize=4, label='Empirical CCDF')
        ax2.set_xlabel('Area (km²)', fontsize=12)
        ax2.set_ylabel('P(X ≥ x)', fontsize=12)
        ax2.set_title('Log-Log Plot: CCDF of Cluster Areas', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, which='both')

        # Plot 3: Compare with other distributions (CCDF)
        ax3 = axes[0, 2]
        fit.power_law.plot_ccdf(ax=ax3, color='r', linestyle='--', linewidth=2, label=f'Power-law (α={fit.alpha:.3f})')
        fit.lognormal.plot_ccdf(ax=ax3, color='g', linestyle=':', linewidth=2, label='Lognormal')
        fit.exponential.plot_ccdf(ax=ax3, color='orange', linestyle='-.', linewidth=2, label='Exponential')
        fit.plot_ccdf(ax=ax3, color='b', linewidth=2, marker='o', markersize=4, label='Empirical')
        ax3.set_xlabel('Area (km²)', fontsize=12)
        ax3.set_ylabel('P(X ≥ x)', fontsize=12)
        ax3.set_title('Distribution Comparison (CCDF)', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3, which='both')

        # Plot 4: Linear scale histogram with fits
        ax4 = axes[1, 0]
        n, bins, patches = ax4.hist(cluster_areas, bins=40, density=True, alpha=0.6, 
                                    color='skyblue', edgecolor='black', label='Data')

        # Generate fitted distributions
        x_range = np.linspace(fit.xmin, cluster_areas.max(), 200)
        pdf_pl = fit.power_law.pdf(x_range)
        pdf_ln = fit.lognormal.pdf(x_range)
        pdf_exp = fit.exponential.pdf(x_range)

        ax4.plot(x_range, pdf_pl, 'r--', linewidth=2, label=f'Power-law (α={fit.alpha:.3f})')
        ax4.plot(x_range, pdf_ln, 'g:', linewidth=2, label='Lognormal')
        ax4.plot(x_range, pdf_exp, 'orange', linestyle='-.', linewidth=2, label='Exponential')
        ax4.set_xlabel('Area (km²)', fontsize=12)
        ax4.set_ylabel('Probability Density', fontsize=12)
        ax4.set_title('Distribution Fits (Linear Scale)', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, cluster_areas.max())

        # Plot 5: Residuals plot
        ax5 = axes[1, 1]
        x_sorted_fit = np.sort(cluster_areas[cluster_areas >= fit.xmin])
        ccdf_fit = np.arange(len(x_sorted_fit), 0, -1) / len(x_sorted_fit)
        ccdf_predicted = fit.power_law.ccdf(x_sorted_fit)
        residuals = np.log10(ccdf_fit) - np.log10(ccdf_predicted)

        ax5.scatter(x_sorted_fit, residuals, alpha=0.5, s=20, color='purple')
        ax5.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax5.set_xlabel('Area (km²)', fontsize=12)
        ax5.set_ylabel('Log-Residuals', fontsize=12)
        ax5.set_title('Power-Law Fit Residuals', fontsize=14, fontweight='bold')
        ax5.set_xscale('log')
        ax5.grid(True, alpha=0.3)

        # Plot 6: Q-Q plot for power law
        ax6 = axes[1, 2]
        b_param = fit.alpha - 1
        quantiles = np.linspace(0.01, 0.99, len(x_fit_range))
        theoretical_quantiles = pareto.ppf(quantiles, b_param, scale=fit.xmin)
        empirical_quantiles = np.sort(x_fit_range)

        ax6.scatter(theoretical_quantiles, empirical_quantiles, alpha=0.5, s=20, color='teal')
        min_val = min(theoretical_quantiles.min(), empirical_quantiles.min())
        max_val = max(theoretical_quantiles.max(), empirical_quantiles.max())
        ax6.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax6.set_xlabel('Theoretical Quantiles', fontsize=12)
        ax6.set_ylabel('Empirical Quantiles', fontsize=12)
        ax6.set_title('Q-Q Plot: Power-Law Fit', fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'C:\\Users\\trique\\Downloads\\MASTER_THESIS\\data_vizualization\\src_EDEN\\powerlaw\\powerlaw_analysis_{filename}.png', 
                   dpi=300, bbox_inches='tight')
        print("✓ Power-law analysis plot saved")

        # ============================================================================
        # 5. TIME EVOLUTION OF POWER-LAW EXPONENT
        # ============================================================================
        print("\nAnalyzing time evolution of power-law exponent...")

        years = sorted(df_filtered['year'].unique())
        alpha_by_year = []
        sigma_by_year = []
        xmin_by_year = []
        r_squared_by_year = []

        for year in years:
            year_data = df_filtered[df_filtered['year'] == year]['area_km2'].values
            if len(year_data) >= 3:
                try:
                    fit_year = powerlaw.Fit(year_data, discrete=False, verbose=False)
                    alpha_by_year.append(fit_year.alpha)
                    sigma_by_year.append(fit_year.sigma)
                    xmin_by_year.append(fit_year.xmin)
                    
                    x_fit_year = year_data[year_data >= fit_year.xmin]
                    if len(x_fit_year) > 1:
                        x_sorted = np.sort(x_fit_year)
                        ccdf = np.arange(len(x_sorted), 0, -1) / len(x_sorted)
                        valid = ccdf > 0
                        if np.sum(valid) > 1:
                            _, _, r_val, _, _ = stats.linregress(
                                np.log10(x_sorted[valid]), 
                                np.log10(ccdf[valid])
                            )
                            r_squared_by_year.append(r_val**2)
                        else:
                            r_squared_by_year.append(np.nan)
                    else:
                        r_squared_by_year.append(np.nan)
                except:
                    alpha_by_year.append(np.nan)
                    sigma_by_year.append(np.nan)
                    xmin_by_year.append(np.nan)
                    r_squared_by_year.append(np.nan)
            else:
                alpha_by_year.append(np.nan)
                sigma_by_year.append(np.nan)
                xmin_by_year.append(np.nan)
                r_squared_by_year.append(np.nan)

        # Create time evolution plots
        fig2, axes2 = plt.subplots(2, 2, figsize=(15, 10))

        # Plot alpha over time
        ax1 = axes2[0, 0]
        valid_mask = ~np.isnan(alpha_by_year)
        ax1.plot(np.array(years)[valid_mask], np.array(alpha_by_year)[valid_mask], 
                'o-', linewidth=2, markersize=6, color='darkblue')
        ax1.axhline(y=fit.alpha, color='r', linestyle='--', linewidth=2, 
                    label=f'Overall α = {fit.alpha:.3f}')
        ax1.fill_between(np.array(years)[valid_mask], 
                        np.array(alpha_by_year)[valid_mask] - np.array(sigma_by_year)[valid_mask],
                        np.array(alpha_by_year)[valid_mask] + np.array(sigma_by_year)[valid_mask],
                        alpha=0.3, color='blue')
        ax1.set_xlabel('Year', fontsize=12)
        ax1.set_ylabel('Power-Law Exponent (α)', fontsize=12)
        ax1.set_title('Time Evolution of Power-Law Exponent', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot xmin over time
        ax2 = axes2[0, 1]
        ax2.plot(np.array(years)[valid_mask], np.array(xmin_by_year)[valid_mask], 
                's-', linewidth=2, markersize=6, color='darkgreen')
        ax2.set_xlabel('Year', fontsize=12)
        ax2.set_ylabel('xmin (km²)', fontsize=12)
        ax2.set_title('Time Evolution of xmin', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Plot R² over time
        ax3 = axes2[1, 0]
        ax3.plot(np.array(years)[valid_mask], np.array(r_squared_by_year)[valid_mask], 
                '^-', linewidth=2, markersize=6, color='purple')
        ax3.set_xlabel('Year', fontsize=12)
        ax3.set_ylabel('R² (log-log fit)', fontsize=12)
        ax3.set_title('Goodness of Fit Over Time', fontsize=14, fontweight='bold')
        ax3.set_ylim([0, 1])
        ax3.grid(True, alpha=0.3)

        # Plot distribution of alphas
        ax4 = axes2[1, 1]
        valid_alphas = np.array(alpha_by_year)[valid_mask]
        ax4.hist(valid_alphas, bins=15, edgecolor='black', alpha=0.7, color='coral')
        ax4.axvline(x=fit.alpha, color='r', linestyle='--', linewidth=2, 
                    label=f'Overall α = {fit.alpha:.3f}')
        ax4.axvline(x=np.nanmean(alpha_by_year), color='blue', linestyle=':', linewidth=2,
                    label=f'Mean α = {np.nanmean(alpha_by_year):.3f}')
        ax4.set_xlabel('Power-Law Exponent (α)', fontsize=12)
        ax4.set_ylabel('Frequency', fontsize=12)
        ax4.set_title('Distribution of Annual α Values', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'C:\\Users\\trique\\Downloads\\MASTER_THESIS\\data_vizualization\\src_EDEN\\powerlaw\\powerlaw_time_evolution_{filename}.png', 
                   dpi=300, bbox_inches='tight')
        print("✓ Time evolution plot saved")

        # ============================================================================
        # 6. SAVE RESULTS TO FILES
        # ============================================================================
        print("\nSaving power-law results...")

        # Save overall power-law results
        powerlaw_results = pd.DataFrame({
            'Parameter': [
                'Alpha (exponent)',
                'Standard Error',
                'xmin',
                'Number of points in fit',
                'R² (log-log)',
                'Lognormal mu',
                'Lognormal sigma',
                'Log-likelihood ratio (PL vs Lognormal)',
                'p-value (PL vs Lognormal)',
                'Log-likelihood ratio (PL vs Exponential)',
                'p-value (PL vs Exponential)',
                'Manual Alpha (from CCDF)',
                'Manual R² (CCDF)'
            ],
            'Value': [
                fit.alpha,
                fit.sigma,
                fit.xmin,
                np.sum(cluster_areas >= fit.xmin),
                r_value**2 if np.sum(valid_idx) > 1 else np.nan,
                fit_lognormal.lognormal.mu,
                fit_lognormal.lognormal.sigma,
                R_pl_ln,
                p_pl_ln,
                R_pl_exp,
                p_pl_exp,
                alpha_manual,
                r_val**2
            ]
        })

        powerlaw_results.to_csv(f'C:\\Users\\trique\\Downloads\\MASTER_THESIS\\data_vizualization\\src_EDEN\\powerlaw\\powerlaw_fit_parameters_{filename}.csv', 
                               index=False)
        print("✓ Power-law parameters saved")

        # Save time evolution results
        time_evolution_df = pd.DataFrame({
            'year': years,
            'alpha': alpha_by_year,
            'sigma': sigma_by_year,
            'xmin': xmin_by_year,
            'r_squared': r_squared_by_year
        })

        time_evolution_df.to_csv(f'C:\\Users\\trique\\Downloads\\MASTER_THESIS\\data_vizualization\\src_EDEN\\powerlaw\\powerlaw_time_evolution_{filename}.csv', 
                                index=False)
        print("✓ Time evolution data saved")

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE FOR " + filename)
        print("="*80)




