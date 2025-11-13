import pandas as pd
import os 
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from datetime import datetime
import matplotlib.cm as cm

# Try to import adjustText for better label placement
try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False
    print("adjustText not installed. Using custom label placement. Install with: pip install adjustText")

ratio_collection=[]
radius_collection=[]
rate_collection=[]
directory='C:\\Users\\trique\\Downloads\\MASTER_THESIS\\data_vizualization\\render_report\\csv_outputs'

data=pd.DataFrame({'City': ['Ningbo','Chengdu Deyang', 'Beijing Lafang','Changzhou','Bengalore','Kolkata','Paris','Bangkok','Cairo','Guatemala City','Johannesburg','London','Mexico City','Nairobi','Santiago','Sao Paulo','Tehran','Las Vegas','Atlanta'],
      'alpha':[0.56, 0.53, 0.54, 0.54, 0.55, 0.52, 0.52 ,0.53, 0.53, 0.52, 0.58, 0.54, 0.55, 0.56, 0.58, 0.51, 0.55, 0.55, 0.56],
      'beta': [0.44, 0.68, 0.41, 0.37, 0.83, 0.34, 0.56, 1.01, 0.37, 0.37, 0.07, 0.01, 0.04, 0.28, 0.62, 0.89, 0.10, 0.41, 0.04],
       '1/z': [0.58, 0.68, 0.58, 0.56, 0.72, 0.74, 0.76, 0.80, 0.54, 0.33, 0.27, 0.21, 0.25, 0.27, 0.74, 0.41, 0.54, 0.37, 0.52]})

data = data.sort_values('City')

for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        filepath = os.path.join(directory, filename)
        print(filename)
        pandas_df=pd.read_csv(filepath)
        df=pd.DataFrame(pandas_df)
        area_init=df.loc[df['year'] == 1985,'area_km2']
        area_finit=df.loc[ df['year'] == 2014, 'area_km2']
        
        rate=(((area_finit[290]/area_init[0]))**(1/30)-1)*100
        radius=df.loc[df['year'] == 1985,'radial_distance_km']
        rate_collection.append(rate)
        
        ratio_area=area_init[0]/area_init[1:].sum()
        
        distance_avg=radius[1:].sum()/((len(radius)-1)*area_finit[290]**0.5)
        ratio_collection.append(ratio_area)
        radius_collection.append(distance_avg)

data['ratio_1985']=ratio_collection
data['radius_1985']=radius_collection
data['LCC_growth _rate']=rate_collection 

# Create a color map for cities
n_cities = len(data)
colors = cm.tab20(np.linspace(0, 1, n_cities))  # Using tab20 colormap for distinct colors
# Alternative: colors = cm.rainbow(np.linspace(0, 1, n_cities))

# Create a consistent color dictionary for each city
city_colors = {city: colors[i] for i, city in enumerate(data['City'])}

# Calculate all correlations
corr_radius=np.corrcoef(data['radius_1985'],data['beta'])
corr_ratio=np.corrcoef(data['ratio_1985'],data['beta'])
corr_bonus=np.corrcoef(data['radius_1985'],data['ratio_1985'])
corr_rate=np.corrcoef(data['LCC_growth _rate'],data['beta'])

res_radius=stats.spearmanr(data['radius_1985'],data['beta'])
res_ratio= stats.spearmanr(data['ratio_1985'],data['beta'])  
res_nonus= stats.spearmanr(data['radius_1985'],data['ratio_1985'])
res_rate= stats.spearmanr(data['LCC_growth _rate'],data['beta'])

# Calculate p-values for Pearson correlations
pearson_p_radius = stats.pearsonr(data['radius_1985'],data['beta'])[1]
pearson_p_ratio = stats.pearsonr(data['ratio_1985'],data['beta'])[1]
pearson_p_bonus = stats.pearsonr(data['radius_1985'],data['ratio_1985'])[1]
pearson_p_rate = stats.pearsonr(data['LCC_growth _rate'],data['beta'])[1]

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

# PLOT 1: Ratio vs Beta
plt.figure(figsize=(14, 8))
fit_ratio=np.polyfit(data['ratio_1985'],data['beta'],1)

# Plot points with different colors
for city in data['City']:
    city_data = data[data['City'] == city]
    plt.scatter(city_data['ratio_1985'], city_data['beta'], 
               color=city_colors[city], s=100, label=city, edgecolors='black', linewidth=0.5)

# Plot fit line
x_fit = np.array([data['ratio_1985'].min(), data['ratio_1985'].max()])
plt.plot(x_fit, fit_ratio[0]*x_fit+fit_ratio[1], '--', color='black',
         label=f'Linear fit: y={fit_ratio[0]:.2f}x + {fit_ratio[1]:.2f}', linewidth=2, alpha=0.7)

# Add city labels
add_city_labels(plt.gca(), data['ratio_1985'], data['beta'], data['City'])

# Add correlation statistics as text box
textstr = f'Pearson r = {corr_ratio[0,1]:.2f} (p = {pearson_p_ratio:.2f})\n' + \
          f'Spearman ρ = {res_ratio.correlation:.2f} (p = {res_ratio.pvalue:.2f})'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

plt.xlabel(r"$\varphi_{area}$", fontsize=12)
plt.ylabel(r"$\beta$", fontsize=12)
plt.title(r"Correlation: $\varphi_{area}$ vs $\beta$", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ratio_vs_beta_correlation_labeled.png', dpi=300, bbox_inches='tight')
plt.show()

# PLOT 2: Radius vs Beta
plt.figure(figsize=(14, 8))
fit=np.polyfit(data['radius_1985'],data['beta'],1)

# Plot points with different colors
for city in data['City']:
    city_data = data[data['City'] == city]
    plt.scatter(city_data['radius_1985'], city_data['beta'], 
               color=city_colors[city], s=100, label=city, edgecolors='black', linewidth=0.5)

# Plot fit line
x_fit = np.array([data['radius_1985'].min(), data['radius_1985'].max()])
plt.plot(x_fit, fit[0]*x_fit+fit[1], '--', color='black',
         label=f'Linear fit: y={fit[0]:.2f}x + {fit[1]:.2f}', linewidth=2, alpha=0.7)

# Add city labels
add_city_labels(plt.gca(), data['radius_1985'], data['beta'], data['City'])

# Add correlation statistics as text box
textstr = f'Pearson r = {corr_radius[0,1]:.2f} (p = {pearson_p_radius:.2f})\n' + \
          f'Spearman ρ = {res_radius.correlation:.2f} (p = {res_radius.pvalue:.2f})'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

plt.xlabel( r"$\bar{r}_{clusters}$", fontsize=12)
plt.ylabel(r"$\beta$", fontsize=12)
plt.title(r"Correlation: $\bar{r}_{clusters}$ (1985) vs $\beta$ ", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('radius_vs_beta_correlation_labeled.png', dpi=300, bbox_inches='tight')
plt.show()

# PLOT 3: Radius vs Ratio
plt.figure(figsize=(14, 8))
fit_bonus=np.polyfit(data['radius_1985'],data['ratio_1985'],1)

# Plot points with different colors
for city in data['City']:
    city_data = data[data['City'] == city]
    plt.scatter(city_data['radius_1985'], city_data['ratio_1985'], 
               color=city_colors[city], s=100, label=city, edgecolors='black', linewidth=0.5)

# Plot fit line
x_fit = np.array([data['radius_1985'].min(), data['radius_1985'].max()])
plt.plot(x_fit, fit_bonus[0]*x_fit+fit_bonus[1], '--', color='black',
         label=f'Linear fit: y={fit_bonus[0]:.2f}x + {fit_bonus[1]:.2f}', linewidth=2, alpha=0.7)

# Add city labels
add_city_labels(plt.gca(), data['radius_1985'], data['ratio_1985'], data['City'])

# Add correlation statistics as text box
textstr = f'Pearson r = {corr_bonus[0,1]:.2f} (p = {pearson_p_bonus:.2f})\n' + \
          f'Spearman ρ = {res_nonus.correlation:.2f} (p = {res_nonus.pvalue:.2f})'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

plt.xlabel('Radius 1985', fontsize=12)   
plt.ylabel('Ratio Area 1985', fontsize=12)
plt.title('Correlation: Radius vs Ratio Area (1985)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('radius_vs_ratio_correlation_labeled.png', dpi=300, bbox_inches='tight')
plt.show()

# PLOT 4: LCC Growth Rate vs Beta
plt.figure(figsize=(14, 8))
fit_rate=np.polyfit(data['LCC_growth _rate'],data['beta'],1)

# Plot points with different colors
for city in data['City']:
    city_data = data[data['City'] == city]
    plt.scatter(city_data['LCC_growth _rate'], city_data['beta'], 
               color=city_colors[city], s=100, label=city, edgecolors='black', linewidth=0.5)

# Plot fit line
x_fit = np.array([data['LCC_growth _rate'].min(), data['LCC_growth _rate'].max()])
plt.plot(x_fit, fit_rate[0]*x_fit+fit_rate[1], '--', color='black',
         label=f'Linear fit: y={fit_rate[0]:.2f}x + {fit_rate[1]:.2f}', linewidth=2, alpha=0.7)

# Add city labels
add_city_labels(plt.gca(), data['LCC_growth _rate'], data['beta'], data['City'])

# Add correlation statistics as text box
textstr = f'Pearson r = {corr_rate[0,1]:.2f} (p = {pearson_p_rate:.2f})\n' + \
          f'Spearman ρ = {res_rate.correlation:.2f} (p = {res_rate.pvalue:.2f})'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

plt.xlabel('LCC Growth Rate (1985-2015)', fontsize=12)  
plt.ylabel(r"$\beta$", fontsize=12)
plt.title(r"Correlation: LCC Growth Rate vs $\beta$", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lcc_growth_rate_vs_beta_correlation_labeled.png', dpi=300, bbox_inches='tight')
plt.show()
# # PLOT 5: Beta vs LCC Growth Rate (reversed axes - as in original)
# plt.figure(figsize=(14, 8))
# # For reversed axes, we need to fit beta as x and growth as y
# fit_reversed=np.polyfit(data['beta'],data['LCC_growth _rate'],1)

# # Plot points with different colors
# for city in data['City']:
#     city_data = data[data['City'] == city]
#     plt.scatter(city_data['beta'], city_data['LCC_growth _rate'], 
#                color=city_colors[city], s=100, label=city, edgecolors='black', linewidth=0.5)

# # Plot fit line
# x_fit = np.array([data['beta'].min(), data['beta'].max()])
# plt.plot(x_fit, fit_reversed[0]*x_fit+fit_reversed[1], '--', color='black',
#          label=f'Linear fit: y={fit_reversed[0]:.2f}x + {fit_reversed[1]:.2f}', linewidth=2, alpha=0.7)

# # Add city labels
# add_city_labels(plt.gca(), data['beta'], data['LCC_growth _rate'], data['City'])

# # Add correlation statistics as text box
# textstr = f'Pearson r = {corr_rate[0,1]:.4f} (p = {pearson_p_rate:.4f})\n' + \
#           f'Spearman ρ = {res_rate.correlation:.4f} (p = {res_rate.pvalue:.4f})'
# props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
# plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=11,
#         verticalalignment='top', bbox=props)

# plt.xlabel(r"$\beta$", fontsize=12)
# plt.ylabel("Growth of LCC from 1985 to 2015", fontsize=12)
# plt.title(r"Correlation: $\beta$ vs LCC Growth Rate", fontsize=14, fontweight='bold')
# # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig('beta_vs_lcc_growth_correlation_labeled.png', dpi=300, bbox_inches='tight')
# plt.show()

# Create a simplified plot without legend for cleaner visualization
# PLOT 6: Summary plot without legend (cleaner version)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Subplot 1: Ratio vs Beta
ax1 = axes[0, 0]
for city in data['City']:
    city_data = data[data['City'] == city]
    ax1.scatter(city_data['ratio_1985'], city_data['beta'], 
               color=city_colors[city], s=100, edgecolors='black', linewidth=0.5)
x_fit = np.array([data['ratio_1985'].min(), data['ratio_1985'].max()])
ax1.plot(x_fit, fit_ratio[0]*x_fit+fit_ratio[1], '--', color='black', linewidth=2, alpha=0.7)
add_city_labels(ax1, data['ratio_1985'], data['beta'], data['City'], fontsize=6)
textstr = f'r = {corr_ratio[0,1]:.3f}, ρ = {res_ratio.correlation:.3f}'
ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax1.set_xlabel('Ratio Area 1985', fontsize=11)
ax1.set_ylabel('Beta Exponent', fontsize=11)
ax1.set_title('Ratio Area vs Beta', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Subplot 2: Radius vs Beta
ax2 = axes[0, 1]
for city in data['City']:
    city_data = data[data['City'] == city]
    ax2.scatter(city_data['radius_1985'], city_data['beta'], 
               color=city_colors[city], s=100, edgecolors='black', linewidth=0.5)
x_fit = np.array([data['radius_1985'].min(), data['radius_1985'].max()])
ax2.plot(x_fit, fit[0]*x_fit+fit[1], '--', color='black', linewidth=2, alpha=0.7)
add_city_labels(ax2, data['radius_1985'], data['beta'], data['City'], fontsize=6)
textstr = f'r = {corr_radius[0,1]:.3f}, ρ = {res_radius.correlation:.3f}'
ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax2.set_xlabel('Radius 1985', fontsize=11)
ax2.set_ylabel('Beta Exponent', fontsize=11)
ax2.set_title('Radius vs Beta', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Subplot 3: LCC Growth Rate vs Beta
ax3 = axes[1, 0]
for city in data['City']:
    city_data = data[data['City'] == city]
    ax3.scatter(city_data['LCC_growth _rate'], city_data['beta'], 
               color=city_colors[city], s=100, edgecolors='black', linewidth=0.5)
x_fit = np.array([data['LCC_growth _rate'].min(), data['LCC_growth _rate'].max()])
ax3.plot(x_fit, fit_rate[0]*x_fit+fit_rate[1], '--', color='black', linewidth=2, alpha=0.7)
add_city_labels(ax3, data['LCC_growth _rate'], data['beta'], data['City'], fontsize=6)
textstr = f'r = {corr_rate[0,1]:.3f}, ρ = {res_rate.correlation:.3f}'
ax3.text(0.02, 0.98, textstr, transform=ax3.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax3.set_xlabel('LCC Growth Rate', fontsize=11)
ax3.set_ylabel('Beta Exponent', fontsize=11)
ax3.set_title('Growth Rate vs Beta', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Subplot 4: Radius vs Ratio
ax4 = axes[1, 1]
for city in data['City']:
    city_data = data[data['City'] == city]
    ax4.scatter(city_data['radius_1985'], city_data['ratio_1985'], 
               color=city_colors[city], s=100, edgecolors='black', linewidth=0.5)
x_fit = np.array([data['radius_1985'].min(), data['radius_1985'].max()])
ax4.plot(x_fit, fit_bonus[0]*x_fit+fit_bonus[1], '--', color='black', linewidth=2, alpha=0.7)
add_city_labels(ax4, data['radius_1985'], data['ratio_1985'], data['City'], fontsize=6)
textstr = f'r = {corr_bonus[0,1]:.3f}, ρ = {res_nonus.correlation:.3f}'
ax4.text(0.02, 0.98, textstr, transform=ax4.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax4.set_xlabel('Radius 1985', fontsize=11)
ax4.set_ylabel('Ratio Area 1985', fontsize=11)
ax4.set_title('Radius vs Ratio', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

plt.suptitle('Urban Morphology Correlations - All Cities', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('all_correlations_summary.png', dpi=300, bbox_inches='tight')
plt.show()

# Print correlation analysis results
print("\n" + "="*80)
print("--- Correlation Analysis for initial state 1985 ---")
print("="*80)
print("\nPEARSON CORRELATION: ")
print(f'correlation radius vs beta: {corr_radius[0,1]:.4f} (p = {pearson_p_radius:.4f})')
print(f'correlation ratio vs beta: {corr_ratio[0,1]:.4f} (p = {pearson_p_ratio:.4f})')
print(f'correlation radius vs ratio: {corr_bonus[0,1]:.4f} (p = {pearson_p_bonus:.4f})')

print("\nSPEARMAN CORRELATION:")
print(f'correlation radius vs beta: {res_radius.correlation:.4f} (p = {res_radius.pvalue:.4f})')
print(f'correlation ratio vs beta: {res_ratio.correlation:.4f} (p = {res_ratio.pvalue:.4f})')
print(f'correlation radius vs ratio: {res_nonus.correlation:.4f} (p = {res_nonus.pvalue:.4f})')

print("\n" + "="*80)
print("--- Correlation Analysis for LCC growth rate ---")
print("="*80)
print("\nPEARSON CORRELATION: ")
print(f'correlation LCC growth rate vs beta: {corr_rate[0,1]:.4f} (p = {pearson_p_rate:.4f})')

print("\nSPEARMAN CORRELATION:")
print(f'correlation LCC growth rate vs beta: {res_rate.correlation:.4f} (p = {res_rate.pvalue:.4f})')

# ============================================================================
# GENERATE OUTPUT FILES
# ============================================================================

def interpret_correlation(rho, p_value):
    """Interpret correlation strength and significance"""
    # Strength interpretation
    abs_rho = abs(rho)
    if abs_rho < 0.20:
        strength = "Very weak"
    elif abs_rho < 0.40:
        strength = "Weak"
    elif abs_rho < 0.60:
        strength = "Moderate"
    elif abs_rho < 0.80:
        strength = "Strong"
    else:
        strength = "Very strong"
    
    # Direction
    direction = "positive" if rho > 0 else "negative"
    
    # Significance
    if p_value < 0.001:
        significance = "highly significant (p < 0.001)"
    elif p_value < 0.01:
        significance = "very significant (p < 0.01)"
    elif p_value < 0.05:
        significance = "significant (p < 0.05)"
    else:
        significance = "not significant (p ≥ 0.05)"
    
    return strength, direction, significance

# Create formatted text report
output_text = []
output_text.append("="*80)
output_text.append("CORRELATION ANALYSIS REPORT")
output_text.append("Urban Morphology and Growth Patterns")
output_text.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
output_text.append("="*80)
output_text.append("")
output_text.append(f"Sample size: {len(data)} cities")
output_text.append("")
output_text.append("Cities analyzed:")
for i, city in enumerate(sorted(data['City']), 1):
    output_text.append(f"  {i:2}. {city}")
output_text.append("")

# Section 1: Initial State Analysis (1985)
output_text.append("="*80)
output_text.append("SECTION 1: INITIAL STATE ANALYSIS (1985)")
output_text.append("="*80)
output_text.append("")

# Radius vs Beta
output_text.append("-" * 80)
output_text.append("1.1 RADIUS vs BETA EXPONENT")
output_text.append("-" * 80)
output_text.append(f"  Pearson correlation:   r   = {corr_radius[0,1]:.4f} (p = {pearson_p_radius:.4f})")
output_text.append(f"  Spearman correlation:  rho = {res_radius.correlation:.4f} (p = {res_radius.pvalue:.4f})")
strength, direction, significance = interpret_correlation(res_radius.correlation, res_radius.pvalue)
output_text.append(f"  Interpretation:        {strength} {direction} correlation, {significance}")
output_text.append("")

# Ratio vs Beta
output_text.append("-" * 80)
output_text.append("1.2 RATIO AREA vs BETA EXPONENT")
output_text.append("-" * 80)
output_text.append(f"  Pearson correlation:   r   = {corr_ratio[0,1]:.4f} (p = {pearson_p_ratio:.4f})")
output_text.append(f"  Spearman correlation:  rho = {res_ratio.correlation:.4f} (p = {res_ratio.pvalue:.4f})")
strength, direction, significance = interpret_correlation(res_ratio.correlation, res_ratio.pvalue)
output_text.append(f"  Interpretation:        {strength} {direction} correlation, {significance}")
output_text.append("")

# Radius vs Ratio
output_text.append("-" * 80)
output_text.append("1.3 RADIUS vs RATIO AREA")
output_text.append("-" * 80)
output_text.append(f"  Pearson correlation:   r   = {corr_bonus[0,1]:.4f} (p = {pearson_p_bonus:.4f})")
output_text.append(f"  Spearman correlation:  rho = {res_nonus.correlation:.4f} (p = {res_nonus.pvalue:.4f})")
strength, direction, significance = interpret_correlation(res_nonus.correlation, res_nonus.pvalue)
output_text.append(f"  Interpretation:        {strength} {direction} correlation, {significance}")
output_text.append("")

# Section 2: Growth Rate Analysis
output_text.append("="*80)
output_text.append("SECTION 2: LCC GROWTH RATE ANALYSIS (1985-2015)")
output_text.append("="*80)
output_text.append("")

output_text.append("-" * 80)
output_text.append("2.1 LCC GROWTH RATE vs BETA EXPONENT")
output_text.append("-" * 80)
output_text.append(f"  Pearson correlation:   r   = {corr_rate[0,1]:.4f} (p = {pearson_p_rate:.4f})")
output_text.append(f"  Spearman correlation:  rho = {res_rate.correlation:.4f} (p = {res_rate.pvalue:.4f})")
strength, direction, significance = interpret_correlation(res_rate.correlation, res_rate.pvalue)
output_text.append(f"  Interpretation:        {strength} {direction} correlation, {significance}")
output_text.append("")

# Summary section
output_text.append("="*80)
output_text.append("SUMMARY OF FINDINGS")
output_text.append("="*80)
output_text.append("")
output_text.append("Correlation Strength Guidelines (Spearman's ρ):")
output_text.append("  |ρ| < 0.20    Very weak")
output_text.append("  0.20-0.39     Weak")
output_text.append("  0.40-0.59     Moderate")
output_text.append("  0.60-0.79     Strong")
output_text.append("  0.80-1.00     Very strong")
output_text.append("")
output_text.append("Statistical Significance:")
output_text.append("  p < 0.05      Statistically significant")
output_text.append("  p < 0.01      Very significant")
output_text.append("  p < 0.001     Highly significant")
output_text.append("")
output_text.append("="*80)

# Write to text file
report_text = "\n".join(output_text)
with open('correlation_analysis_report_clusters.txt', 'w', encoding='utf-8') as f:
    f.write(report_text)

print("\n" + "="*80)
print("OUTPUT FILES GENERATED:")
print("="*80)
print("✓ correlation_analysis_report_clusters.txt - Detailed formatted report")

# Create CSV summary with both Pearson and Spearman p-values
results_summary = {
    'Variable_Pair': [
        'Radius_1985 vs Beta',
        'Ratio_1985 vs Beta',
        'Radius_1985 vs Ratio_1985',
        'LCC_Growth_Rate vs Beta'
    ],
    'Pearson_r': [
        corr_radius[0,1],
        corr_ratio[0,1],
        corr_bonus[0,1],
        corr_rate[0,1]
    ],
    'Pearson_p_value': [
        pearson_p_radius,
        pearson_p_ratio,
        pearson_p_bonus,
        pearson_p_rate
    ],
    'Spearman_rho': [
        res_radius.correlation,
        res_ratio.correlation,
        res_nonus.correlation,
        res_rate.correlation
    ],
    'Spearman_p_value': [
        res_radius.pvalue,
        res_ratio.pvalue,
        res_nonus.pvalue,
        res_rate.pvalue
    ],
    'Significant_at_0.05': [
        'Yes' if res_radius.pvalue < 0.05 else 'No',
        'Yes' if res_ratio.pvalue < 0.05 else 'No',
        'Yes' if res_nonus.pvalue < 0.05 else 'No',
        'Yes' if res_rate.pvalue < 0.05 else 'No'
    ]
}

results_df = pd.DataFrame(results_summary)
results_df.to_csv('correlation_results_summary.csv', index=False)
print("✓ correlation_results_summary.csv - Tabular summary for further analysis")
print("✓ 6 PNG figures with city labels and correlation statistics")
print("  - ratio_vs_beta_correlation_labeled.png")
print("  - radius_vs_beta_correlation_labeled.png")
print("  - radius_vs_ratio_correlation_labeled.png")
print("  - lcc_growth_rate_vs_beta_correlation_labeled.png")
print("  - beta_vs_lcc_growth_correlation_labeled.png")
print("  - all_correlations_summary.png (4-panel summary)")
print("="*80)

# Create a data summary CSV with all city data
city_summary = data[['City', 'alpha', 'beta', '1/z', 'ratio_1985', 'radius_1985', 'LCC_growth _rate']]
city_summary = city_summary.round(4)
city_summary.to_csv('city_data_summary.csv', index=False)
print("✓ city_data_summary.csv - Complete dataset with all city metrics")
print("="*80)