import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import csv
import ast
import numpy as np
import matplotlib.cm as cm
from adjustText import adjust_text

df_gruyere=pd.read_csv('/Users/mika/Documents/DATA/src/gruyere_metric_2015.csv')


# data={'City': ['Ningbo','Chengdu Deyang', 'Beijing Lafang','Changzhou','Bengalore','Kolkata','Paris','Bangkok','Cairo','Guatemala City','Johannesburg','London','Mexico City','Nairobi','Santiago','Sao Paulo','Tehran','Las Vegas','Atlanta'],
#       'alpha':[0.56, 0.53, 0.54, 0.54, 0.55, 0.52, 0.52 ,0.53, 0.53, 0.52, 0.58, 0.54, 0.55, 0.56, 0.58, 0.51, 0.55, 0.55, 0.56],
#       'beta': [0.44, 0.68, 0.41, 0.37, 0.83, 0.34, 0.56, 1.01, 0.37, 0.37, 0.07, 0.01, 0.04, 0.28, 0.62, 0.89, 0.10, 0.41, 0.04],
#        '1/z': [0.58, 0.68, 0.58, 0.56, 0.72, 0.74, 0.76, 0.80, 0.54, 0.33, 0.27, 0.21, 0.25, 0.27, 0.74, 0.41, 0.54, 0.37, 0.52]}


area_population=[3.63*10**(-4),                   3.81*10**(-4), 2.73*10**(-4),         3.85*10**(-4), 
                 (1.53*10**(-4)+6.64*10**(-5))/2, 1.87*10**(-4), (8.23+3.73)/2*10**(-4),(5.38*10**(-4)+1.17*10**(-3))/2,
                 8.09*10**(-5),                   5.54*10**(-5), 3.56*10**(-4),         2.5*10**(-4), 
                 5.12*10**(-5),                   8*10**(-5),    2*10**(-4),           (1.31*10**(-4)+4.44*10**(-5))/2,
                 (3.66+1.90)/2*10**(-4),         (6.64+3.39)/2*10**(-4),               (5.64+1.52)/2*10**(-4)]

df_gruyere['area_population']=area_population


cities=df_gruyere['City'].tolist()


df_gruyere=pd.read_csv('/Users/mika/Documents/DATA/src/data_3_CV_report_render.csv')

# Sous ensemble selection
low_beta=df_gruyere[df_gruyere["City"].isin(['Kolkata','Nairobi','Atlanta','London','Johannesburg','Mexico City','Tehran'])]

high_density=df_gruyere[df_gruyere["City"].isin(['Las Vegas','Paris','Chengdu Deyang','Bangkok'])]

high_kappa=df_gruyere[df_gruyere["City"].isin(['Ningbo','Beijing','Las Vegas','Changzhou','Bengalore','Bangkok','Santiago','Paris','Kolkata','Tehran','Chengdu Deyang'])]
high_kappa=df_gruyere[df_gruyere["City"].isin(['Ningbo','Beijing','Las Vegas','Changzhou','Bengalore','Santiago','Paris','Kolkata','Chengdu Deyang'])]

kappa_1=df_gruyere[df_gruyere["City"].isin(['Cairo','Beijing Lafang','Las Vegas', 'Changzhou', 'Ningbo'])]


#
n_cities = len(high_kappa['City'])
colors = cm.tab20(np.linspace(0, 1, n_cities))  # Using tab20 colormap for distinct colors
city_colors = {city: colors[i] for i, city in enumerate(high_kappa['City'])}

# plt.figure()
# plt.scatter(low_beta['area_population'],low_beta['beta'])
# plt.xlabel('area/population')
# plt.ylabel('beta')
# plt.show()









# plt.figure()
# plt.plot(df_gruyere['metric_perimeter'],df_gruyere['beta'],'o')
# plt.ylabel(r'\beta')
# plt.xlabel(r'\frac{A_{non-urb}}/{A_urb}')
# texts = [plt.text(df_gruyere['metric_perimeter'].iloc[i],df_gruyere['beta'].iloc[i], cities[i]) for i in range(len(cities))] 
# adjust_text(texts)
# plt.savefig('metric_perimeter_beta')
# plt.show()


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


add_city_labels=add_city_labels_with_adjusttext



# PLOT 1: metric perimeter vs Beta
plt.figure(figsize=(14, 8))
fit_ratio=np.polyfit(high_kappa['metric_perimeter'],high_kappa['beta'],1)

# Plot points with different colors
for city in high_kappa['City']:
    city_data = high_kappa[high_kappa['City'] == city]
    plt.scatter(city_data['metric_perimeter'], city_data['beta'], 
               color=city_colors[city], s=100, label=city, edgecolors='black', linewidth=0.5)

# Plot fit line
x_fit = np.array([high_kappa['metric_perimeter'].min(), high_kappa['metric_perimeter'].max()])
plt.plot(x_fit, fit_ratio[0]*x_fit+fit_ratio[1], '--', color='black',
         label=f'Linear fit: y={fit_ratio[0]:.2f}x + {fit_ratio[1]:.2f}', linewidth=2, alpha=0.7)

# Add city labels
add_city_labels(plt.gca(), high_kappa['metric_perimeter'], high_kappa['beta'], high_kappa['City'])

# Add correlation statistics as text box
corr_metric_perimeter=np.corrcoef(high_kappa['metric_perimeter'],high_kappa['beta'])
res_spearman_metric_perimeter=stats.spearmanr(high_kappa['metric_perimeter'],high_kappa['beta'])
res_pearson_metric_perimeter = stats.pearsonr(high_kappa['metric_perimeter'],high_kappa['beta'])
textstr = f'Pearson r = {res_pearson_metric_perimeter[0]:.2f} (p = {res_pearson_metric_perimeter[1]:.2f})\n' + \
          f'Spearman ρ = {res_spearman_metric_perimeter.correlation:.2f} (p = {res_spearman_metric_perimeter.pvalue:.2f})'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

plt.xlabel(r"$A_{non-urb}/A_{urb}$", fontsize=16)
plt.ylabel(r"$\beta$", fontsize=16)
plt.title(r"Correlation: $A_{non-urb}/A_{urb}$ vs $\beta$ within the front of LCC", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
# plt.savefig('metric_perimeter_vs_beta_correlation_labeled.png', dpi=300, bbox_inches='tight')
plt.show()


# PLOT 2: hull vs Beta
plt.figure(figsize=(14, 8))
fit_ratio=np.polyfit(df_gruyere['metric_hull'],df_gruyere['beta'],1)

# Plot points with different colors
for city in df_gruyere['City']:
    city_data = df_gruyere[df_gruyere['City'] == city]
    plt.scatter(city_data['metric_hull'], city_data['beta'], 
               color=city_colors[city], s=100, label=city, edgecolors='black', linewidth=0.5)

# Plot fit line
x_fit = np.array([df_gruyere['metric_hull'].min(), df_gruyere['metric_hull'].max()])
plt.plot(x_fit, fit_ratio[0]*x_fit+fit_ratio[1], '--', color='black',
         label=f'Linear fit: y={fit_ratio[0]:.2f}x + {fit_ratio[1]:.2f}', linewidth=2, alpha=0.7)

# Add city labels
add_city_labels(plt.gca(), df_gruyere['metric_hull'], df_gruyere['beta'], df_gruyere['City'])

# Add correlation statistics as text box
corr_metric_perimeter=np.corrcoef(df_gruyere['metric_hull'],df_gruyere['beta'])
res_spearman_metric_perimeter=stats.spearmanr(df_gruyere['metric_hull'],df_gruyere['beta'])
res_pearson_metric_perimeter = stats.pearsonr(df_gruyere['metric_hull'],df_gruyere['beta'])
textstr = f'Pearson r = {res_pearson_metric_perimeter[0]:.2f} (p = {res_pearson_metric_perimeter[1]:.2f})\n' + \
          f'Spearman ρ = {res_spearman_metric_perimeter.correlation:.2f} (p = {res_spearman_metric_perimeter.pvalue:.2f})'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

plt.xlabel(r"$A_{non-urb}/A_{urb}$", fontsize=12)
plt.ylabel(r"$\beta$", fontsize=12)
plt.title(r"Correlation: $A_{non-urb}/A_{urb}$ vs $\beta$ within the hull of LCC ", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
# plt.savefig('metric_hull_vs_beta_correlation_labeled.png', dpi=300, bbox_inches='tight')
plt.show()
# PLOT 3: bbox vs Beta
plt.figure(figsize=(14, 8))
fit_ratio=np.polyfit(df_gruyere['metric_bbox'],df_gruyere['beta'],1)

# Plot points with different colors
for city in df_gruyere['City']:
    city_data = df_gruyere[df_gruyere['City'] == city]
    plt.scatter(city_data['metric_bbox'], city_data['beta'], 
               color=city_colors[city], s=100, label=city, edgecolors='black', linewidth=0.5)

# Plot fit line
x_fit = np.array([df_gruyere['metric_bbox'].min(), df_gruyere['metric_bbox'].max()])
plt.plot(x_fit, fit_ratio[0]*x_fit+fit_ratio[1], '--', color='black',
         label=f'Linear fit: y={fit_ratio[0]:.2f}x + {fit_ratio[1]:.2f}', linewidth=2, alpha=0.7)

# Add city labels
add_city_labels(plt.gca(), df_gruyere['metric_bbox'], df_gruyere['beta'], df_gruyere['City'])

# Add correlation statistics as text box
corr_metric_perimeter=np.corrcoef(df_gruyere['metric_bbox'],df_gruyere['beta'])
res_spearman_metric_perimeter=stats.spearmanr(df_gruyere['metric_bbox'],df_gruyere['beta'])
res_pearson_metric_perimeter = stats.pearsonr(df_gruyere['metric_bbox'],df_gruyere['beta'])
textstr = f'Pearson r = {res_pearson_metric_perimeter[0]:.2f} (p = {res_pearson_metric_perimeter[1]:.2f})\n' + \
          f'Spearman ρ = {res_spearman_metric_perimeter.correlation:.2f} (p = {res_spearman_metric_perimeter.pvalue:.2f})'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

plt.xlabel(r"$A_{non-urb}/A_{urb}$", fontsize=12)
plt.ylabel(r"$\beta$", fontsize=12)
plt.title(r"Correlation: $A_{non-urb}/A_{urb}$ vs $\beta$ within the rectangle of LCC ", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
# plt.savefig('metric_bbox_vs_beta_correlation_labeled.png', dpi=300, bbox_inches='tight')
plt.show()











plt.figure()
plt.plot(area_population,df_gruyere['beta'],'o')
plt.xlabel('area/population')
plt.ylabel('beta')
texts = [plt.text(area_population[i],df_gruyere['beta'].iloc[i], cities[i]) for i in range(len(cities))] 
adjust_text(texts)
# plt.savefig('densities_beta')
plt.show()


plt.figure()
plt.scatter(high_density['metric_hull'],high_density['beta'])
texts = [plt.text(high_density['metric_hull'].iloc[i],high_density['beta'].iloc[i], high_density['City'].iloc[i]) for i in range(len( high_density['City']))] 
adjust_text(texts)
plt.xlabel('constraint density')
plt.ylabel('beta')
plt.show()


plt.figure()
plt.scatter(high_density['metric_hull'],high_density['beta'])
texts = [plt.text(high_density['metric_hull'].iloc[i],high_density['beta'].iloc[i], high_density['City'].iloc[i]) for i in range(len( high_density['City']))] 
adjust_text(texts)
plt.xlabel('constraint density')
plt.ylabel('beta')
plt.show()





# plt.figure()
# plt.scatter(kappa_1['metric_perimeter'],high_kappa['beta'])
# plt.xlabel('constraint density')
# plt.ylabel('beta')
# plt.show()


corr_beta_high=stats.spearmanr(high_kappa['beta'],high_kappa['metric_perimeter'])
print(f'correlation on high kappa cities constrain beta {corr_beta_high}')



corr_beta_high=stats.spearmanr(high_density['beta'],high_density['metric_perimeter'])
print(f'correlation on selected cities constrain beta {corr_beta_high}')

corr_beta_metric_perimeter=stats.spearmanr(df_gruyere['beta'],df_gruyere['metric_perimeter'])
print(f'correlation all cities constraint beta{corr_beta_metric_perimeter}')

corr_beta_densities=stats.spearmanr(df_gruyere['beta'],area_population)
print(corr_beta_densities)






# PLOT 2: hull vs Beta
plt.figure(figsize=(14, 8))
fit_ratio=np.polyfit((np.ones((len(df_gruyere)))-df_gruyere['metric_hull'])/df_gruyere['metric_hull'],df_gruyere['beta'],1)

# Plot points with different colors
for city in df_gruyere['City']:
    city_data = df_gruyere[df_gruyere['City'] == city]
    plt.scatter((np.ones((len(city_data['metric_hull'])))-city_data['metric_hull'])/city_data['metric_hull'], city_data['beta'], 
               color=city_colors[city], s=100, label=city, edgecolors='black', linewidth=0.5)

# Plot fit line
x_fit = np.array([df_gruyere['metric_hull'].min(), df_gruyere['metric_hull'].max()])
plt.plot(x_fit, fit_ratio[0]*x_fit+fit_ratio[1], '--', color='black',
         label=f'Linear fit: y={fit_ratio[0]:.2f}x + {fit_ratio[1]:.2f}', linewidth=2, alpha=0.7)

# Add city labels
add_city_labels(plt.gca(),(np.ones((len(df_gruyere)))-df_gruyere['metric_hull'])/df_gruyere['metric_hull'], df_gruyere['beta'], df_gruyere['City'])

# Add correlation statistics as text box
corr_metric_perimeter=np.corrcoef((np.ones((len(df_gruyere)))-df_gruyere['metric_hull'])/df_gruyere['metric_hull'],df_gruyere['beta'])
res_spearman_metric_perimeter=stats.spearmanr((np.ones((len(df_gruyere)))-df_gruyere['metric_hull'])/df_gruyere['metric_hull'],df_gruyere['beta'])
res_pearson_metric_perimeter = stats.pearsonr((np.ones((len(df_gruyere)))-df_gruyere['metric_hull'])/df_gruyere['metric_hull'],df_gruyere['beta'])
textstr = f'Pearson r = {res_pearson_metric_perimeter[0]:.2f} (p = {res_pearson_metric_perimeter[1]:.2f})\n' + \
          f'Spearman ρ = {res_spearman_metric_perimeter.correlation:.2f} (p = {res_spearman_metric_perimeter.pvalue:.2f})'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

plt.xlabel(r"A_second_urb/A_holes", fontsize=12)
plt.ylabel(r"$\beta$", fontsize=12)
plt.title(r"Correlation: $A_{non-urb}/A_{urb}$ vs $\beta$ within the hull of LCC ", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
# plt.savefig('A_second_A_hole_vs_beta_correlation_labeled.png', dpi=300, bbox_inches='tight')
plt.show()







plt.figure(figsize=(14, 8))
fit_ratio=np.polyfit(high_kappa['metric_hull'],high_kappa['beta'],1)

# Plot points with different colors
for city in high_kappa['City']:
    city_data = high_kappa[high_kappa['City'] == city]
    plt.scatter(city_data['metric_hull'], city_data['beta'], 
               color=city_colors[city], s=100, label=city, edgecolors='black', linewidth=0.5)

# Plot fit line
x_fit = np.array([high_kappa['metric_hull'].min(), high_kappa['metric_hull'].max()])
plt.plot(x_fit, fit_ratio[0]*x_fit+fit_ratio[1], '--', color='black',
         label=f'Linear fit: y={fit_ratio[0]:.2f}x + {fit_ratio[1]:.2f}', linewidth=2, alpha=0.7)

# Add city labels
add_city_labels(plt.gca(),high_kappa['metric_hull'], high_kappa['beta'], high_kappa['City'])

# Add correlation statistics as text box
corr_metric_perimeter=np.corrcoef(high_kappa['metric_hull'],high_kappa['beta'])
res_spearman_metric_perimeter=stats.spearmanr(high_kappa['metric_hull'],high_kappa['beta'])
res_pearson_metric_perimeter = stats.pearsonr(high_kappa['metric_hull'],high_kappa['beta'])
textstr = f'Pearson r = {res_pearson_metric_perimeter[0]:.2f} (p = {res_pearson_metric_perimeter[1]:.2f})\n' + \
          f'Spearman ρ = {res_spearman_metric_perimeter.correlation:.2f} (p = {res_spearman_metric_perimeter.pvalue:.2f})'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

plt.xlabel(r"$A_{non-urb}$", fontsize=12)
plt.ylabel(r"$\beta$", fontsize=12)
plt.title(r"Correlation: $A_{non-urb}$ vs $\beta$ within the hull of LCC ", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
# plt.savefig('high_kappa_hull_metric', dpi=300, bbox_inches='tight')
plt.show()













df_clusters=pd.read_csv('/Users/mika/Documents/DATA/src/quenching_metrics.py',on_bad_lines='skip')



# def compute_stats(filename):
#     with open(filename, 'r', encoding='utf-8') as f:
#         reader = csv.reader(f, delimiter=',', quotechar='"')
#         rows = list(reader)
    
#     mean_area_col = []
#     std_area_col=[]
#     mean_dist_col = []
#     std_dist_col=[]
#     for row in rows[1:]:  # Skip header if present, but in this case, first row is indices
#         areas = []
#         distances=[]
#         for cell in row:
#             if cell.strip():  # Skip empty cells
#                 try:
#                     data_dict = ast.literal_eval(cell.strip())
#                     if 'area_km2' in data_dict:
#                         areas.append(float(data_dict['area_km2']))
#                     if 'distance_km' in data_dict:
#                         distances.append(float(data_dict['distance_km']))
#                 except (ValueError, SyntaxError):
#                     pass  # Skip invalid entries
       
#         if areas:
#             mean = np.mean(areas)
#             std = np.std(areas)
#             mean_area_col.append(mean)
#             std_area_col.append(std)

#         if distances:
      
#             mean_dist=np.mean(distances)
#             std_dist=np.std(distances)
#             mean_dist_col.append(mean_dist)
#             std_dist_col.append(std_dist)

#     return mean_area_col,std_area_col,mean_dist_col,std_dist_col

# # Usage
# filename = 'non_urban_clusters_metric_1985.csv'
# mean_area,std_area,mean_dist,std_dist = compute_stats(filename)



# corr_mean =stats.spearmanr(df_gruyere['beta'], mean_area)
# corr_std= stats.spearmanr(df_gruyere['beta'],  std_area)
# print(corr_mean)
# print(corr_std)




# corr_mean =stats.spearmanr(df_gruyere['beta'], mean_dist)
# corr_std= stats.spearmanr(df_gruyere['beta'],  std_dist)
# print(corr_mean)
# print(corr_std)

# plt.figure()
# plt.plot(mean_dist,mean_area,'o')
# plt.xlabel('mean distance of constraints')
# plt.ylabel('mean area of constraints')
# plt.show()