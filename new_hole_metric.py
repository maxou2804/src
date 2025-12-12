import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from scipy import stats
from adjustText import adjust_text
import matplotlib.cm as cm


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

data=pd.DataFrame({'City': ['Ningbo','Chengdu Deyang', 'Beijing Lafang','Changzhou','Bengalore','Kolkata','Paris','Bangkok','Cairo','Guatemala City','Johannesburg','London','Mexico City','Nairobi','Santiago','Sao Paulo','Tehran','Las Vegas','Atlanta'],
      'alpha':[0.56, 0.53, 0.54, 0.54, 0.55, 0.52, 0.52 ,0.53, 0.53, 0.52, 0.58, 0.54, 0.55, 0.56, 0.58, 0.51, 0.55, 0.55, 0.56],
      'beta': [0.44, 0.68, 0.41, 0.37, 0.83, 0.34, 0.56, 1.01, 0.37, 0.37, 0.07, 0.01, 0.04, 0.28, 0.62, 0.89, 0.10, 0.41, 0.04],
       '1/z': [0.58, 0.68, 0.58, 0.56, 0.72, 0.74, 0.76, 0.80, 0.54, 0.33, 0.27, 0.21, 0.25, 0.27, 0.74, 0.41, 0.54, 0.37, 0.52]})



directory="/Users/mika/Documents/DATA/src/emptiness_metric.csv"




with open(directory, 'r') as f:
    reader = csv.reader(f)
    cities = next(reader)
    data_holes = [[float(val) for val in row] for row in reader]

years = list(range(1985, 2016))

for i, city in enumerate(cities):
    values = [row[i] for row in data_holes]
    plt.plot(years, values, label=city.strip())

plt.xlabel('Year')
plt.ylabel('Value')
plt.title('Values for Each City (1985-2015)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.savefig('hole_evolution_time.png',dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()



cities_kappa_1=['Cairo','Ningbo','Beijing Lafang','Changzhou','Las Vegas']

for i, city in enumerate(cities_kappa_1):
    values = [row[i] for row in data_holes]
    plt.plot(years, values, label=city.strip())

plt.xlabel('Year')
plt.ylabel('Value')
plt.title('Values for Each City (1985-2015)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

plt.tight_layout()
plt.show()

avg_collection=[]
var_col=[]

for i, city in enumerate(cities):
    values = [row[i] for row in data_holes]
    avg=np.mean(values)
    variation=(values[-1]-values[0])/avg
    avg_collection.append(avg)
    var_col.append(variation)


data['hole_avg']=avg_collection
data['hole_var']=var_col



n_cities = len(data['City'])
colors = cm.tab20(np.linspace(0, 1, n_cities))  # Using tab20 colormap for distinct colors
city_colors = {city: colors[i] for i, city in enumerate(data['City'])}

# Plot points with different colors
for city in data['City']:
    city_data = data[data['City'] == city]
    plt.scatter(city_data['hole_avg'], city_data['beta'], 
               color=city_colors[city], s=100, label=city, edgecolors='black', linewidth=0.5)
corr_hole_avg=np.corrcoef(data['hole_avg'],data['beta'])
res_spearman_hole_avg=stats.spearmanr(data['hole_avg'],data['beta'])
res_pearson_hole_avg = stats.pearsonr(data['hole_avg'],data['beta'])
# Plot fit line
x_fit = np.array([data['hole_avg'].min(), data['hole_avg'].max()])


# Add city labels
add_city_labels(plt.gca(), data['hole_avg'], data['beta'], data['City'])

# Add correlation statistics as text box
corr_hole_avg=np.corrcoef(data['hole_avg'],data['beta'])
res_spearman_hole_avg=stats.spearmanr(data['hole_avg'],data['beta'])
res_pearson_hole_avg = stats.pearsonr(data['hole_avg'],data['beta'])
textstr = f'Pearson r = {res_pearson_hole_avg[0]:.2f} (p = {res_pearson_hole_avg[1]:.2f})\n' + \
          f'Spearman ρ = {res_spearman_hole_avg.correlation:.2f} (p = {res_spearman_hole_avg.pvalue:.2f})'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

plt.xlabel(r"$\langle A_{non-urb} \rangle$", fontsize=12)
plt.ylabel(r"$\beta$", fontsize=12)
plt.title(r"Correlation: $\langle A_{non-urb} \rangle$ vs $\beta$ within the front of LCC", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hole_avg_vs_beta_correlation_labeled.png', dpi=300, bbox_inches='tight')
plt.show()


for city in data['City']:
    city_data = data[data['City'] == city]
    plt.scatter(city_data['hole_var'], city_data['beta'], 
               color=city_colors[city], s=100, label=city, edgecolors='black', linewidth=0.5)
corr_hole_avg=np.corrcoef(data['hole_var'],data['beta'])
res_spearman_hole_avg=stats.spearmanr(data['hole_var'],data['beta'])
res_pearson_hole_avg = stats.pearsonr(data['hole_var'],data['beta'])
# Plot fit line
x_fit = np.array([data['hole_var'].min(), data['hole_var'].max()])


# Add city labels
add_city_labels(plt.gca(), data['hole_var'], data['beta'], data['City'])

# Add correlation statistics as text box
corr_hole_avg=np.corrcoef(data['hole_var'],data['beta'])
res_spearman_hole_avg=stats.spearmanr(data['hole_var'],data['beta'])
res_pearson_hole_avg = stats.pearsonr(data['hole_var'],data['beta'])
textstr = f'Pearson r = {res_pearson_hole_avg[0]:.2f} (p = {res_pearson_hole_avg[1]:.2f})\n' + \
          f'Spearman ρ = {res_spearman_hole_avg.correlation:.2f} (p = {res_spearman_hole_avg.pvalue:.2f})'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

plt.xlabel(r"$\langle dA_{non-urb} \rangle$", fontsize=12)
plt.ylabel(r"$\beta$", fontsize=12)
plt.title(r"Correlation: $\langle dA_{non-urb} \rangle$ vs $\beta$ within the front of LCC", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hole_var_vs_beta_correlation_labeled.png', dpi=300, bbox_inches='tight')
plt.show()


