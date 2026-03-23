import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from adjustText import adjust_text
import matplotlib.cm as cm
import matplotlib as mpl

def add_city_labels_with_adjusttext(ax, x_data, y_data, cities, fontsize=16):
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


mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['axes.labelsize']=16
mpl.rcParams['legend.fontsize']=16

high_color = "#f28482" 
low_color = "#4646C0" 


kappa=pd.read_csv('kappa.csv')

print(kappa['0'])

data['kappa']=kappa['0']

supra_linear_data=data[data['kappa']>=1]
sub_linear_data=data[data['kappa']<1] 


# Create a color map for cities
n_cities = len(data)
colors = cm.tab20(np.linspace(0, 1, n_cities))  # Using tab20 colormap for distinct colors
# Alternative: colors = cm.rainbow(np.linspace(0, 1, n_cities))

# Create a consistent color dictionary for each city
city_colors = {city: colors[i] for i, city in enumerate(data['City'])}

# PLOT 1: Ratio vs Beta



corr=np.corrcoef(data['kappa'],data['beta'])
res=stats.spearmanr(data['kappa'],data['beta'])
pearson_p = stats.pearsonr(data['kappa'],data['beta'])[1]

plt.figure(figsize=(14, 8))
fit_ratio=np.polyfit(data['kappa'],data['beta'],1)

# Plot points with different colors
plt.scatter(
    supra_linear_data['kappa'],
    supra_linear_data["beta"],
    color=high_color,
    s=120,
    edgecolors="black",
    linewidth=0.7,
    label="κ \geq 1",
    zorder=1
)

plt.scatter(
    sub_linear_data['kappa'],
    sub_linear_data["beta"],
    color=low_color,
    s=120,
    edgecolors="black",
    linewidth=0.7,
    label="κ<1",
    zorder=1
)

# Plot fit line
x_fit = np.array([data['kappa'].min(), data['kappa'].max()])
plt.plot(x_fit, fit_ratio[0]*x_fit+fit_ratio[1], '--', color='black',
         label=f'Linear fit: y={fit_ratio[0]:.2f}x + {fit_ratio[1]:.2f}', linewidth=2, alpha=0.3)

# Add city labels
add_city_labels_with_adjusttext(plt.gca(), data['kappa'], data['beta'], data['City'])

# Add correlation statistics as text box
textstr = f'Pearson r = {corr[0,1]:.2f} (p = {pearson_p:.2f})\n' + \
          f'Spearman ρ = {res.correlation:.2f} (p = {res.pvalue:.2f})'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=16,
        verticalalignment='top', bbox=props)

plt.xlabel(r"$\kappa$")
plt.ylabel(r"$\beta$" )
plt.grid(True, alpha=0.3)
plt.savefig('beta_vs_kappa.png')
plt.tight_layout()
plt.show()



# plt.figure(figsize=(14, 8))
# fit_ratio=np.polyfit(data['kappa'],data['beta'],1)

# # Plot points with different colors
# for city in data['City']:
#     city_data = data[data['City'] == city]
#     plt.scatter(city_data.index,city_data['beta']/city_data['kappa'],
#                color=city_colors[city], s=100, label=city, edgecolors='black', linewidth=0.5)

# # Plot fit line


# # Add city labels
# add_city_labels(plt.gca(),data.index, data['beta']/data['kappa'], data['City'])

# plt.plot(1/3*np.ones((19,1)))
# plt.plot((1/3-0.05)*np.ones((19,1)),'-')
# plt.plot((1/3+0.05)*np.ones((19,1)),'-')

# plt.xlabel(r"$City$", fontsize=12)
# plt.ylabel(r"$\beta / \kappa$", fontsize=12)
# plt.title(r"Corrected $\beta$", fontsize=14, fontweight='bold')
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()



