import pandas as pd
from adjustText import adjust_text
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from scipy import stats
import matplotlib as mpl
data=pd.read_csv('/Users/mika/Documents/DATA/src/data_1985_new.csv')

print(data)

high_kappa=data[data["City"].isin(['Ningbo','Beijing','Las Vegas','Changzhou','Bengalore','Santiago','Paris','Kolkata','Chengdu Deyang'])]
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['axes.labelsize']=16
mpl.rcParams['legend.fontsize']=16

# Create a color map for cities
n_cities = len(data)
colors = cm.tab20(np.linspace(0, 1, n_cities))  #
city_colors = {city: colors[i] for i, city in enumerate(data['City'])}

corr_ratio=np.corrcoef(1/high_kappa['initial_ratio'],high_kappa['beta'])
res_ratio= stats.spearmanr(1/high_kappa['initial_ratio'],high_kappa['beta'])  
pearson_p_ratio = stats.pearsonr(1/high_kappa['initial_ratio'],high_kappa['beta'])[1]





def add_all_city_labels(ax, x_data, y_data, cities, high_kappa_cities,
                         fontsize_high=16, fontsize_low=14):
    texts = []

    for x, y, city in zip(x_data, y_data, cities):
        if city in high_kappa_cities:
            txt = ax.annotate(
                city, (x, y),
                fontsize=fontsize_high,
                color="black",
                zorder=5
            )
        else:
            txt = ax.annotate(
                city, (x, y),
                fontsize=fontsize_low,
                color="grey",
                alpha=0.7,
                zorder=2
            )
        texts.append(txt)

    adjust_text(
        texts,
        x=x_data, y=y_data,
        arrowprops=dict(
            arrowstyle="-",
            color="grey",
            lw=0.4,
            alpha=0.4
        ),
        force_text=(0.3, 0.5),
        expand_text=(1.2, 1.4),
        expand_points=(1.2, 1.2)
    )

# Split data
high_kappa_cities = high_kappa["City"].unique()
low_kappa = data[~data["City"].isin(high_kappa_cities)]

# Colors
high_color = "#f28482"   # pastel blue (use "#f28482" for pastel red)
low_color = "lightgrey"

plt.figure(figsize=(14, 8))

# -------------------------------
# Plot LOW kappa cities (grey)
# -------------------------------
plt.scatter(
    1 / low_kappa["initial_ratio"],
    low_kappa["beta"],
    color=low_color,
    s=60,
    alpha=0.6,
    label="Other cities",
    edgecolors="none"
)

# -------------------------------
# Plot HIGH kappa cities (pastel)
# -------------------------------
plt.scatter(
    1 / high_kappa["initial_ratio"],
    high_kappa["beta"],
    color=high_color,
    s=120,
    edgecolors="black",
    linewidth=0.7,
    label="High κ cities",
    zorder=3
)

# -------------------------------
# Linear fit (HIGH kappa only)
# -------------------------------
fit_ratio = np.polyfit(
    1 / high_kappa["initial_ratio"],
    high_kappa["beta"],
    1
)

x_fit = np.array([
    (1 / high_kappa["initial_ratio"]).min(),
    (1 / high_kappa["initial_ratio"]).max()
])

plt.plot(
    x_fit,
    fit_ratio[0] * x_fit + fit_ratio[1],
    "--",
    color="black",
    linewidth=2,
    alpha=0.3,
    
)

# -------------------------------
# City labels (HIGH kappa only)
# -------------------------------
add_all_city_labels(
    plt.gca(),
    1 / data["initial_ratio"],
    data["beta"],
    data["City"],
    high_kappa_cities
)

# -------------------------------
# Correlation statistics box
# -------------------------------
textstr = (
    f"Pearson r = {corr_ratio[0,1]:.2f} (p = {pearson_p_ratio:.2f})\n"
    f"Spearman ρ = {res_ratio.correlation:.2f} (p = {res_ratio.pvalue:.2f})"
)

props = dict(boxstyle="round", facecolor="wheat", alpha=0.85)
plt.text(
    0.72, 0.1, textstr,
    transform=plt.gca().transAxes,
    fontsize=15,
    verticalalignment="top",
    bbox=props
)

# -------------------------------
# Labels & formatting
# -------------------------------
plt.xlabel(r"$\varphi_{area}$" )
plt.ylabel(r"$\beta$")

plt.grid(True, alpha=0.3)
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig('report_ratio_beta_correlation.png')
plt.show()



