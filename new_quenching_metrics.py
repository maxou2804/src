import pandas as pd
from adjustText import adjust_text
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from scipy import stats
import matplotlib as mpl

mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['axes.labelsize']=16
mpl.rcParams['legend.fontsize']=16

data=pd.read_csv('/Users/mika/Documents/DATA/src/data_quenching_new.csv')
data['anisotropy']=[0.3, 0.45, 0.45, 0.25, 0.7, 1.45, 1.7, 0.85, 0.5, 0.7, 0.75,0.82, 0.80, 0.95, 1.05, 0.8, 1.1, 0.5, 1.2 ]

high_kappa=data[data["City"].isin(['Ningbo','Beijing','Las Vegas','Changzhou','Bengalore','Santiago','Paris','Kolkata','Chengdu Deyang'])]


# Create a color map for cities
n_cities = len(data)
colors = cm.tab20(np.linspace(0, 1, n_cities))  #
city_colors = {city: colors[i] for i, city in enumerate(data['City'])}





######################### Filled Hole Metrics ##############################



corr_ratio=np.corrcoef(high_kappa['hole_metric_filled'],high_kappa['beta'])
res_ratio= stats.spearmanr(high_kappa['hole_metric_filled'],high_kappa['beta'])  
pearson_p_ratio = stats.pearsonr(high_kappa['hole_metric_filled'],high_kappa['beta'])[1]





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
    low_kappa["hole_metric_filled"],
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
    high_kappa["hole_metric_filled"],
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
    high_kappa["hole_metric_filled"],
    high_kappa["beta"],
    1
)

x_fit = np.array([
    (high_kappa["hole_metric_filled"]).min(),
    (high_kappa["hole_metric_filled"]).max()
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
    data["hole_metric_filled"],
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
plt.xlabel(r"CD" )
plt.ylabel(r"$\beta$")

plt.grid(True, alpha=0.3)
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig('report_filled_holes_beta_correlation.png')
plt.show()



######################### Anisotropy + Constraint metric´ ##############################



corr_ratio=np.corrcoef(high_kappa['hole_metric_filled']*high_kappa['anisotropy'],high_kappa['beta'])
res_ratio= stats.spearmanr(high_kappa['hole_metric_filled']*high_kappa['anisotropy'],high_kappa['beta'])  
pearson_p_ratio = stats.pearsonr(high_kappa['hole_metric_filled']*high_kappa['anisotropy'],high_kappa['beta'])[1]





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
    low_kappa['hole_metric_filled']*low_kappa['anisotropy'],
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
    high_kappa['hole_metric_filled']*high_kappa['anisotropy'],
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
     high_kappa['hole_metric_filled']*high_kappa['anisotropy'],
    high_kappa["beta"],
    1
)

x_fit = np.array([
    (high_kappa['hole_metric_filled']*high_kappa['anisotropy']).min(),
    (high_kappa['hole_metric_filled']*high_kappa['anisotropy']).max()
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
    data['hole_metric_filled']*data['anisotropy'],
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
plt.xlabel(r"CD" )
plt.ylabel(r"$\beta$")

plt.grid(True, alpha=0.3)
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig('report_mixed_quenching_metric.png')
plt.show()

