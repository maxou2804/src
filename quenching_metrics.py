import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import csv
import ast
import numpy as np
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

low_beta=df_gruyere[df_gruyere["City"].isin(['Kolkata','Nairobi','Atlanta','London','Johannesburg','Mexico City','Tehran'])]

high_density=df_gruyere[df_gruyere["City"].isin(['Las Vegas','Paris','Chengdu Deyang','Bangkok'])]

plt.figure()
plt.scatter(low_beta['area_population'],low_beta['beta'])
plt.xlabel('area/population')
plt.ylabel('beta')
plt.show()

plt.figure()
plt.plot(df_gruyere['metric_perimeter'],df_gruyere['beta'],'o')
plt.ylabel(r'\beta')
plt.xlabel(r'\frac{A_{non-urb}}/{A_urb}')
texts = [plt.text(df_gruyere['metric_perimeter'].iloc[i],df_gruyere['beta'].iloc[i], cities[i]) for i in range(len(cities))] 
adjust_text(texts)
plt.savefig('metric_perimeter_beta')
plt.show()


plt.figure()
plt.plot(area_population,df_gruyere['beta'],'o')
plt.xlabel('area/population')
plt.ylabel('beta')
texts = [plt.text(area_population[i],df_gruyere['beta'].iloc[i], cities[i]) for i in range(len(cities))] 
adjust_text(texts)
plt.savefig('densities_beta')
plt.show()


plt.figure()
plt.scatter(high_density['metric_hull'],high_density['beta'])
texts = [plt.text(high_density['metric_hull'].iloc[i],high_density['beta'].iloc[i], high_density['City'].iloc[i]) for i in range(len( high_density['City']))] 
adjust_text(texts)
plt.xlabel('constraint density')
plt.ylabel('beta')
plt.show()

corr_beta_high=stats.spearmanr(high_density['beta'],high_density['metric_perimeter'])
print(corr_beta_high)

corr_beta_metric_perimeter=stats.spearmanr(df_gruyere['beta'],df_gruyere['metric_perimeter'])
print(corr_beta_metric_perimeter)

corr_beta_densities=stats.spearmanr(df_gruyere['beta'],area_population)
print(corr_beta_densities)



df_clusters=pd.read_csv('/Users/mika/Documents/DATA/src/quenching_metrics.py',on_bad_lines='skip')



def compute_stats(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        rows = list(reader)
    
    mean_area_col = []
    std_area_col=[]
    mean_dist_col = []
    std_dist_col=[]
    for row in rows[1:]:  # Skip header if present, but in this case, first row is indices
        areas = []
        distances=[]
        for cell in row:
            if cell.strip():  # Skip empty cells
                try:
                    data_dict = ast.literal_eval(cell.strip())
                    if 'area_km2' in data_dict:
                        areas.append(float(data_dict['area_km2']))
                    if 'distance_km' in data_dict:
                        distances.append(float(data_dict['distance_km']))
                except (ValueError, SyntaxError):
                    pass  # Skip invalid entries
       
        if areas:
            mean = np.mean(areas)
            std = np.std(areas)
            mean_area_col.append(mean)
            std_area_col.append(std)

        if distances:
      
            mean_dist=np.mean(distances)
            std_dist=np.std(distances)
            mean_dist_col.append(mean_dist)
            std_dist_col.append(std_dist)

    return mean_area_col,std_area_col,mean_dist_col,std_dist_col

# Usage
filename = 'non_urban_clusters_metric_1985.csv'
mean_area,std_area,mean_dist,std_dist = compute_stats(filename)



corr_mean =stats.spearmanr(df_gruyere['beta'], mean_area)
corr_std= stats.spearmanr(df_gruyere['beta'],  std_area)
print(corr_mean)
print(corr_std)




corr_mean =stats.spearmanr(df_gruyere['beta'], mean_dist)
corr_std= stats.spearmanr(df_gruyere['beta'],  std_dist)
print(corr_mean)
print(corr_std)

plt.figure()
plt.plot(mean_dist,mean_area,'o')
plt.show()