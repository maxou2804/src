from urban_analysis_lib import *
import csv
from acceleration_metric_function import *
import matplotlib.pyplot as plt
from adjustText import adjust_text
from scipy import stats
from  matplotlib.pyplot import cm


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




data=pd.DataFrame({'City': ['Ningbo','Chengdu Deyang', 'Beijing','Changzhou','Bengalore','Kolkata','Paris','Bangkok','Cairo','Guatemala City','Johannesburg','London','Mexico City','Nairobi','Santiago','Sao Paulo','Tehran','Las Vegas','Atlanta'],
      'alpha':[0.56, 0.53, 0.54, 0.54, 0.55, 0.52, 0.52 ,0.53, 0.53, 0.52, 0.58, 0.54, 0.55, 0.56, 0.58, 0.51, 0.55, 0.55, 0.56],
      'beta': [0.44, 0.68, 0.41, 0.37, 0.83, 0.34, 0.56, 1.01, 0.37, 0.37, 0.07, 0.01, 0.04, 0.28, 0.62, 0.89, 0.10, 0.41, 0.04],
       '1/z': [0.58, 0.68, 0.58, 0.56, 0.72, 0.74, 0.76, 0.80, 0.54, 0.33, 0.27, 0.21, 0.25, 0.27, 0.74, 0.41, 0.54, 0.37, 0.52]})


metrics_collection=[]
metrics_collection_2=[]
metrics_collection_3=[]
non_urban_coll=[]

df=pd.DataFrame(data)

cities=df['City'].tolist()




# output_directory="outputs_evolution"
# for name in cities:
#     analyze_urban_growth(city_name=name,radius_km=50)

# import os, json
# import pandas as pd

# path_to_json = '/Users/mika/Documents/DATA/src/output'
# json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
# jsons_data = pd.DataFrame(columns=['city','secondary_urbanization','number_clusters','s_bar'])

# for index, js in enumerate(json_files):
#     with open(os.path.join(path_to_json, js)) as json_file:
#         json_text = json.load(json_file)

#         city= json_text["city"]
#         secondary_urbanized_area= json_text['secondary_to_2015_lcc_ratio']
#         area_lcc_2015=json_text['lcc_2015_area_km2']
#         area_lcc_1985=json_text['lcc_1985_in_2015_lcc_km2']
#         number_clusters=json_text['num_clusters_1985']/secondary_urbanized_area
#         s_bar=json_text['mean_secondary_cluster_km2']/json_text['lcc_2015_area_km2']**(0.5)
#         # here I push a list of data into a pandas DataFrame at row given by 'index'
#         jsons_data.loc[index] =[city, secondary_urbanized_area,number_clusters,s_bar]



# jsons_data=jsons_data.sort_values('city')
# print(jsons_data)

# df=df.sort_values('City')
# print(df)

# df['secondary_urbanization']=jsons_data['secondary_urbanization'].to_numpy()
# print(df)




# cities=df['City'].tolist()


# fig,ax=plt.subplots(figsize=(16,9))
# plt.plot(jsons_data['city'],jsons_data['secondary_urbanization'],'o')
# ax.tick_params("x",rotation=80)
# plt.axhline(np.mean(jsons_data['secondary_urbanization']))
# plt.ylabel('secondary urbanization')
# plt.show()



# fig,ax=plt.subplots()
# plt.plot(jsons_data['city'],jsons_data['s_bar'],'o')
# ax.tick_params("x",rotation=90)
# plt.axhline(np.mean(jsons_data['s_bar']))
# plt.ylabel('$s_{\\text{eff}}$')
# plt.show()



# fig=plt.figure()
# plt.scatter(jsons_data['s_bar'],jsons_data['secondary_urbanization'])
# plt.title('Acceleration sites')
# plt.xlabel('$s_{\\text{eff}}$')
# plt.ylabel('secondary urbanization')
# plt.show()



# # plt.figure()
# # plt.scatter(jsons_data['secondary_urbanization_km2'],df['beta'])
# # texts = [plt.text(jsons_data['secondary_urbanization_km2'].iloc[i],df['beta'].iloc[i], cities[i]) for i in range(len(cities))] 
# # adjust_text(texts)
# # plt.xlabel('secondary urbanized area in 1985')
# # plt.ylabel('beta')
# # plt.show()

# # plt.figure()
# # plt.scatter(jsons_data['number_clusters'],df['beta'])
# # texts = [plt.text(jsons_data['number_clusters'].iloc[i],df['beta'].iloc[i], cities[i]) for i in range(len(cities))] 
# # adjust_text(texts)
# # plt.xlabel('normalized number of clusters')
# # plt.ylabel('beta')
# # plt.show()




# # corr_beta_high=stats.spearmanr(jsons_data['number_clusters'],df['beta'])
# # print(corr_beta_high)

# # # Create a color map for cities
# # n_cities = len(data)
# # colors = cm.tab20(np.linspace(0, 1, n_cities))  # Using tab20 colormap for distinct colors
# # city_colors = {city: colors[i] for i, city in enumerate(data['City'])}

# # # Alternative: colors = cm.rainbow(np.linspace(0, 1, n_cities))

# # # Create a consistent color dictionary for each city

# # add_city_labels = add_city_labels_with_adjusttext





# # high_kappa=df[df["City"].isin(['Ningbo','Beijing Lafang','Las Vegas','Changzhou','Bengalore','Bangkok','Santiago','Paris','Kolkata','Tehran'])]

# # print(high_kappa)


# # corr_ratio=np.corrcoef(high_kappa['secondary_urbanization_km2'],high_kappa['beta'])
# # res_ratio= stats.spearmanr(high_kappa['secondary_urbanization_km2'],high_kappa['beta'])  
# # pearson_p_ratio = stats.pearsonr(high_kappa['secondary_urbanization_km2'],high_kappa['beta'])[1]



# # plt.figure(figsize=(14, 8))
# # fit_ratio=np.polyfit(high_kappa['secondary_urbanization_km2'],high_kappa['beta'],1)
# # # Plot points with different colors
# # for city in high_kappa['City']:
# #     city_data = high_kappa[high_kappa['City'] == city]
# #     plt.scatter(city_data['secondary_urbanization_km2'], city_data['beta'], 
# #                color=city_colors[city], s=100, label=city, edgecolors='black', linewidth=0.5)

# # # Plot fit line
# # x_fit = np.array([high_kappa['secondary_urbanization_km2'].min(), high_kappa['secondary_urbanization_km2'].max()])
# # plt.plot(x_fit, fit_ratio[0]*x_fit+fit_ratio[1], '--', color='black',
# #          label=f'Linear fit: y={fit_ratio[0]:.2f}x + {fit_ratio[1]:.2f}', linewidth=2, alpha=0.7)

# # # Add city labels
# # add_city_labels(plt.gca(), high_kappa['secondary_urbanization_km2'], high_kappa['beta'], high_kappa['City'])

# # # Add correlation statistics as text box
# # textstr = f'Pearson r = {corr_ratio[0,1]:.2f} (p = {pearson_p_ratio:.2f})\n' + \
# #           f'Spearman ρ = {res_ratio.correlation:.2f} (p = {res_ratio.pvalue:.2f})'
# # props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
# # plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=11,
# #         verticalalignment='top', bbox=props)

# # plt.xlabel(r"secondary urb", fontsize=12)
# # plt.ylabel(r"$\beta$", fontsize=12)
# # plt.title(r"Correlation: secondary urb vs $\beta$", fontsize=14, fontweight='bold')
# # plt.grid(True, alpha=0.3)
# # plt.tight_layout()
# # plt.savefig('high_kappa_secondary_urb_vs_beta_correlation_labeled.png', dpi=300, bbox_inches='tight')
# # plt.show()






