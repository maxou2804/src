import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
import matplotlib as mpl


mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['axes.labelsize']=16
mpl.rcParams['legend.fontsize']=16


def add_city_labels_with_adjusttext(ax, x_data, y_data, cities, fontsize=13):
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

############### Constraints ##################################################################################


data_s_eff=pd.read_csv('/Users/mika/Documents/DATA/src/constraint_s_eff.csv')
data_p_eff=pd.read_csv('/Users/mika/Documents/DATA/src/data_quenching_new.csv')



############### Acceleration ######################################################################################
output_directory="outputs_evolution"


import os, json
import pandas as pd

path_to_json = '/Users/mika/Documents/DATA/src/output'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
jsons_data = pd.DataFrame(columns=['city','secondary_urbanization','number_clusters','s_bar','s_bar_top_10'])

for index, js in enumerate(json_files):
    with open(os.path.join(path_to_json, js)) as json_file:
        json_text = json.load(json_file)

        city= json_text["city"]
        secondary_urbanized_area= json_text['secondary_to_2015_lcc_ratio']
        area_lcc_2015=json_text['lcc_2015_area_km2']
        area_lcc_1985=json_text['lcc_1985_in_2015_lcc_km2']
        number_clusters=json_text['num_clusters_1985']/secondary_urbanized_area
        s_bar=json_text['mean_secondary_cluster_km2']/json_text['lcc_2015_area_km2']**(0.5)
        s_bar_top_10=json_text['mean_top10_secondary_cluster_km2']/json_text['lcc_2015_area_km2']**(0.5)
        # here I push a list of data into a pandas DataFrame at row given by 'index'
        jsons_data.loc[index] =[city, secondary_urbanized_area,number_clusters,s_bar,s_bar_top_10]



jsons_data=jsons_data.sort_values('city')
data_s_eff=data_s_eff.sort_values('City')
data_p_eff=data_p_eff.sort_values('City')


print(jsons_data)

###################################### Constraints ########################################################################
# fig, ax= plt.subplots()
# plt.plot(data_s_eff['City'],data_s_eff['s_eff'],'o')
# ax.tick_params("x",rotation=80)
# plt.axhline(np.mean(data_s_eff['s_eff']))
# plt.ylabel('$s_{\\text{s_eff,c}}$')
# plt.show()

# fig, ax= plt.subplots()
# plt.plot(data_p_eff['City'],data_p_eff['hole_metric_filled'],'o')
# ax.tick_params("x",rotation=80)
# plt.axhline(np.mean(data_p_eff['hole_metric_filled']))
# plt.ylabel('$p_{\\text{eff}}$')
# plt.show()

fig=plt.figure()
plt.scatter(data_s_eff['s_eff'],data_p_eff['hole_metric_filled'])
plt.xlabel('$s_{\\text{eff}}$')
plt.ylabel('secondary urbanization')
plt.show()

# #################################### Acceleration ########################################################################


# fig,ax=plt.subplots()
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



fig=plt.figure()
plt.scatter(jsons_data['s_bar'],jsons_data['secondary_urbanization'])
plt.title('Acceleration sites')
plt.xlabel('$s_{\\text{eff}}$')
plt.ylabel('secondary urbanization')
plt.show()


#################################### Coupled ########################################################################

plt.figure(figsize=(14, 8))
plt.plot(jsons_data['secondary_urbanization'],data_p_eff['hole_metric_filled'],'o',markersize=12)
add_city_labels(plt.gca(), jsons_data['secondary_urbanization'], data_p_eff['hole_metric_filled'], data_p_eff['City'])
plt.xlabel('acceleration $p_{\\text{eff}}$')
plt.ylabel('constraints $p_{\\text{eff}}$')
plt.savefig('p_eff_corr.png')
plt.show()
plt.close()


plt.figure(figsize=(14, 8))
plt.plot(jsons_data['s_bar'],data_s_eff['s_eff'],'o',markersize=12)
add_city_labels(plt.gca(), jsons_data['s_bar'], data_s_eff['s_eff'], data_s_eff['City'])
plt.xlabel('acceleration $s_{\\text{eff}}$')
plt.ylabel('constraints $s_{\\text{eff}}$')
plt.savefig('s_eff_corr.png')
plt.show()
plt.close()

plt.figure(figsize=(14, 8))
plt.plot(jsons_data['s_bar'],jsons_data['s_bar_top_10'],'o')
add_city_labels(plt.gca(), jsons_data['s_bar'], jsons_data['s_bar_top_10'], jsons_data['city'])
plt.xlabel('acceleration $s_{\\text{eff}}$')
plt.ylabel('constraints $s_{\\text{eff}}$')
plt.show()
plt.close()
