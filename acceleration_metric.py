from urban_analysis_lib import *
import csv
from acceleration_metric_function import *
import matplotlib.pyplot as plt
from adjustText import adjust_text



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




output_directory="outputs_evolution"
for name in cities:
    analyze_urban_growth(city_name=name,radius_km=50)

import os, json
import pandas as pd

path_to_json = '/Users/mika/Documents/DATA/src/output'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
jsons_data = pd.DataFrame(columns=['city','secondary_urbanization_km2'])

for index, js in enumerate(json_files):
    with open(os.path.join(path_to_json, js)) as json_file:
        json_text = json.load(json_file)

        # here you need to know the layout of your json and each json has to have
        # the same structure (obviously not the structure I have here)
        city= json_text["city"]
        secondary_urbanized_area= json_text['secondary_urbanization_km2']
        area_lcc_2015=json_text['lcc_2015_area_km2']
        area_lcc_1985=json_text['lcc_1985_in_2015_lcc_km2']
        num

        secondary_urbanized_area=secondary_urbanized_area/area_lcc_2015
      
        # here I push a list of data into a pandas DataFrame at row given by 'index'
        jsons_data.loc[index] =[city, secondary_urbanized_area]



jsons_data.sort_values('city')
print(jsons_data)

df.sort_values('City')
print(df)

cities=df['City'].tolist()

plt.figure()
plt.scatter(jsons_data['secondary_urbanization_km2'],df['beta'])
texts = [plt.text(jsons_data['secondary_urbanization_km2'].iloc[i],df['beta'].iloc[i], cities[i]) for i in range(len(cities))] 
adjust_text(texts)
plt.xlabel('urbanized area in 1985')
plt.ylabel('beta')
plt.show()
