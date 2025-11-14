from urban_analysis_lib import *
import pandas as pd
import csv

data={'City': ['Ningbo','Chengdu Deyang', 'Beijing Lafang','Changzhou','Bengalore','Kolkata','Paris','Bangkok','Cairo','Guatemala City','Johannesburg','London','Mexico City','Nairobi','Santiago','Sao Paulo','Tehran','Las Vegas','Atlanta'],
      'alpha':[0.56, 0.53, 0.54, 0.54, 0.55, 0.52, 0.52 ,0.53, 0.53, 0.52, 0.58, 0.54, 0.55, 0.56, 0.58, 0.51, 0.55, 0.55, 0.56],
      'beta': [0.44, 0.68, 0.41, 0.37, 0.83, 0.34, 0.56, 1.01, 0.37, 0.37, 0.07, 0.01, 0.04, 0.28, 0.62, 0.89, 0.10, 0.41, 0.04],
       '1/z': [0.58, 0.68, 0.58, 0.56, 0.72, 0.74, 0.76, 0.80, 0.54, 0.33, 0.27, 0.21, 0.25, 0.27, 0.74, 0.41, 0.54, 0.37, 0.52]}

df=pd.DataFrame(data)
cities=df['City'].tolist()



year=1985
mask_collection=[]
cities_list=['Bangkok']
n_clusters=300
for city in cities_list:
    # Download and load data
    downloader = WSFTileManager(cache_dir="./wsf_cache")
    lat, lon = geocode_city(f'{city}')
    results = downloader.download_region(lat, lon, radius_km=20)
    analyzer = BuiltAreaAnalyzer()
    data, metadata = analyzer.load_tiles_from_download_result(results)
    mask=extract_lcc_and_n_clusters_mask(wsf_data=data, search_radius_factor=8,year=year,center_lcc=True,n_clusters=n_clusters,output_csv=f'/Users/mika/Documents/DATA/masks/mask_{city}_n_clusters={n_clusters}_{year}.csv',coarse_grain_factor=5)
    # mask_regional=extract_lcc_region_mask(wsf_data=data,year=year,region_size=2000,coarse_grain_factor=4,label_all_clusters=True,output_csv=f'/Users/mika/Documents/DATA/masks/mask_regional_{city}_n_clusters={n_clusters}_{year}.csv')



 


