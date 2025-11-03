from urban_analysis_lib import *
import pandas as pd
import csv
year=1985

mask_collection=[]

    # Download and load data
downloader = WSFTileManager(cache_dir="./wsf_cache")
lat, lon = geocode_city('Grenoble')
results = downloader.download_region(lat, lon, radius_km=20)
analyzer = BuiltAreaAnalyzer()
data, metadata = analyzer.load_tiles_from_download_result(results)

year_list=range(1985,2015)

for year in year_list:
    print(year)
    mask=extract_year_mask_numba(wsf_data=data, year=year)
    df = pd.DataFrame(mask)

    df.to_csv(f'grenoble_mask_{year}.csv', index=True, header=True)



 


