# from wsf_evolution_lcc import WSFTileManager, BuiltAreaAnalyzer, geocode_city, export_lcc_coordinates_all_years
import pandas as pd
from perimeter_function import extract_perimeter_from_bbox_optimized
from urban_analysis_lib import geocode_city, WSFTileManager, BuiltAreaAnalyzer
# create the directory where to store tiles
downloader = WSFTileManager(cache_dir="./wsf_cache")


from urban_analysis_lib import geocode_city
# find the lat and lon of the city trough its name



data={'City': ['Ningbo','Chengdu Deyang', 'Beijing','Changzhou','Bengalore','Kolkata','Paris','Bangkok','Cairo','Guatemala City','Johannesburg','London','Mexico City','Nairobi','Santiago','Sao Paulo','Tehran','Las Vegas','Atlanta'],
      'alpha':[0.56, 0.53, 0.54, 0.54, 0.55, 0.52, 0.52 ,0.53, 0.53, 0.52, 0.58, 0.54, 0.55, 0.56, 0.58, 0.51, 0.55, 0.55, 0.56],
      'beta': [0.44, 0.68, 0.41, 0.37, 0.83, 0.34, 0.56, 1.01, 0.37, 0.37, 0.07, 0.01, 0.04, 0.28, 0.62, 0.89, 0.10, 0.41, 0.04],
       '1/z': [0.58, 0.68, 0.58, 0.56, 0.72, 0.74, 0.76, 0.80, 0.54, 0.33, 0.27, 0.21, 0.25, 0.27, 0.74, 0.41, 0.54, 0.37, 0.52]}


df=pd.DataFrame(data)
cities=df['City'].tolist()

for name in cities:


    
    output_path=f"C:\\Users\\trique\\Downloads\\MASTER_THESIS\\data_vizualization\\src_EDEN\\perimeter\\{name}_perimeter.csv"

    lat,lon=geocode_city(name)
    # calculates the required tiles based on the position and the radius
    tiles = downloader.calculate_required_tiles(lat, lon, radius_km=100)

    # downloads the corresponding tiles
    results= downloader.download_region(lat,lon, radius_km=50)

    # vizualisze the tiles if necessary
    #downloader.visualize_coverage(results=results)



    # create the built analzyer
    analyzer = BuiltAreaAnalyzer()

    data, metadata = analyzer.load_tiles_from_download_result(results)

    data_subset, meta_subset = analyzer.extract_built_area_bbox(
        data=data,
        transform=metadata['transform'],
        center_lat=31.8122,
        center_lon=119.9692,
        size_km=50



    )

    data=extract_perimeter_from_bbox_optimized(data_subset,meta_subset['transform'],10000 ,use_numba=True)  
    data.to_csv(output_path, index=False)