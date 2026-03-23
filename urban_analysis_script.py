from urban_analysis_lib import *
import csv
import matplotlib.pyplot as plt
from CV_analysis import *
from acceleration_metric_function import analyze_urban_growth
from fast_mask_to_population import *



# il reste à partir de sao paulo 
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

non_urban_clusters=[]
cv_mean_col=[]
cv_std_col=[]


collection_emptiness=pd.DataFrame()
collection_non_urbanized_clusters=pd.DataFrame()





high_kappa_cities=['Beijing','Paris','Las Vegas','Changzhou','Ningbo','Chengdu Deyang','Bengalore','Bangkok','Santiago','Kolkata']

cv_stats_col=[]
s_eff=[]

output_directory="outputs_evolution"
ratio_col=[]
cities=['Ningbo','Chengdu Deyang', 'Beijing','Bengalore','Kolkata','Paris','Cairo','Guatemala City','Johannesburg','London','Mexico City','Nairobi','Santiago','Sao Paulo','Tehran','Las Vegas','Atlanta']
years=range(2000,2016)

for name in cities:
    lcc_area_arr=[]
    pop_arr=[]
    # analyze_urban_growth(name,80)

    # radius_factor=5.0


    # # Download and load data
    downloader = WSFTileManager(cache_dir="./wsf_cache")
    lat, lon = geocode_city(name)
    results = downloader.download_region(lat, lon, radius_km=80)
    analyzer = BuiltAreaAnalyzer()
    data, metadata = analyzer.load_tiles_from_download_result(results)
    center_row, center_col=analyzer.latlon_to_pixel(lat,lon,metadata['transform'])


    for year in years:
        output_name= f'new_mask_LCC_{name}_{year}.csv'
        mask,_,lcc_area_new,lcc_radius, centroid_lon, centroid_lat= extract_lcc_and_n_clusters_mask(wsf_data=data,year=year,center_row=center_row,transform=metadata['transform'],center_col=center_col,
                                  n_clusters=1,region_size=6000,analyzer=analyzer, output_csv=output_name)
    

        pop = fast_mask_to_population(
                csv_path=f'new_mask_LCC_{name}_{year}.csv',
                center_lon=centroid_lon,
                center_lat=centroid_lat,
                year=year)
        # region_mask,_ = extract_lcc_region(data,year,1000)
        # mask,_=analyzer.find_city_specific_lcc(region_mask,center_row,center_col,200)
        # df = pd.DataFrame(mask)
        # df.to_csv(output_name, index=False, header=False)
        # wsf,dict=analyzer.extract_built_area_bbox(data=data,transform=metadata['transform'],center_lat=lat,center_lon=lon,size_km=20)
        
        print(f'the population is :{pop}')
        
       
    # 
    # df=extract_perimeter_from_bbox_optimized(data,transform=metadata['transform'],n_sectors=500,use_numba=True)










    # # print(center_col)
    # # Track evolution (FAST - 8-10× speedup!)
    # tracker = ClusterEvolutionTracker(
    #     analyzer, 
    #     n_clusters=10,
    #     radius_factor=radius_factor # NEW! Focus on specific region
    # )
    # tracker.export_evolution_csv(data, f"output_{name}.csv")

    # Visualize (NEW!)
        # stats, lcc_area = visualize_clusters_optimized(
        #     wsf_data=data, analyzer=analyzer, year=year,
        #     radius_factor=1, n_clusters=1,center_row=center_row,center_col=center_col,dpi=900,show_circle=False,crop_factor=3,
        #      )
    # ratio=stats['area_ratio_top10_to_lcc']
    # dict=analyze_local_density_cv(wsf_data=data,analysis_year=1985,delimiter_year=2015,window_size=5,center_col=center_col,center_row=center_row,analyzer=analyzer,output_csv=f'CV_window_5_{name}.csv')
    # cv_mean_col.append(dict["cv_stats"]["mean"])
    # cv_std_col.append(dict["cv_stats"]["std"])

    # dict=calculate_lcc_density_metrics(wsf_data=data,year=2015,analyzer=analyzer,center_row=center_row,center_col=center_col,min_cluster_size_km2=2)
    # s_eff.append(dict['mean_non_urban_cluster_size_km2'])

    # metrics_collection.append(dict['bbox_non_urbanized_ratio'])
    # metrics_collection_2.append(dict['convex_hull_non_urbanized_ratio'])
    # metrics_collection_3.append(dict['filled_non_urbanized_ratio'])
        # pop_arr.append(pop)
    #     print('avec visulaisation le lcc area est : ')
    #     print(lcc_area)
    #     lcc_area_arr.append(lcc_area_new)

    # # pd.DataFrame(pop_arr).to_csv(f'pop_{name}.csv')
    # pd.DataFrame(lcc_area_arr).to_csv(f'area_{name}.csv ')

# df['s_eff']=s_eff
# df.to_csv('constraint_s_eff')
# # df.to_csv(f'perimeter_fractal_{name}_500.csv')


# print('lcc area with the new function')
# print(lcc_area_new)
# print('lcc area with the old function')
# print(lcc_area)