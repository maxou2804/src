from urban_analysis_lib import *
data=pd.DataFrame({'City': ['Ningbo','Chengdu Deyang', 'Beijing','Changzhou','Bengalore','Kolkata','Paris','Bangkok','Cairo','Guatemala City','Johannesburg','London','Mexico City','Nairobi','Santiago','Sao Paulo','Tehran','Las Vegas','Atlanta'],
      'alpha':[0.56, 0.53, 0.54, 0.54, 0.55, 0.52, 0.52 ,0.53, 0.53, 0.52, 0.58, 0.54, 0.55, 0.56, 0.58, 0.51, 0.55, 0.55, 0.56],
      'beta': [0.44, 0.68, 0.41, 0.37, 0.83, 0.34, 0.56, 1.01, 0.37, 0.37, 0.07, 0.01, 0.04, 0.28, 0.62, 0.89, 0.10, 0.41, 0.04],
       '1/z': [0.58, 0.68, 0.58, 0.56, 0.72, 0.74, 0.76, 0.80, 0.54, 0.33, 0.27, 0.21, 0.25, 0.27, 0.74, 0.41, 0.54, 0.37, 0.52]})




df=pd.DataFrame(data)
cities=df['City'].tolist()

cities=['Bangkok']
output_directory="outputs_evolution"
for name in cities:

    year=1985
    radius_factor=5.0


    # Download and load data
    downloader = WSFTileManager(cache_dir="./wsf_cache")
    lat, lon = geocode_city(name)
    results = downloader.download_region(lat, lon, radius_km=100)
    analyzer = BuiltAreaAnalyzer()
    data, metadata = analyzer.load_tiles_from_download_result(results)

    # # Track evolution (FAST - 8-10× speedup!)
    # tracker = ClusterEvolutionTracker(
    #     analyzer, 
    #     n_clusters=10,
    #     radius_factor=radius_factor # NEW! Focus on specific region
    # )
    # tracker.export_evolution_csv(data, f"output_{name}.csv")

    # Visualize (NEW!)
    stats = visualize_clusters_optimized(
        wsf_data=data, analyzer=analyzer, year=year,
        radius_factor=radius_factor, n_clusters=30,dpi=900,show_circle=False,crop_factor=2,
        output_path=f"clusters_{name}_{year}.png"
    )


