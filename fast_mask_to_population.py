import pandas as pd
import numpy as np
import math
import json
import requests
from affine import Affine
import rasterio.features
from shapely.geometry import shape
from shapely.ops import unary_union
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import time


def fast_mask_to_population(csv_path, center_lon, center_lat, year):
    mask = pd.read_csv(csv_path)
    mask = mask.to_numpy().astype("uint8")

    plt.figure()
    plt.imshow(mask)
    plt.show()

    rows, cols = mask.shape
    year = int(year)

    # Resolution
    dy = 0.03 / 111
    dx = 0.03 / (111 * math.cos(math.radians(center_lat)))

    # Top-left origin
    west = center_lon - (cols / 2) * dx
    north = center_lat + (rows / 2) * dy
    transform = Affine.translation(west, north) * Affine.scale(dx, -dy)

    # Extract shapes
    shapes_gen = rasterio.features.shapes(
        mask.astype("uint8"),
        transform=transform
    )
    polygons = [shape(geom) for geom, value in shapes_gen if value == 1]

    if not polygons:
        return 0





    # Merge and simplify polygons
    merged = unary_union(polygons)
    merged = merged.simplify(0.0005)

    print(merged.geom_type)
    print(f"Size of merged: {len(merged.exterior.coords)}")

    if merged.geom_type == "Polygon":
        x, y = merged.exterior.xy




    geojson = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {},
            "geometry": merged.__geo_interface__
        }]
    }

    pixel_area_m2=30*30
    expected_area_m2=pixel_area_m2*mask.sum()
    print("Expected Area (m^2)", expected_area_m2)
    deg_to_m_lat = 111_000
    deg_to_m_lon = 111_000 * math.cos(math.radians(center_lat))
    poly_area_m2 = merged.area * deg_to_m_lat * deg_to_m_lon
    print("Polygon area (m²):", poly_area_m2)

    # Submit WorldPop request
    r = requests.post(
        "https://api.worldpop.org/v1/services/stats",
        json={
            "dataset": "wpgppop",
            "year": year,
            "geojson": json.dumps(geojson)
        }
    )

    if r.status_code != 200:
        print("Full error:", r.text)
        r.raise_for_status()

    task_id = r.json()["taskid"]
    print(f"Task created: {task_id}")

    # Poll until result is ready
    poll_url = f"https://api.worldpop.org/v1/tasks/{task_id}"
    poll = requests.get(poll_url)
    result = poll.json()
    print(json.dumps(result, indent=2))

    for attempt in range(20):  # max ~60s wait
        time.sleep(3)
        poll = requests.get(poll_url)
        result = poll.json()
        print(f"Poll {attempt + 1}: status = {result.get('status')}")

        if result.get("status") == "finished":
            if result.get("error"):
                raise RuntimeError(f"WorldPop error: {result['error_message']}")
            return result["data"]["total_population"]

        if result.get("status") == "failed":
            raise RuntimeError(f"WorldPop task failed: {result}")

    raise TimeoutError("WorldPop task did not finish in time")