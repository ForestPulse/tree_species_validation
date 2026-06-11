pip install exactextract

# Validation Pipeline Step 2:
# Sampling Raster-datasets for Prediction-values

from exactextract import exact_extract
from exactextract.raster import RasterioRasterSource
import geopandas as gpd
import pandas as pd
import rasterio
import numpy as np

# Input
raster_class_path = "" # Tree species fractions raster
raster_heightmask_path = ""               # Binary Height Raster : 0 = vegetation < 2m  /  1 = vegetation > 2m
vector_path = ""                          # Polygons with inventory values from step 2
output_path_frac = ""                     # specify output

# Look-up for raster
raster_bands_frac = {

    1:  "fichte",
    2:  "kiefer",
    3:  "tanne",
    4:  "douglasie",
    5:  "larche",
    6:  "buche",
    7:  "eiche",
    8:  "ahorn",
    9:  "birke",
    10: "erle",
    11: "pappel", # later mixed into class "andere"
    12: "andere",
    13: "boden",
}

# Sampling tree species fractions
gdf = gpd.read_file(vector_path)

with rasterio.open(raster_class_path) as src:
    if gdf.crs != src.crs:
        gdf = gdf.to_crs(src.crs)

    for band_idx in range(1, src.count + 1):
        species = raster_bands_frac[band_idx]
        col_name = f"pred_{species}"
        print(f"Band {band_idx} → {col_name} ...")

        rast_class = RasterioRasterSource(src, band_idx=band_idx)

        stats_class = exact_extract(
            rast_class,
            gdf,
            "mean(min_coverage_frac=1.0)",
            output="pandas"
        )

        gdf[col_name] = stats_class

# Mix class "pappel" into class "andere"
gdf["pred_andere"] = gdf["pred_andere"].fillna(0) + gdf["pred_pappel"].fillna(0)
gdf.drop(columns=["pred_pappel"], inplace=True)

# Sampling binary height raster
with rasterio.open(raster_heightmask_path) as src:
    if gdf.crs != src.crs:
        gdf = gdf.to_crs(src.crs)

    rast_heightmask = RasterioRasterSource(src)

    stats_height_mask = exact_extract(
        rast_heightmask,
        gdf,
        "mean(min_coverage_frac=1.0)",
        output="pandas"
    )

    gdf["coverage"] = stats_height_mask

gdf.to_file(output_path_frac, driver="GPKG")
