import os
import pickle
from pathlib import Path
from dotenv import load_dotenv
import geopandas as gpd
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

def add_soil_features_to_tiles(
    tiles_gdf: gpd.GeoDataFrame,
    soil_dir: Path,
    raster_files: dict = None,
    crs_raster: str = "EPSG:4326",
    sample_plot: bool = False
) -> gpd.GeoDataFrame:
    """
    Adds soil features (clay, pH, SOC) from SoilGrids rasters to tile centroids.
    """
    if raster_files is None:
        raster_files = {
            "clay_0_5cm":   soil_dir / "clay_0-5cm_mean_5000.tif",
            "ph_h2o_0_5cm": soil_dir / "phh2o_0-5cm_mean_5000.tif",
            "soc_0_5cm":    soil_dir / "soc_0-5cm_mean_5000.tif",
        }

    tiles_latlon = tiles_gdf.to_crs(crs_raster)
    coords = [(pt.x, pt.y) for pt in tiles_latlon.geometry_point]

    soil_data = {}
    for name, path in raster_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Raster not found: {path}")
        with rasterio.open(path) as src:
            arr = [val[0] for val in src.sample(coords)]
            arr = [v if v != src.nodata else None for v in arr]
            if "clay" in name:
                arr = [v / 10 if v is not None else None for v in arr]  # scale %
            soil_data[name] = arr

    soil_df = pd.DataFrame(soil_data, index=tiles_gdf.index)
    return pd.concat([tiles_gdf, soil_df], axis=1)


def get_soil_features():
    load_dotenv()

    # paths from env or default
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    processed_dir = Path(os.getenv("PROCESSED_DIR", data_dir / "processed"))
    soil_dir = data_dir / "raw" / "datasets" 
    # Load tiles with relief/river/mountain features
    with open(processed_dir / "all_tiles_flat.pkl", "rb") as f:
        gdf = pickle.load(f)

    if "geometry_point" not in gdf.columns:
        raise RuntimeError("Missing `geometry_point` column in GeoDataFrame")

    gdf_with_soil = add_soil_features_to_tiles(gdf, soil_dir)

    # Save to pickle
    out_path = processed_dir / "all_tiles_features_with_soil.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(gdf_with_soil, f)

    print(f"✅ Saved with soil features → {out_path}")


if __name__ == "__main__":
    get_soil_features()