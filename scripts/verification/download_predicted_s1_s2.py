# scripts/ee/download_tiles.py

import os
import csv
import logging
import shutil
import urllib.request

import ee
import numpy as np
import rasterio
from PIL import Image
from dotenv import load_dotenv
from ee import ServiceAccountCredentials
from pathlib import Path
import yaml


def retrieve_tiles(config_path: str = "config.yaml"):
    """
    Download LiDAR, Sentinel-1 and Sentinel-2 images for candidate tiles,
    compute hillshade composites, and save everything under RESULTS_DIR/predicted/<tile>/.
    Each run clears out the old `predicted` folder first.
    """
    # ─────────────────────── Load environment & configuration ───────────────────────
    load_dotenv()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    RAW_DATA_DIR = Path(cfg["raw_data_dir"])
    RESULTS_DIR  = Path(cfg["results_dir"])
    OUT_DIR      = RESULTS_DIR / "predicted"

    # LiDAR DTM tiles
    DTM_DIR = (
        RAW_DATA_DIR
        / "datasets"
        / "nasa-amazon-lidar-2008-2018"
        / "Nasa_lidar_2008_to_2018_DTMs"
        / "DTM_tiles"
    )

    # CSV listing candidate tiles
    PATH_TO_COORDS_CSV = RESULTS_DIR / "candidates_top500.csv"
    if not PATH_TO_COORDS_CSV.exists():
        raise FileNotFoundError(f"CSV not found: {PATH_TO_COORDS_CSV}")

    # ─────────────────────── Earth Engine auth ───────────────────────
    GSA_EMAIL = os.getenv("GSA_EMAIL")
    KEY_PATH  = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    creds     = ServiceAccountCredentials(GSA_EMAIL, KEY_PATH)
    ee.Initialize(credentials=creds, project=os.getenv("GEE_PROJECT", "kaggle-ai-to-z"))

    # ─────────────────────── Parameters & viz settings ───────────────────────
    MAX_PHOTOS_PER_PREFIX = int(os.getenv("MAX_PHOTOS_PER_PREFIX", 3))
    BUFFER_METERS         = int(os.getenv("BUFFER_RADIUS_METERS", 1500))
    DATE                  = os.getenv("GE_DATE", "2025-05-01")
    AZ1, ALT1             = 315, 45
    AZ2, ALT2             = 45, 30

    S1_COLL, S1_VIS = "COPERNICUS/S1_GRD", {'bands': ['VV'], 'min': -25, 'max': 5}
    S2_COLL, S2_VIS = "COPERNICUS/S2_SR_HARMONIZED", {'bands': ['B4','B3','B2'], 'min': 0, 'max': 3000, 'gamma':1.3}

    # ─────────────────────── Logging & cleanup ───────────────────────
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ─────────────────────── Helpers ───────────────────────
    def get_best(collection, pt, start, end, vis_params, cloud_filter=False):
        geom = ee.Geometry.Point(pt).buffer(BUFFER_METERS).bounds()
        col  = ee.ImageCollection(collection).filterBounds(ee.Geometry.Point(pt)).filterDate(start, end)
        if cloud_filter:
            col = col.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30)).sort("CLOUDY_PIXEL_PERCENTAGE")
        else:
            col = col.sort("system:time_start")
        img = col.first()
        return img, geom

    def save_visual(img, region, path, vis_params):
        vis = img.visualize(**vis_params)
        url = vis.getThumbURL({"region": region, "dimensions": 800, "format": "jpg"})
        path.write_bytes(urllib.request.urlopen(url).read())

    def save_geotiff(img, region, path, bands=None):
        params = {"scale":10, "region":region, "format":"GEO_TIFF", "crs":"EPSG:4326"}
        if bands: params["bands"] = bands
        path.write_bytes(urllib.request.urlopen(img.getDownloadURL(params)).read())

    def hillshade(arr, az, alt):
        az, alt = np.deg2rad([az, alt])
        dy, dx  = np.gradient(arr.astype("float32"), edge_order=2)
        slope   = np.arctan(np.hypot(dx, dy))
        aspect  = np.arctan2(dy, -dx)
        hs = (np.sin(alt)*np.cos(slope) + np.cos(alt)*np.sin(slope)*np.cos(az-aspect))
        return (np.clip(hs, 0, 1)*255).astype("uint8")

    # ─────────────────────── Main loop ───────────────────────
    matches = []
    counts  = {}
    with open(PATH_TO_COORDS_CSV, newline="", encoding="utf-8") as cf:
        for row in csv.DictReader(cf):
            stem   = Path(row["filename"]).stem
            prefix = "_".join(stem.split("_")[:2])
            cnt    = counts.get(prefix, 0)
            if cnt >= MAX_PHOTOS_PER_PREFIX:
                continue
            counts[prefix] = cnt + 1
            minx, miny = float(row["min_lon"]), float(row["min_lat"])
            maxx, maxy = float(row["max_lon"]), float(row["max_lat"])
            center     = [(minx+maxx)/2, (miny+maxy)/2]
            matches.append((stem, center))

    logging.info(f"Found {len(matches)} tiles across {len(counts)} prefixes")

    for stem, pt in matches:
        logging.info(f"Processing {stem}")
        td = OUT_DIR / stem
        td.mkdir(exist_ok=True)

        # LiDAR
        ld = DTM_DIR / f"{stem}.tif"
        if not ld.exists():
            logging.warning(f"Missing DTM for {stem}")
            continue
        shutil.copy(ld, td / f"{stem}_lidar.tif")

        start, end = "2024-01-01", f"{DATE}T23:59:59"
        # S1
        img1, reg1 = get_best(S1_COLL, pt, start, end, S1_VIS)
        if img1:
            save_visual(img1, reg1, td / f"{stem}_S1_{DATE}.jpg", S1_VIS)
            save_geotiff(img1, reg1, td / f"{stem}_S1_{DATE}.tif")
        else:
            logging.warning(f"No S1 for {stem}")

        # S2
        img2, reg2 = get_best(S2_COLL, pt, start, end, S2_VIS, cloud_filter=True)
        if img2:
            save_visual(img2, reg2, td / f"{stem}_S2_{DATE}.jpg", S2_VIS)
            save_geotiff(img2, reg2, td / f"{stem}_S2_{DATE}.tif", bands=['B4','B3','B2'])
        else:
            logging.warning(f"No S2 for {stem}")

        # hillshade
        arr = rasterio.open(ld).read(1).astype("float32")
        arr[arr == rasterio.open(ld).nodata] = np.nan
        hs1 = hillshade(arr, AZ1, ALT1)
        hs2 = hillshade(arr, AZ2, ALT2)
        comp = np.concatenate([hs1, hs2], axis=1)
        Image.fromarray(comp).save(td / f"{stem}_lidar_hillshade.jpg", quality=90)

    logging.info("Done processing all tiles.")
    
if __name__ == "__main__":
    retrieve_tiles()
