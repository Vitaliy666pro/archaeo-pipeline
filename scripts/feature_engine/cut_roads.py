import os
import yaml
import pathlib
import requests
import zipfile
import pandas as pd
import geopandas as gpd
import pyogrio
from shapely.geometry import box, Point
from tqdm.auto import tqdm

def cut_roads():
    """
    1) Ensures the Central South America roads dataset is downloaded and extracted.
    2) Loads tile polygons and road lines, then removes all tiles that intersect any road.
    3) Saves filtered tiles to processed_data_dir/all_tiles_without_roads.pkl.

    Returns:
        GeoDataFrame: Tiles without roads (same columns as input).
    """
    # ─────────────────────────── Load config ───────────────────────────
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    RAW_DIR       = pathlib.Path(config["raw_data_dir"])
    PROCESSED_DIR = pathlib.Path(config["processed_data_dir"])
    DATASETS_DIR  = RAW_DIR / "datasets"
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    # ───────────────────────── Download & extract roads dataset ─────────────────────────
    ZIP_PATH = DATASETS_DIR / "central_southamerica.zip"
    URL      = "https://datacatalogfiles.worldbank.org/ddh-published/0040289/DR0050247/central_southamerica.zip"
    if not ZIP_PATH.exists():
        print("→ Downloading central_southamerica.zip …")
        with requests.get(URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(ZIP_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"✓ Saved to {ZIP_PATH}")
    else:
        print(f"✓ Archive already present: {ZIP_PATH}")

    print(f"→ Extracting roads dataset to {DATASETS_DIR} …")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(DATASETS_DIR)
    print(f"✓ Extracted to {DATASETS_DIR}")

    # ───────────────────────── Read tiles GeoDataFrame ─────────────────────────
    TILES_PKL = PROCESSED_DIR / "all_tiles_features_with_emb.pkl"
    if not TILES_PKL.exists():
        raise FileNotFoundError(f"{TILES_PKL} not found! Run embedding step first.")
    tiles_df = pd.read_pickle(TILES_PKL)
    tiles = gpd.GeoDataFrame(tiles_df, geometry="geometry")
    # ensure CRS is Web Mercator
    if tiles.crs is None or tiles.crs.to_epsg() != 3857:
        tiles = tiles.set_crs(3857, allow_override=True)

    # ───────────────────────── Load roads from GDB ─────────────────────────
    # Find the first .gdb folder in datasets
    gdb_dirs = [p for p in DATASETS_DIR.rglob("*.gdb") if p.is_dir()]
    if not gdb_dirs:
        raise FileNotFoundError(f"No .gdb found in {DATASETS_DIR} (check extraction).")
    gdb_path = gdb_dirs[0]
    layer_name = pyogrio.list_layers(gdb_path)[0][0]

    roads = gpd.read_file(gdb_path, layer=layer_name, engine="pyogrio")
    # project to EPSG:3857 if necessary
    if roads.crs is None or roads.crs.to_epsg() != 3857:
        roads = roads.to_crs(3857)

    # ───────────────────────── Clip roads to AOI ─────────────────────────
    minx, miny, maxx, maxy = tiles.total_bounds
    aoi = box(minx-10_000, miny-10_000, maxx+10_000, maxy+10_000)
    roads_clip = roads.clip(aoi)

    # ───────────────────────── Filter out tiles intersecting roads ─────────────────────────
    ridx = roads_clip.sindex
    keep = []
    for poly in tqdm(tiles.geometry, desc="Filtering tiles without roads"):
        candidates = list(ridx.query(poly, predicate="intersects"))
        if not candidates:
            keep.append(True)
        else:
            intersects_any = roads_clip.iloc[candidates].intersects(poly).any()
            keep.append(not intersects_any)

    clean_tiles = tiles.loc[keep].copy()

    # ───────────────────────── Save filtered tiles ─────────────────────────
    OUT_PKL = PROCESSED_DIR / "all_tiles_without_roads.pkl"
    clean_tiles.to_pickle(OUT_PKL)
    print(f"✓ Saved filtered tiles to {OUT_PKL} ({len(clean_tiles)}/{len(tiles)})")

    return clean_tiles


if __name__ == "__main__":
    cut_roads()
