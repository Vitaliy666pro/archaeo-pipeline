import os
import yaml
import pathlib
import pandas as pd
import geopandas as gpd
import pyogrio
from shapely.geometry import box
from tqdm.auto import tqdm

def cut_roads():
    """
    Loads tile polygons and road lines, then removes all tiles that intersect any road.
    Uses paths from config.yaml. Saves filtered tiles to processed_data_dir/all_tiles_features_WITHOUT_roads.pkl.

    Returns:
        GeoDataFrame: Tiles without roads (same columns as input).
    """
    # 1. Load paths from config.yaml
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    RAW_DIR = pathlib.Path(config["raw_data_dir"])
    PROCESSED_DIR = pathlib.Path(config["processed_data_dir"])
    DATASETS_DIR = RAW_DIR / "datasets"

    # 2. Read tiles from pickle (as pandas, then GeoDataFrame)
    TILES_PKL = PROCESSED_DIR / "all_tiles_features_with_emb.pkl"
    if not TILES_PKL.exists():
        raise FileNotFoundError(f"{TILES_PKL} not found!")
    tiles_df = pd.read_pickle(TILES_PKL)
    tiles = gpd.GeoDataFrame(tiles_df, geometry="geometry")
    if tiles.crs is None or tiles.crs.to_epsg() != 3857:
        tiles = tiles.set_crs(3857, allow_override=True)

    # 3. Find any GDB in datasets (after unzip)
    gdb_dirs = [p for p in DATASETS_DIR.rglob("*.gdb") if p.is_dir()]
    if not gdb_dirs:
        raise FileNotFoundError(f"No .gdb found in {DATASETS_DIR} (run unzip first!)")
    gdb_path = gdb_dirs[0]
    layers = list(pyogrio.list_layers(gdb_path))
    layer_name = layers[0][0]  # You can set manually if needed

    # 4. Read roads, project to EPSG:3857
    roads = gpd.read_file(gdb_path, layer=layer_name, engine="pyogrio")
    if roads.crs is None or roads.crs.to_epsg() != 3857:
        roads = roads.to_crs(3857)

    # 5. Clip roads to AOI with safety buffer
    minx, miny, maxx, maxy = tiles.total_bounds
    aoi = box(minx-10_000, miny-10_000, maxx+10_000, maxy+10_000)
    roads_clip = roads.clip(aoi)

    # 6. Build spatial index for roads and mask out intersecting tiles
    ridx = roads_clip.sindex
    keep_mask = []
    for poly in tqdm(tiles.geometry, total=len(tiles), desc="Filtering tiles without roads"):
        # Find road candidates by bbox
        cand_idx = list(ridx.query(poly, predicate="intersects"))
        if not cand_idx:
            keep_mask.append(True)
        else:
            # True if no real intersection
            keep_mask.append(
                not roads_clip.iloc[cand_idx].intersects(poly).any()
            )
    clean_tiles = tiles.loc[keep_mask].copy()

    # 7. Save result to pickle
    OUT_PKL = PROCESSED_DIR / "all_tiles_features_WITHOUT_roads.pkl"
    clean_tiles.to_pickle(OUT_PKL)
    print(f"✓ Saved filtered tiles to {OUT_PKL} ({len(clean_tiles)}/{len(tiles)})")

    return clean_tiles

if __name__ == "__main__":
    all_tiles_without_roads = cut_roads()
