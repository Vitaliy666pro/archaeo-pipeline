import os
import glob
import yaml
import pickle
import itertools

import numpy as np
import pandas as pd
import polars as pl
import geopandas as gpd

from shapely.geometry import Point
from sklearn.neighbors import BallTree
from sklearn.decomposition import PCA
from tqdm.auto import tqdm


def pick_point(row):
    pt = row.get("geometry_point")
    if isinstance(pt, Point):
        return pt
    return row.geometry.centroid


def build_tile_dataframe(df, crs="EPSG:3857"):
    gdf = gpd.GeoDataFrame(df.copy(), geometry=df["geometry"], crs=crs)
    gdf["geometry_point"] = df.apply(pick_point, axis=1)
    return gdf


def load_embedding_metadata(parquet_dir):
    parquets = glob.glob(os.path.join(parquet_dir, "*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No .parquet files found in {parquet_dir}")

    coords, ids, offsets = [], [], []
    offset = 0

    for path in tqdm(parquets, desc="Reading coords from parquet"):
        part = (
            pl.read_parquet(path, columns=["unique_id", "centre_lat", "centre_lon"])
            .to_pandas()
        )
        coords.append(np.vstack([part["centre_lat"], part["centre_lon"]]).T)
        ids.append(part["unique_id"].tolist())
        offsets.append((path, offset, offset + len(part)))
        offset += len(part)

    return np.vstack(coords), list(itertools.chain.from_iterable(ids)), offsets


def match_tiles_to_embeddings(gdf, emb_coords, emb_ids):
    # reproject to lat/lon
    gdf = gdf.to_crs("EPSG:4326")
    tile_xy = np.vstack([gdf.geometry_point.y.values, gdf.geometry_point.x.values]).T
    tree = BallTree(np.radians(emb_coords), metric="haversine")
    dist_rad, idx = tree.query(np.radians(tile_xy), k=1)
    dist_m = dist_rad[:, 0] * 6_371_000

    gdf["match_id"]   = [emb_ids[i] for i in idx[:, 0]]
    gdf["dist_to_emb"] = dist_m
    return gdf


def load_required_embeddings(needed_ids, file_offsets, emb_ids_flat):
    emb_vectors = {}
    emb_cols    = None

    for path, start, end in tqdm(file_offsets, desc="Loading embeddings blocks"):
        block_ids = emb_ids_flat[start:end]
        want     = needed_ids.intersection(block_ids)
        if not want:
            continue

        part_pl = (
            pl.read_parquet(path)
            .filter(pl.col("unique_id").is_in(list(want)))
            .select(["unique_id", "embedding"])
        )
        part = part_pl.to_pandas()
        mat  = np.vstack(part["embedding"].values)
        cols = [f"emb_{i}" for i in range(mat.shape[1])]
        if emb_cols is None:
            emb_cols = cols

        df_emb = pd.DataFrame(mat, columns=cols)
        df_emb.insert(0, "unique_id", part["unique_id"].values)
        for _, row in df_emb.iterrows():
            emb_vectors[row["unique_id"]] = row[cols].to_dict()

    return emb_vectors, emb_cols


def attach_embeddings(gdf, emb_vectors, emb_cols):
    # build a DataFrame from the dict and merge
    emb_items = list(emb_vectors.items())
    emb_df = pd.DataFrame(
        [v for _, v in tqdm(emb_items, desc="Building embedding DataFrame")],
        index=[k for k, _ in emb_items],
        columns=emb_cols,
    )
    emb_df.index.name = "match_id"
    return gdf.merge(emb_df, how="left", left_on="match_id", right_index=True)


def get_emb_pca(config_path="config.yaml"):
    """
    Load tiles+soil features, attach embeddings, perform PCA, and save to a pickle.
    Returns the final GeoDataFrame.
    """
    # 1) Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    processed_dir   = config["processed_data_dir"]
    parquet_dir     = config["embedding_parquet_dir"]
    input_path      = os.path.join(processed_dir, "all_tiles_features_with_soil.pkl")
    output_path     = os.path.join(processed_dir, "all_tiles_features_with_emb.pkl")

    # 2) Load raw DataFrame
    with open(input_path, "rb") as f:
        df = pickle.load(f)

    # 3) Build GeoDataFrame and match embeddings
    gdf, emb_cols = build_tile_dataframe(df), None
    emb_coords, emb_ids_flat, file_offsets = load_embedding_metadata(parquet_dir)
    gdf = match_tiles_to_embeddings(gdf, emb_coords, emb_ids_flat)

    needed_ids = set(gdf["match_id"])
    emb_vectors, emb_cols = load_required_embeddings(needed_ids, file_offsets, emb_ids_flat)
    gdf = attach_embeddings(gdf, emb_vectors, emb_cols)

    # 4) PCA on embeddings
    gdf[emb_cols] = gdf[emb_cols].fillna(0).astype("float32")
    pca = PCA(n_components=7, random_state=42)
    pcs = pca.fit_transform(gdf[emb_cols].values)

    for i in range(pcs.shape[1]):
        gdf[f"PC{i+1}"] = pcs[:, i]

    # 5) Select and type-cast final columns
    base_cols = [
        "tile_id", "n_geoglyphs", "has_geoglyph",
        "mean_elev_m", "mean_slope_deg", "geometry_point",
        "is_mountain", "dist_to_mountain_m", "dist_to_river_m",
        "country", "drainage_density_m",
        "tile_area_km2", "drainage_density", "tri", "twi",
        "curv_plan", "curv_prof", "geometry", "coordinates",
        "source", "bbox", "longitude", "latitude",
        "region", "clay_0_5cm", "ph_h2o_0_5cm", "soc_0_5cm",
    ]
    pc_cols  = [f"PC{i}" for i in range(1, 8)]
    all_cols = base_cols + pc_cols

    missing = [c for c in all_cols if c not in gdf.columns]
    if missing:
        raise KeyError(f"Missing expected columns: {missing}")

    final_gdf = gdf[all_cols].copy()

    # 6) Save to pickle
    with open(output_path, "wb") as f:
        pickle.dump(final_gdf, f)

    print(f"[✓] Saved {final_gdf.shape[1]} columns × {final_gdf.shape[0]} rows to:\n  {output_path}")
    return final_gdf


if __name__ == "__main__":
    get_emb_pca()
