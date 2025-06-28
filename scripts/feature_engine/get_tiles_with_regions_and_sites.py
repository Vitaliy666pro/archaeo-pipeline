import os
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from pyproj import Transformer
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import contextily as cx

# === Load config.yaml ===
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

RAW_DIR = Path(config["raw_data_dir"])
PROCESSED_DIR = Path(config["processed_data_dir"])
AOI_CRS = config["aoi_crs"]
METRIC_CRS = config["metric_crs"]

# === Core functions ===
def load_sites_from_df(sites_df, coord_col="coordinates", input_crs="EPSG:4326", metric_crs="EPSG:3857", aoi_bbox=None):
    coords = sites_df[coord_col].apply(eval)
    lons, lats = zip(*coords)
    df = sites_df.copy()
    df["longitude"] = pd.to_numeric(lons, errors="coerce")
    df["latitude"]  = pd.to_numeric(lats, errors="coerce")
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs=input_crs).dropna(subset=["geometry"])
    if aoi_bbox is not None:
        xmin, ymin, xmax, ymax = aoi_bbox
        gdf = gdf.cx[xmin:xmax, ymin:ymax]
    return gdf.to_crs(metric_crs)

def load_main_rivers(rivers_path, aoi_bbox, aoi_crs, metric_crs):
    rivers = gpd.read_file(rivers_path, bbox=aoi_bbox).to_crs(aoi_crs)
    main_rivers = rivers[rivers["ORD_FLOW"].astype(float) <= 6].copy()
    return main_rivers.to_crs(metric_crs)

def get_distance_to_rivers_parallel(geo_gdf, riv_gdf, n_jobs=-1):
    from shapely.ops import unary_union
    river_union = unary_union(riv_gdf.geometry)
    def _dist(pt): return pt.distance(river_union)
    distances = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(_dist)(pt) for pt in tqdm(geo_gdf.geometry, desc="dist-to-river"))
    geo_gdf = geo_gdf.copy()
    geo_gdf["dist_to_river_m"] = distances
    return geo_gdf

def generate_tiles_from_buffer(in_gdf, aoi_bbox, aoi_crs, tile_size_km, out_path, n_jobs=-1):
    in_gdf = in_gdf.to_crs(aoi_crs)
    lon_min, lat_min, lon_max, lat_max = aoi_bbox
    tile_deg = tile_size_km * 0.008333

    footprint = in_gdf.geometry.unary_union  # Optional: .buffer(0) if geometry is dirty
    lat_steps = int((lat_max - lat_min) / tile_deg)
    lon_steps = int((lon_max - lon_min) / tile_deg)

    print(f"⏳ Generating {lat_steps * lon_steps} potential tiles in parallel...")

    def process_lat_band(lat_idx):
        lat = lat_max - lat_idx * tile_deg
        row_tiles = []
        for lon_idx in range(lon_steps):
            lon = lon_min + lon_idx * tile_deg
            geom = box(lon, lat - tile_deg, lon + tile_deg, lat)
            if geom.intersects(footprint):
                tid = lat_idx * lon_steps + lon_idx
                row_tiles.append({"tile_id": tid, "geometry": geom})
        return row_tiles

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_lat_band)(lat_idx)
        for lat_idx in tqdm(range(lat_steps), desc="Latitude bands")
    )

    tiles_flat = [tile for sublist in results for tile in sublist]
    tiles_gdf = gpd.GeoDataFrame(tiles_flat, crs=aoi_crs, geometry="geometry")

    print(f"• {len(tiles_gdf):,} tiles intersect the buffer")
    tiles_gdf.to_file(out_path, driver="GPKG")
    print(f"✓ Grid written → {out_path}")
    tiles_gdf = tiles_gdf.to_crs("EPSG:3857")
    return tiles_gdf

def tag_tiles_by_geoglyphs(tiles_gdf, geoglyphs_gdf):
    tiles = tiles_gdf.to_crs("EPSG:4326")
    glyphs = geoglyphs_gdf.to_crs("EPSG:4326")
    hits = gpd.sjoin(tiles[["tile_id", "geometry"]], glyphs, how="left", predicate="intersects").drop(columns="index_right")
    point_fields = [c for c in glyphs.columns if c != "geometry"]
    agg_dict = {"n_geoglyphs": ("name", "count"), "has_geoglyph": ("name", lambda s: s.notna().any())}
    for fld in point_fields:
        agg_dict[f"{fld}_list"] = (fld, lambda s: list(s.dropna()))
    summary = hits.groupby("tile_id").agg(**agg_dict).reset_index()
    out = tiles_gdf.merge(summary, on="tile_id", how="left")
    out["n_geoglyphs"] = out["n_geoglyphs"].fillna(0).astype(int)
    out["has_geoglyph"] = out["has_geoglyph"].fillna(False).astype(bool)
    for fld in point_fields:
        col = f"{fld}_list"
        out[col] = out[col].apply(lambda x: x if isinstance(x, list) else [])
    return out


# 3. Crop all layers by latitude cutoff
def crop_and_plot_by_latitude(rivers_gdf, sites_gdf, lidar_gdf, lat_cut_deg, aoi_crs="EPSG:4326", metric_crs="EPSG:3857", figsize=(7, 7)):
    tf = Transformer.from_crs(aoi_crs, metric_crs, always_xy=True)
    _, y_cut = tf.transform(0, lat_cut_deg)
    xmin, _, xmax, ymax = sites_gdf.total_bounds

    rivers_cut = rivers_gdf.cx[xmin:xmax, y_cut:ymax]
    sites_cut  = sites_gdf.cx[xmin:xmax, y_cut:ymax]
    lidar_cut  = lidar_gdf.cx[xmin:xmax, y_cut:ymax]

    fig, ax = plt.subplots(figsize=figsize)
    rivers_cut.plot(ax=ax, linewidth=1, color="blue", label="Main rivers", zorder=1)
    sites_cut.plot(ax=ax, color="red", markersize=20, label="Arch sites", zorder=2)
    lidar_cut.plot(ax=ax, marker='o', facecolor='yellow', edgecolor='black', linewidth=0.5, markersize=200, alpha=1.0, label='LiDAR coverage', zorder=5)

    cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(y_cut, ymax)
    ax.set_axis_off()
    ax.legend(fontsize=10, loc="upper right")
    plt.tight_layout()
    plt.show()

    return rivers_cut, sites_cut, lidar_cut

# 4. Split AOI into grid
def split_aoi_into_grid(rivers_gdf, sites_gdf, lidar_gdf, n_cols, n_rows, figsize_per_cell=(5, 5)):
    xmin, ymin, xmax, ymax = sites_gdf.total_bounds
    xs = np.linspace(xmin, xmax, n_cols + 1)
    ys = np.linspace(ymin, ymax, n_rows + 1)

    regions = []
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_per_cell[0] * n_cols, figsize_per_cell[1] * n_rows), squeeze=False)

    for i in range(n_rows):
        for j in range(n_cols):
            bx, by = xs[j], ys[i]
            tx, ty = xs[j + 1], ys[i + 1]
            bbox = (bx, by, tx, ty)

            rivers_cell = rivers_gdf.cx[bx:tx, by:ty]
            sites_cell  = sites_gdf.cx[bx:tx, by:ty]
            lidar_cell  = lidar_gdf.cx[bx:tx, by:ty]

            ax = axes[i][j]
            rivers_cell.plot(ax=ax, linewidth=1, color="blue", zorder=1)
            sites_cell.plot(ax=ax, color="red", markersize=20, zorder=2)
            lidar_cell.plot(ax=ax, marker='o', facecolor='yellow', edgecolor='black', linewidth=0.5, markersize=200, alpha=1.0, zorder=5)

            cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
            ax.set_xlim(bx, tx)
            ax.set_ylim(by, ty)
            ax.set_axis_off()
            ax.set_title(f"Zone {i+1},{j+1}", fontsize=10)

            regions.append((rivers_cell, sites_cell, lidar_cell, bbox))

    plt.tight_layout()
    plt.show()
    return regions


# 8. Plot tiles by geoglyph tag
def tag_and_plot_tiles_region(tiles_gdf, geoglyphs_gdf, metric_crs="EPSG:3857", title_suffix=""):
    tagged = tag_tiles_by_geoglyphs(tiles_gdf, geoglyphs_gdf)
    hit = tagged[tagged["has_geoglyph"]]
    not_hit = tagged[~tagged["has_geoglyph"]]
    hit_web = hit.to_crs(metric_crs)
    not_hit_web = not_hit.to_crs(metric_crs)

    fig, ax = plt.subplots(figsize=(8, 8))
    hit_web.plot(ax=ax, edgecolor="green", facecolor="none", linewidth=2, label="With geoglyphs")
    not_hit_web.plot(ax=ax, edgecolor="red", facecolor="none", linewidth=0.5, label="Without geoglyphs")

    cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
    ax.set_title(f"Tiles Containing Geoglyphs {title_suffix}", fontsize=14)
    ax.set_axis_off()
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def get_tiles_with_reg_and_sites():
    sites = pd.read_csv(PROCESSED_DIR / "all_coords.csv")
    lidar_csv = pd.read_csv(RAW_DIR / "datasets" / "cms_brazil_lidar_tile_inventory.csv")
    gdf_sites = load_sites_from_df(sites, "coordinates", AOI_CRS, METRIC_CRS, None)
    lidar_gdf = gpd.GeoDataFrame(lidar_csv, geometry=lidar_csv.apply(lambda r: box(r.min_lon, r.min_lat, r.max_lon, r.max_lat), axis=1), crs=AOI_CRS).to_crs(METRIC_CRS)
    lidar_centroids = lidar_gdf.copy()
    lidar_centroids["geometry"] = lidar_gdf.geometry.centroid

    # Calculate full extent
    xmin, ymin, xmax, ymax = min(gdf_sites.total_bounds[0], lidar_gdf.total_bounds[0]), \
                             min(gdf_sites.total_bounds[1], lidar_gdf.total_bounds[1]), \
                             max(gdf_sites.total_bounds[2], lidar_gdf.total_bounds[2]), \
                             max(gdf_sites.total_bounds[3], lidar_gdf.total_bounds[3])
    transformer = Transformer.from_crs(METRIC_CRS, AOI_CRS, always_xy=True)
    min_lon, min_lat = transformer.transform(xmin, ymin)
    max_lon, max_lat = transformer.transform(xmax, ymax)
    aoi_bbox = (min_lon, min_lat, max_lon, max_lat)
    gdf_all = load_sites_from_df(sites, "coordinates", AOI_CRS, METRIC_CRS, aoi_bbox)
    rivers_path = RAW_DIR / "datasets" / "hydrorivers-dataset" / "HydroRIVERS.gdb"
    main_rivers_web = load_main_rivers(rivers_path, aoi_bbox, AOI_CRS, METRIC_CRS)
    buffers = main_rivers_web.buffer(9000)
    buffers_gdf = gpd.GeoDataFrame(geometry=buffers, crs=METRIC_CRS)

    tiles_gdf = generate_tiles_from_buffer(
        in_gdf=buffers_gdf,
        aoi_bbox=aoi_bbox,
        aoi_crs=AOI_CRS,
        tile_size_km=9,
        out_path=PROCESSED_DIR / "tiles_9km.gpkg",
        n_jobs=-1
    )

    tiles_web = tiles_gdf.to_crs(METRIC_CRS)

    # Generate regional bboxes (hardcoded 3x2 split like before)
    from shapely.geometry import box as make_box
    xs = np.linspace(*gdf_all.total_bounds[[0, 2]], 4)
    ys = np.linspace(*gdf_all.total_bounds[[1, 3]], 3)
    region_bboxes = [(xs[j], ys[i], xs[j+1], ys[i+1]) for i in range(2) for j in range(3)]

    # Tag and save per region
    for idx, bbox in enumerate(region_bboxes, start=1):
        xmin, ymin, xmax, ymax = bbox
        tiles_sub = tiles_web.cx[xmin:xmax, ymin:ymax].copy()
        tagged = tag_tiles_by_geoglyphs(tiles_sub, gdf_all)
        out_path = PROCESSED_DIR / f"region_{idx}_tagged_with_site.gpkg"
        tagged.to_file(out_path, driver="GPKG")
        print(f"✔ Saved {len(tagged)} tiles → {out_path}")

    return tiles_web

# Optional script entrypoint
if __name__ == "__main__":
    get_tiles_with_reg_and_sites()
