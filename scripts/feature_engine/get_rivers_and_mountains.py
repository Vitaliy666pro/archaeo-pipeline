# import os
# import yaml
# import pickle
# import numpy as np
# import pandas as pd
# import geopandas as gpd
# import requests
# from pathlib import Path
# from tqdm import tqdm
# from shapely.geometry import box, Point, mapping
# from rasterio.mask import mask as rio_mask
# import rasterio
# from joblib import Parallel, delayed
# from dotenv import load_dotenv

# # ========== Relief Features ==========

# def load_dem_data(dem_type, aoi, out_file, interim_dir="."):
#     url = "https://portal.opentopography.org/API/globaldem"
#     params = {
#         "demtype": dem_type,
#         "west": aoi[0], "south": aoi[1],
#         "east": aoi[2], "north": aoi[3],
#         "outputFormat": "GTiff",
#         "API_Key": os.getenv("OPENTOPO_KEY"),
#     }
#     os.makedirs(interim_dir, exist_ok=True)
#     out_path = os.path.join(interim_dir, out_file)
#     resp = requests.get(url, params=params, timeout=180)
#     resp.raise_for_status()
#     with open(out_path, "wb") as f:
#         f.write(resp.content)
#     return out_path

# def add_relief_features(gdf, dem_path):
#     means, slopes = [], []
#     with rasterio.open(dem_path) as src:
#         crs = src.crs
#         arr = src.read(1).astype(np.float32)
#         nodata = src.nodata
#         arr[arr == nodata] = np.nan if nodata is not None else arr
#         dx, dy = abs(src.transform.a), abs(src.transform.e)
#         slope = np.degrees(np.arctan(np.hypot(np.gradient(arr, dx, axis=1), np.gradient(arr, dy, axis=0))))
#         for poly in tqdm(gdf.to_crs(crs).geometry, desc="relief"):
#             try:
#                 window, _ = rio_mask(src, [mapping(poly)], crop=True, filled=False)
#                 band = window[0]
#                 means.append(float(np.nanmean(band)))
#                 cen = poly.centroid
#                 row, col = src.index(cen.x, cen.y)
#                 slopes.append(float(slope[row, col]))
#             except:
#                 means.append(np.nan)
#                 slopes.append(np.nan)
#     gdf["mean_elev_m"] = means
#     gdf["mean_slope_deg"] = slopes
#     return gdf

# def add_geometry_point(gdf, method="centroid"):
#     def _make_point(poly):
#         if poly is None or poly.is_empty:
#             return None
#         if method == "centroid":
#             return poly.centroid
#         elif method == "first_vert":
#             return Point(*poly.exterior.coords[0])
#         elif method == "bounds":
#             return Point(poly.bounds[0], poly.bounds[1])
#         return None

#     crs = gdf.crs or "EPSG:3857"  # ← гарантируем, что CRS есть
#     gdf["geometry_point"] = gpd.GeoSeries(gdf.geometry.apply(_make_point), crs=crs)
#     return gdf


# def flag_mountains(gdf, slope_field="mean_slope_deg", slope_thr=20, elev_thr=300):
#     gdf["is_mountain"] = (gdf[slope_field] > slope_thr) & (gdf["mean_elev_m"] > elev_thr)
#     return gdf

# def add_distance_to_mountains(gdf, mountain_col="is_mountain", point_col="geometry_point"):
#     pts = np.array([[p.x, p.y] for p in gdf.loc[gdf[mountain_col], point_col]])
#     dists = []
#     for _, row in tqdm(gdf.iterrows(), total=len(gdf), desc="dist2mountain"):
#         pt = row[point_col]
#         dists.append(0.0 if row[mountain_col] else float(np.hypot(pts[:, 0] - pt.x, pts[:, 1] - pt.y).min()) if pts.size else np.nan)
#     gdf["dist_to_mountain_m"] = dists
#     return gdf

# # ========== River Features ==========
# def get_distance_to_rivers_parallel(
#     geo_gdf: gpd.GeoDataFrame,
#     riv_gdf: gpd.GeoDataFrame,
#     geometry_col: str = "geometry",
#     n_jobs: int = -1
# ) -> gpd.GeoDataFrame:

#     if geometry_col not in geo_gdf.columns:
#         raise ValueError(f"COL '{geometry_col}' NOT FOUND IN geo_gdf.")

#     river_union = riv_gdf.geometry.unary_union

#     def _dist(pt):
#         return pt.distance(river_union) if pt else None

#     distances = Parallel(n_jobs=n_jobs, prefer="threads")(
#         delayed(_dist)(pt)
#         for pt in tqdm(geo_gdf[geometry_col], desc=f"dist-to-river ({geometry_col})")
#     )

#     out = geo_gdf.copy()
#     out["dist_to_river_m"] = distances
#     return out

# def add_river_attrs_sjoin(tiles, rivers, tile_pt_col="geometry_point", river_geom_col="geometry", river_rank_col="ORD_FLOW", river_area_col="UPLAND_SKM", metric_crs="EPSG:3857"):
#     tiles_m = tiles.set_geometry(tile_pt_col).to_crs(metric_crs)
#     rivers_m = rivers[[river_geom_col, river_rank_col, river_area_col]].set_geometry(river_geom_col).to_crs(metric_crs)
#     joined = gpd.sjoin_nearest(tiles_m, rivers_m, how="left", distance_col="dist_to_river_m")
#     joined = joined.rename(columns={river_rank_col: "ord_flow", river_area_col: "upland_skm"})
#     return joined.to_crs(tiles.crs)

# def compute_drainage_density(
#     tiles_gdf,
#     rivers_gdf,
#     tile_geom_col: str = "geometry",
#     river_geom_col: str = "geometry",
#     metric_crs: str = "EPSG:3857",
#     source_crs: str = "EPSG:4326"
# ) -> gpd.GeoDataFrame:
#     """
#     Computes drainage density per tile.

#     Parameters:
#     - tiles_gdf: GeoDataFrame with tile polygons
#     - rivers_gdf: GeoDataFrame with river geometries
#     - tile_geom_col: name of the geometry column in tiles
#     - river_geom_col: name of the geometry column in rivers
#     - metric_crs: projected CRS for distance/area
#     - source_crs: fallback CRS if not defined

#     Returns:
#     - tiles_gdf copy with columns: tile_area_km2, drainage_density_m, drainage_density
#     """
#     # Ensure valid geometry columns
#     tiles = tiles_gdf.copy().set_geometry(tile_geom_col)
#     rivers = rivers_gdf.copy().set_geometry(river_geom_col)

#     # Set CRS if missing
#     if tiles.crs is None:
#         tiles = tiles.set_crs(metric_crs if abs(tiles.total_bounds[2]) > 180 else source_crs)
#     if rivers.crs is None:
#         rivers = rivers.set_crs(metric_crs if abs(rivers.total_bounds[2]) > 180 else source_crs)

#     # Project to metric CRS
#     tiles = tiles.to_crs(metric_crs)
#     rivers = rivers.to_crs(metric_crs)

#     # Build spatial index
#     sindex = rivers.sindex

#     # Prepare outputs
#     lengths = []
#     areas = []
#     densities = []

#     for poly in tqdm(tiles.geometry, desc="drainage_density", unit="tile"):
#         area_km2 = poly.area / 1e6  # m² → km²
#         idxs = list(sindex.intersection(poly.bounds))

#         if not idxs:
#             length_m = 0.0
#         else:
#             candidates = rivers.iloc[idxs]
#             inter = candidates[candidates.geometry.intersects(poly)]
#             if inter.empty:
#                 length_m = 0.0
#             else:
#                 clipped = inter.geometry.intersection(poly)
#                 # clipped is a GeoSeries — .length is vectorized
#                 length_m = clipped.length.sum()

#         lengths.append(length_m)
#         areas.append(area_km2)
#         densities.append((length_m / 1000) / area_km2 if area_km2 > 0 else np.nan)

#     out = tiles_gdf.copy()
#     out["tile_area_km2"] = areas
#     out["drainage_density_m"] = lengths
#     out["drainage_density"] = densities
#     return out

# # ========== Other ==========

# def flatten_tile_hits_with_negatives(df):
#     list_cols = [c for c in df.columns if c.endswith("_list")]
#     static_cols = [c for c in df.columns if c not in list_cols + ["geometry"]]
#     rows = []
#     for _, row in df.iterrows():
#         n = int(row.get("n_geoglyphs", 0))
#         if n > 0:
#             for i in range(n):
#                 rec = {c: row[c] for c in static_cols}
#                 rec["geometry"] = row.geometry
#                 for lc in list_cols:
#                     base = lc[:-5]
#                     lst = row[lc] or []
#                     rec[base] = lst[i] if i < len(lst) else None
#                 rows.append(rec)
#         else:
#             rec = {c: row[c] for c in static_cols}
#             rec["geometry"] = row.geometry
#             for lc in list_cols:
#                 rec[lc[:-5]] = None
#             rows.append(rec)
#     return gpd.GeoDataFrame(pd.DataFrame.from_records(rows), geometry="geometry", crs=df.crs)

# def load_country_shapes(crs="EPSG:3857", use_bbox=False, countries=None):
#     if countries is None:
#         countries = ["Brazil", "Bolivia", "Colombia", "Ecuador", "Guyana", "Peru", "Suriname", "Venezuela", "French Guiana"]
#     try:
#         from geodatasets import get_path
#         world = gpd.read_file(get_path("naturalearth_lowres"))
#     except:
#         url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
#         world = gpd.read_file(url)
#     name_col = next(c for c in ("NAME", "ADMIN", "NAME_EN") if c in world.columns)
#     world = world.rename(columns={name_col: "name"})
#     world = world[world["name"].str.lower().isin([c.lower() for c in countries])]
#     if use_bbox:
#         world["geometry"] = world.geometry.bounds.apply(lambda r: box(*r), axis=1)
#     return world.to_crs(crs)[["name", "geometry"]]

# # ========== Final Entrypoint ==========

# def get_rivers_and_mountains():
#     load_dotenv()
#     with open("config.yaml", "r") as f:
#         cfg = yaml.safe_load(f)

#     raw_dir = Path(cfg["raw_data_dir"])
#     processed_dir = Path(cfg["processed_data_dir"])
#     interim_dir = Path(cfg["interim_data_dir"])
#     aoi_crs = cfg["aoi_crs"]
#     metric_crs = cfg["metric_crs"]
#     aoi_box = cfg["aoi_box"]
#     dem_type = "SRTMGL3"

#     rivers_path = raw_dir / "datasets/hydrorivers-dataset/HydroRIVERS.gdb"
#     rivers = gpd.read_file(rivers_path, bbox=tuple(aoi_box))
#     rivers = rivers[rivers["ORD_FLOW"].astype(float) <= 6].to_crs(metric_crs)

#     regions = {}
#     for i in range(1, 7):
#         fp = processed_dir / f"region_{i}_tagged_with_site.gpkg"
#         gdf = gpd.read_file(fp).to_crs(aoi_crs)
#         regions[f"region_{i}"] = gdf

#     result = {}
#     for name, gdf in regions.items():
#         bbox = list(gdf.to_crs(aoi_crs).total_bounds)
#         dem_path = load_dem_data(dem_type, bbox, f"{name.lower()}_{dem_type.lower()}_90m.tif", interim_dir)
#         gdf = add_relief_features(gdf, dem_path)
#         gdf = add_geometry_point(gdf)
#         gdf = flag_mountains(gdf)
#         gdf = add_distance_to_mountains(gdf)
#         get_distance_to_rivers_parallel(
#             geo_gdf      = gdf,
#             riv_gdf      = rivers,
#             geometry_col = "geometry", 
#             n_jobs       = -1
#         )        
#         gdf = add_river_attrs_sjoin(gdf, rivers)
#         gdf = compute_drainage_density(gdf, rivers, metric_crs=metric_crs, source_crs=aoi_crs)
#         gdf = add_geometry_point(gdf)
#         result[name] = gdf

#     flattened = {}
#     for name, tiles in result.items():
#         tiles = tiles.to_crs(metric_crs)
#         tiles = add_geometry_point(tiles)
#         flat = flatten_tile_hits_with_negatives(tiles)
#         flat["region"] = name
#         flattened[name] = flat

#     countries_gdf = load_country_shapes(crs=metric_crs)
#     for region, flat in flattened.items():
#         flat = flat.set_geometry("geometry_point").to_crs(metric_crs)
#         joined = gpd.sjoin(flat, countries_gdf, how="left", predicate="within")
#         flat["country"] = joined["name"]
#         flattened[region] = flat

#     all_tiles_flat = gpd.GeoDataFrame(pd.concat(flattened.values(), ignore_index=True), geometry="geometry", crs=metric_crs)
#     with open(processed_dir / "all_tiles_flat.pkl", "wb") as f:
#         pickle.dump(all_tiles_flat, f)

#     return all_tiles_flat


# if __name__ == "__main__":
#     get_rivers_and_mountains()
import os
import yaml
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from pathlib import Path
from tqdm import tqdm
from shapely.geometry import box, Point, mapping
from rasterio.mask import mask as rio_mask
import rasterio
from joblib import Parallel, delayed
from dotenv import load_dotenv


# ========== Relief Features ==========

def load_dem_data(dem_type, aoi, out_file, interim_dir="."):
    url = "https://portal.opentopography.org/API/globaldem"
    params = {
        "demtype": dem_type,
        "west": aoi[0], "south": aoi[1],
        "east": aoi[2], "north": aoi[3],
        "outputFormat": "GTiff",
        "API_Key": os.getenv("OPENTOPO_KEY"),
    }
    os.makedirs(interim_dir, exist_ok=True)
    out_path = os.path.join(interim_dir, out_file)
    resp = requests.get(url, params=params, timeout=180)
    resp.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(resp.content)
    return out_path


def add_relief_features(gdf, dem_path):
    means, slopes = [], []
    with rasterio.open(dem_path) as src:
        crs = src.crs
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata
        arr[arr == nodata] = np.nan if nodata is not None else arr
        dx, dy = abs(src.transform.a), abs(src.transform.e)
        slope = np.degrees(np.arctan(np.hypot(np.gradient(arr, dx, axis=1), np.gradient(arr, dy, axis=0))))
        for poly in tqdm(gdf.to_crs(crs).geometry, desc="relief"):
            try:
                window, _ = rio_mask(src, [mapping(poly)], crop=True, filled=False)
                band = window[0]
                means.append(float(np.nanmean(band)))
                cen = poly.centroid
                row, col = src.index(cen.x, cen.y)
                slopes.append(float(slope[row, col]))
            except:
                means.append(np.nan)
                slopes.append(np.nan)
    gdf["mean_elev_m"] = means
    gdf["mean_slope_deg"] = slopes
    return gdf


def add_geometry_point(gdf, method="centroid"):
    def _make_point(poly):
        if poly is None or poly.is_empty:
            return None
        if method == "centroid":
            return poly.centroid
        elif method == "first_vert":
            return Point(*poly.exterior.coords[0])
        elif method == "bounds":
            return Point(poly.bounds[0], poly.bounds[1])
        return None

    crs = gdf.crs or "EPSG:3857"
    gdf["geometry_point"] = gpd.GeoSeries(gdf.geometry.apply(_make_point), crs=crs)
    return gdf


def flag_mountains(gdf, slope_field="mean_slope_deg", slope_thr=20, elev_thr=300):
    gdf["is_mountain"] = (gdf[slope_field] > slope_thr) & (gdf["mean_elev_m"] > elev_thr)
    return gdf


def add_distance_to_mountains(gdf, mountain_col="is_mountain", point_col="geometry_point"):
    pts = np.array([[p.x, p.y] for p in gdf.loc[gdf[mountain_col], point_col]])
    dists = []
    for _, row in tqdm(gdf.iterrows(), total=len(gdf), desc="dist2mountain"):
        pt = row[point_col]
        dists.append(0.0 if row[mountain_col] else float(np.hypot(pts[:, 0] - pt.x, pts[:, 1] - pt.y).min()) if pts.size else np.nan)
    gdf["dist_to_mountain_m"] = dists
    return gdf


# ========== River Features ==========

def get_distance_to_rivers_parallel(geo_gdf, riv_gdf, geometry_col="geometry", n_jobs=-1):
    river_union = riv_gdf.geometry.unary_union
    def _dist(pt):
        return pt.distance(river_union) if pt else None
    distances = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_dist)(pt) for pt in tqdm(geo_gdf[geometry_col], desc="dist-to-river")
    )
    geo_gdf = geo_gdf.copy()
    geo_gdf["dist_to_river_m"] = distances
    return geo_gdf


def add_river_attrs_sjoin(tiles, rivers, tile_pt_col="geometry_point", river_geom_col="geometry", river_rank_col="ORD_FLOW", river_area_col="UPLAND_SKM", metric_crs="EPSG:3857"):
    tiles_m = tiles.set_geometry(tile_pt_col).to_crs(metric_crs)
    rivers_m = rivers[[river_geom_col, river_rank_col, river_area_col]].set_geometry(river_geom_col).to_crs(metric_crs)
    joined = gpd.sjoin_nearest(tiles_m, rivers_m, how="left", distance_col="dist_to_river_m")
    joined = joined.rename(columns={river_rank_col: "ord_flow", river_area_col: "upland_skm"})
    return joined.to_crs(tiles.crs)


def compute_drainage_density(tiles_gdf, rivers_gdf, tile_geom_col="geometry", river_geom_col="geometry", metric_crs="EPSG:3857", source_crs="EPSG:4326"):
    tiles = tiles_gdf.copy().set_geometry(tile_geom_col)
    rivers = rivers_gdf.copy().set_geometry(river_geom_col)
    if tiles.crs is None:
        tiles = tiles.set_crs(metric_crs if abs(tiles.total_bounds[2]) > 180 else source_crs)
    if rivers.crs is None:
        rivers = rivers.set_crs(metric_crs if abs(rivers.total_bounds[2]) > 180 else source_crs)
    tiles = tiles.to_crs(metric_crs)
    rivers = rivers.to_crs(metric_crs)
    sindex = rivers.sindex
    lengths, areas, densities = [], [], []
    for poly in tqdm(tiles.geometry, desc="drainage_density", unit="tile"):
        area_km2 = poly.area / 1e6
        idxs = list(sindex.intersection(poly.bounds))
        if not idxs:
            length_m = 0.0
        else:
            candidates = rivers.iloc[idxs]
            inter = candidates[candidates.geometry.intersects(poly)]
            if inter.empty:
                length_m = 0.0
            else:
                clipped = inter.geometry.intersection(poly)
                length_m = clipped.length.sum()
        lengths.append(length_m)
        areas.append(area_km2)
        densities.append((length_m / 1000) / area_km2 if area_km2 > 0 else np.nan)
    out = tiles_gdf.copy()
    out["tile_area_km2"] = areas
    out["drainage_density_m"] = lengths
    out["drainage_density"] = densities
    return out


# ========== Other ==========

def flatten_tile_hits_with_negatives(df):
    list_cols = [c for c in df.columns if c.endswith("_list")]
    static_cols = [c for c in df.columns if c not in list_cols + ["geometry"]]
    rows = []
    for _, row in df.iterrows():
        n = int(row.get("n_geoglyphs", 0))
        if n > 0:
            for i in range(n):
                rec = {c: row[c] for c in static_cols}
                rec["geometry"] = row.geometry
                for lc in list_cols:
                    base = lc[:-5]
                    lst = row[lc] or []
                    rec[base] = lst[i] if i < len(lst) else None
                rows.append(rec)
        else:
            rec = {c: row[c] for c in static_cols}
            rec["geometry"] = row.geometry
            for lc in list_cols:
                rec[lc[:-5]] = None
            rows.append(rec)
    return gpd.GeoDataFrame(pd.DataFrame.from_records(rows), geometry="geometry", crs=df.crs)


def load_country_shapes(crs="EPSG:3857", use_bbox=False, countries=None):
    if countries is None:
        countries = ["Brazil", "Bolivia", "Colombia", "Ecuador", "Guyana", "Peru", "Suriname", "Venezuela", "French Guiana"]
    try:
        from geodatasets import get_path
        world = gpd.read_file(get_path("naturalearth_lowres"))
    except:
        url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
        world = gpd.read_file(url)
    name_col = next(c for c in ("NAME", "ADMIN", "NAME_EN") if c in world.columns)
    world = world.rename(columns={name_col: "name"})
    world = world[world["name"].str.lower().isin([c.lower() for c in countries])]
    if use_bbox:
        world["geometry"] = world.geometry.bounds.apply(lambda r: box(*r), axis=1)
    return world.to_crs(crs)[["name", "geometry"]]


# ========== Final Entrypoint ==========

def get_rivers_and_mountains():
    load_dotenv()
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(cfg["raw_data_dir"])
    processed_dir = Path(cfg["processed_data_dir"])
    interim_dir = Path(cfg["interim_data_dir"])
    aoi_crs = cfg["aoi_crs"]
    metric_crs = cfg["metric_crs"]
    aoi_box = cfg["aoi_box"]
    dem_type = "SRTMGL3"

    rivers_path = raw_dir / "datasets/hydrorivers-dataset/HydroRIVERS.gdb"
    rivers = gpd.read_file(rivers_path, bbox=tuple(aoi_box))
    rivers = rivers[rivers["ORD_FLOW"].astype(float) <= 6].to_crs(metric_crs)

    regions = {}
    for i in range(1, 7):
        fp = processed_dir / f"region_{i}_tagged_with_site.gpkg"
        gdf = gpd.read_file(fp).to_crs(aoi_crs)
        regions[f"region_{i}"] = gdf

    result = {}
    for name, gdf in regions.items():
        bbox = list(gdf.to_crs(aoi_crs).total_bounds)
        dem_path = load_dem_data(dem_type, bbox, f"{name.lower()}_{dem_type.lower()}_90m.tif", interim_dir)
        gdf = add_relief_features(gdf, dem_path)
        gdf = add_geometry_point(gdf)
        gdf = flag_mountains(gdf)
        gdf = add_distance_to_mountains(gdf)
        gdf = get_distance_to_rivers_parallel(gdf, rivers, geometry_col="geometry", n_jobs=-1)
        gdf = add_river_attrs_sjoin(gdf, rivers)
        gdf = compute_drainage_density(gdf, rivers, metric_crs=metric_crs, source_crs=aoi_crs)
        gdf = add_geometry_point(gdf)
        result[name] = gdf

    regions_with_river_features = {k: v.to_crs(metric_crs) for k, v in result.items()}

    regions_updated = {}
    for name, gdf in regions_with_river_features.items():
        gdf = gdf.to_crs(metric_crs)
        gdf = add_geometry_point(gdf, method="centroid")
        regions_updated[name] = gdf

    flattened = {}
    for region, tiles in regions_updated.items():
        flat = flatten_tile_hits_with_negatives(tiles)
        flat = flat.drop(columns=["index_right", "name"], errors="ignore")
        flat["region"] = region
        flattened[region] = flat

    countries_gdf = load_country_shapes(crs=metric_crs, use_bbox=False)
    if "ADMIN" in countries_gdf.columns:
        countries_gdf = countries_gdf.rename(columns={"ADMIN": "name"})
    elif "NAME_EN" in countries_gdf.columns:
        countries_gdf = countries_gdf.rename(columns={"NAME_EN": "name"})
    countries_gdf = countries_gdf[["name", "geometry"]]

    regions_flat_with_country = {}
    for region, flat in flattened.items():
        flat = flat.set_geometry("geometry_point")
        if flat.crs is None:
            flat = flat.set_crs(metric_crs, allow_override=True)
        else:
            flat = flat.to_crs(metric_crs)
        joined = gpd.sjoin(flat, countries_gdf, how="left", predicate="within")
        out = flat.copy()
        out["country"] = joined["name"]
        regions_flat_with_country[region] = out

    all_tiles_flat = pd.concat(regions_flat_with_country.values(), ignore_index=True)
    all_tiles_flat = gpd.GeoDataFrame(all_tiles_flat, geometry="geometry", crs=metric_crs)

    with open(processed_dir / "all_tiles_flat.pkl", "wb") as f:
        pickle.dump(all_tiles_flat, f)

    return all_tiles_flat


if __name__ == "__main__":
    get_rivers_and_mountains()


