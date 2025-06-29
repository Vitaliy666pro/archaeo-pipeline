import os
import json
import math
import yaml
import logging
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, Point, mapping
from pyproj import Transformer
from pathlib import Path
from dotenv import load_dotenv
import rasterio
from rasterio.windows import Window
from shapely.geometry import box

from joblib import Parallel, delayed
from tqdm import tqdm
import pickle
from rasterio.mask import mask as rio_mask
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

load_dotenv()
pd.set_option("display.max_columns", 250)
pd.set_option("display.max_rows", 250)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)
log = logging.getLogger(__name__)

def load_dem_data(dem_type, aoi, out_file, interim_dir="."):
    url = "https://portal.opentopography.org/API/globaldem"
    params = {
        "demtype":      dem_type,
        "west":         aoi[0],
        "south":        aoi[1],
        "east":         aoi[2],
        "north":        aoi[3],
        "outputFormat": "GTiff",
        "API_Key":      os.getenv("OPENTOPO_KEY"),
    }
    os.makedirs(interim_dir, exist_ok=True)
    out_path = os.path.join(interim_dir, out_file)
    response = requests.get(url, params=params, timeout=180)
    response.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(response.content)
    # Quick check
    try:
        with rasterio.open(out_path) as src:
            arr = src.read(1)
            log.info(f"DEM shape: {arr.shape}, min/max: {np.nanmin(arr)}/{np.nanmax(arr)}")
    except Exception as e:
        log.warning(f"Error reading DEM: {e}")
    return out_path

def add_relief_features(gdf: gpd.GeoDataFrame, dem_path: str) -> gpd.GeoDataFrame:
    means, slopes = [], []
    with rasterio.open(dem_path) as src:
        crs_dem = src.crs
        transform = src.transform
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None and nodata != 0:
            arr[arr == nodata] = np.nan
        dx, dy = abs(transform.a), abs(transform.e)
        dzdx = np.gradient(arr, dx, axis=1)
        dzdy = np.gradient(arr, dy, axis=0)
        slope_arr = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
        for poly in tqdm(gdf.to_crs(crs_dem).geometry, desc="relief"):
            try:
                window, _ = rio_mask(src, [mapping(poly)], crop=True, filled=False)
                band = window[0]
                means.append(float(np.nanmean(band)))
                cen = poly.centroid
                row, col = src.index(cen.x, cen.y)
                slopes.append(float(slope_arr[row, col]))
            except Exception:
                means.append(np.nan)
                slopes.append(np.nan)
    out = gdf.copy()
    out["mean_elev_m"]    = means
    out["mean_slope_deg"] = slopes
    return out

def add_geometry_point(gdf: gpd.GeoDataFrame, method: str = "centroid") -> gpd.GeoDataFrame:
    if "geometry" not in gdf.columns:
        raise ValueError("gdf must have a 'geometry' column.")
    def _make_point(poly):
        if poly is None or poly.is_empty:
            return None
        if method == "centroid":
            return poly.centroid
        elif method == "first_vert":
            x, y = poly.exterior.coords[0]
            return Point(x, y)
        elif method == "bounds":
            minx, miny, *_ = poly.bounds
            return Point(minx, miny)
        else:
            raise ValueError("Method must be 'centroid', 'first_vert', or 'bounds'.")
    out = gdf.copy()
    out["geometry_point"] = gpd.GeoSeries(out.geometry.apply(_make_point), crs=gdf.crs)
    return out

def flag_mountains(gdf: gpd.GeoDataFrame,
                   slope_field: str = "mean_slope_deg",
                   slope_thr: float = 20,
                   elev_thr: float = 300,
                   out_col: str = "is_mountain") -> gpd.GeoDataFrame:
    condition = (gdf[slope_field] > slope_thr) & (gdf["mean_elev_m"] > elev_thr)
    out = gdf.copy()
    out[out_col] = condition
    return out

def add_distance_to_mountains(gdf: gpd.GeoDataFrame,
                              mountain_col: str = "is_mountain",
                              point_col: str = "geometry_point",
                              out_col: str = "dist_to_mountain_m") -> gpd.GeoDataFrame:
    pts = np.array([[p.x, p.y] for p in gdf.loc[gdf[mountain_col], point_col]])
    dists = []
    for _, row in tqdm(gdf.iterrows(), total=len(gdf), desc="dist2mountain"):
        pt = row[point_col]
        if row[mountain_col]:
            dists.append(0.0)
        elif pts.size:
            dists.append(float(np.hypot(pts[:,0]-pt.x, pts[:,1]-pt.y).min()))
        else:
            dists.append(np.nan)
    out = gdf.copy()
    out[out_col] = dists
    return out   

def get_distance_to_rivers_parallel(
    geo_gdf: gpd.GeoDataFrame,
    riv_gdf: gpd.GeoDataFrame,
    geometry_col: str = "geometry",
    n_jobs: int = -1
) -> gpd.GeoDataFrame:
    if geometry_col not in geo_gdf.columns:
        raise ValueError(f"Колонка '{geometry_col}' не найдена в geo_gdf.")
    river_union = riv_gdf.geometry.unary_union
    def _dist(pt):
        return pt.distance(river_union) if pt else None
    distances = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_dist)(pt)
        for pt in tqdm(geo_gdf[geometry_col], desc=f"dist-to-river ({geometry_col})")
    )
    out = geo_gdf.copy()
    out["dist_to_river_m"] = distances
    return out

from shapely.strtree import STRtree

def get_distance_to_rivers_fast(
    geo_gdf: gpd.GeoDataFrame,
    riv_gdf: gpd.GeoDataFrame,
    geometry_col: str = "geometry"
) -> gpd.GeoDataFrame:
    if geometry_col not in geo_gdf.columns:
        raise ValueError(f"Column '{geometry_col}' not found in geo_gdf.")

    river_geoms = []
    for geom in riv_gdf.geometry:
        if geom is not None and hasattr(geom, "is_empty") and not geom.is_empty:
            river_geoms.append(geom)

    if not river_geoms:
        raise ValueError("No valid geometries found in riv_gdf.")

    tree = STRtree(river_geoms)

    def nearest_distance(pt):
        if pt is None or not hasattr(pt, "is_empty") or pt.is_empty:
            return None
        try:
            nearest_geom = tree.nearest(pt)
            if nearest_geom is None or not hasattr(nearest_geom, "distance"):
                return None
            return pt.distance(nearest_geom)
        except Exception:
            return None

    distances = [nearest_distance(pt) for pt in tqdm(geo_gdf[geometry_col], desc="dist-to-river")]
    out = geo_gdf.copy()
    out["dist_to_river_m"] = distances
    return out

def add_river_attrs_sjoin(tiles, rivers, tile_pt_col="geometry_point", river_geom_col="geometry", river_rank_col="ORD_FLOW", river_area_col="UPLAND_SKM", metric_crs="EPSG:3857"):
    tiles_m = tiles.set_geometry(tile_pt_col).to_crs(metric_crs)
    rivers_m = rivers[[river_geom_col, river_rank_col, river_area_col]].set_geometry(river_geom_col).to_crs(metric_crs)
    joined = gpd.sjoin_nearest(tiles_m, rivers_m, how="left", distance_col="dist_to_river_m")
    joined = joined.rename(columns={river_rank_col: "ord_flow", river_area_col: "upland_skm"})
    return joined.to_crs(tiles.crs)

def compute_drainage_density(
    tiles_gdf,
    rivers_gdf,
    tile_geom_col: str = "geometry",
    river_geom_col: str = "geometry",
    metric_crs: str = "EPSG:3857",
    source_crs: str = "EPSG:4326"):
    tiles = tiles_gdf.copy().set_geometry(tile_geom_col)
    rivers = rivers_gdf.copy().set_geometry(river_geom_col)
    if tiles.crs is None:
        bounds = tiles.total_bounds
        tiles.set_crs(metric_crs if abs(bounds[2]) > 180 else source_crs, inplace=True)
    if rivers.crs is None:
        bounds = rivers.total_bounds
        rivers.set_crs(metric_crs if abs(bounds[2]) > 180 else source_crs, inplace=True)
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
    out["drainage_density_m"] = lengths
    out["tile_area_km2"] = areas
    out["drainage_density"] = densities
    return out

def flatten_tile_hits_with_negatives(tiles_tagged: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    list_cols = [c for c in tiles_tagged.columns if c.endswith("_list")]
    static_cols = [c for c in tiles_tagged.columns if c not in list_cols + ["geometry"]]
    records = []
    for _, row in tiles_tagged.iterrows():
        n = int(row.get("n_geoglyphs", 0))
        if n > 0:
            for i in range(n):
                rec = {c: row[c] for c in static_cols}
                rec["geometry"] = row.geometry
                for lc in list_cols:
                    fld = lc[:-5]
                    lst = row[lc] or []
                    rec[fld] = lst[i] if i < len(lst) else None
                records.append(rec)
        else:
            rec = {c: row[c] for c in static_cols}
            rec["geometry"] = row.geometry
            for lc in list_cols:
                rec[lc[:-5]] = None
            records.append(rec)
    df = pd.DataFrame.from_records(records)
    return gpd.GeoDataFrame(df, geometry="geometry", crs=tiles_tagged.crs)

def load_country_shapes(crs: str = "EPSG:3857",
                        use_bbox: bool = False,
                        countries: list = None) -> gpd.GeoDataFrame:
    if countries is None:
        countries = [
            "Brazil", "Bolivia", "Colombia", "Ecuador", "Guyana",
            "Peru", "Suriname", "Venezuela", "French Guiana"
        ]
    try:
        from geodatasets import get_path
        world = gpd.read_file(get_path("naturalearth_lowres"))
    except:
        url = ("https://naciscdn.org/naturalearth/110m/cultural/"
               "ne_110m_admin_0_countries.zip")
        world = gpd.read_file(url)
    name_col = next(c for c in ("NAME", "ADMIN", "NAME_EN") if c in world.columns)
    world = world.rename(columns={name_col: "name"})
    mask = world["name"].str.lower().isin([c.lower() for c in countries])
    world = world[mask].to_crs(crs)
    if use_bbox:
        world["geometry"] = world.geometry.bounds.apply(lambda r: box(*r), axis=1)
    return world[["name", "geometry"]]

def load_main_rivers(RAW_DIR, AOI_BOX, METRIC_CRS):
    rivers_path = RAW_DIR / "datasets/hydrorivers-dataset/HydroRIVERS.gdb"
    rivers = gpd.read_file(rivers_path, bbox=AOI_BOX)
    rivers = rivers[rivers["ORD_FLOW"].astype(float) <= 6]
    return rivers.to_crs(METRIC_CRS)


def point_to_rowcol(pt, src):
    col, row = src.index(pt.x, pt.y)
    return int(row), int(col)

def tri_at(idx, dem_fp, WIN_SIZE):
    r, c = idx
    half = WIN_SIZE // 2
    with rasterio.open(dem_fp) as src:
        arr = src.read(
            1,
            window=Window(c-half, r-half, WIN_SIZE, WIN_SIZE),
            boundless=True,
            fill_value=np.nan
        ).astype(np.float32)
    center = arr[half, half]
    return np.nanmean(np.abs(arr - center))

def twi_at(idx, dem_fp, WIN_SIZE):
    r, c = idx
    half = WIN_SIZE // 2
    with rasterio.open(dem_fp) as src:
        arr = src.read(
            1,
            window=Window(c-half, r-half, WIN_SIZE, WIN_SIZE),
            boundless=True,
            fill_value=np.nan
        ).astype(np.float32)
        res_x, res_y = src.res
    dzdx = (arr[half, half+1] - arr[half, half-1]) / (2*res_x)
    dzdy = (arr[half+1, half] - arr[half-1, half]) / (2*res_y)
    slope_rad = np.arctan(np.hypot(dzdx, dzdy))
    As = np.count_nonzero(~np.isnan(arr)) * (res_x * res_y)
    return np.log1p(As / (np.tan(slope_rad) + 1e-6))

def curv_at(idx, dem_fp, WIN_SIZE):
    r, c = idx
    half = WIN_SIZE // 2
    with rasterio.open(dem_fp) as src:
        arr = src.read(
            1,
            window=Window(c-half, r-half, WIN_SIZE, WIN_SIZE),
            boundless=True,
            fill_value=np.nan
        ).astype(np.float32)
    flat = arr.flatten()
    # planform curvature approximate
    plan = np.nanmean(flat) - np.nanmean(flat[[0,2,6,8]])
    # profile curvature approximate
    prof = np.nanmean(np.abs(flat - np.nanmean(flat)))
    return float(plan), float(prof)

# ----------------------------------------------------------------------
# reg processing
# ----------------------------------------------------------------------
def process_region(region_name, tiles_gdf, RAW_DIR, WIN_SIZE, N_JOBS=-1):
    dem_fp = RAW_DIR / f"{region_name}_srtmgl3_90m.tif"
    if not dem_fp.exists():
        raise FileNotFoundError(f"DEM for {region_name} not found: {dem_fp}")
    with rasterio.open(dem_fp) as src:
        pts = tiles_gdf.to_crs(src.crs).geometry.centroid
        idxs = [point_to_rowcol(pt, src) for pt in pts]
    tris  = Parallel(n_jobs=N_JOBS)(delayed(tri_at)(idx, dem_fp, WIN_SIZE) for idx in idxs)
    twis  = Parallel(n_jobs=N_JOBS)(delayed(twi_at)(idx, dem_fp, WIN_SIZE) for idx in idxs)
    curvs = Parallel(n_jobs=N_JOBS)(delayed(curv_at)(idx, dem_fp, WIN_SIZE) for idx in idxs)
    plan, prof = zip(*curvs)
    out = tiles_gdf.copy()
    out['tri']       = tris
    out['twi']       = twis
    out['curv_plan'] = plan
    out['curv_prof'] = prof
    return region_name, out


def get_rivers_and_mountains():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    RAW_DIR = Path(config["raw_data_dir"])
    PROCESSED_DIR = Path(config["processed_data_dir"])
    AOI_CRS = config["aoi_crs"]
    METRIC_CRS = config["metric_crs"]
    AOI_BOX = tuple(config["aoi_box"])
    DEM_TYPE = "SRTMGL3"
    OPENTOPO_KEY = os.getenv("OPENTOPO_KEY")
    if OPENTOPO_KEY is None:
        raise EnvironmentError("❌ Missing OPENTOPO_KEY in your .env file")

    N_JOBS = -1
    WIN_SIZE = 3  # <-- define globally!

    # 1. Load all tagged tile files
    regions_gdf = {}
    for i in range(1, 7): #7
        fp = PROCESSED_DIR / f"region_{i}_tagged_with_site.gpkg"
        if not fp.exists():
            raise FileNotFoundError(f"Missing: {fp}")
        gdf = gpd.read_file(fp).to_crs(AOI_CRS)
        #gdf = gdf.sample(frac=0.03, random_state=42).reset_index(drop=True)
        print(len(gdf))
        regions_gdf[f"region_{i}"] = gdf
    # 2. Load main rivers
    main_rivers_web = load_main_rivers(RAW_DIR, AOI_BOX, METRIC_CRS)
    # 3. Relief features
    relief_regions = {}
    for region_name, gdf in regions_gdf.items():
        gdf_geo = gdf.to_crs(AOI_CRS)
        minx, miny, maxx, maxy = gdf_geo.total_bounds
        aoi_bbox = [minx, miny, maxx, maxy]
        dem_file = f"{region_name.lower()}_{DEM_TYPE.lower()}_90m.tif"
        dem_path = load_dem_data(
            dem_type   = DEM_TYPE,
            aoi        = aoi_bbox,
            out_file   = dem_file,
            interim_dir = "data/raw"
        )
        gdf_relief = add_relief_features(gdf, dem_path)
        relief_regions[region_name] = gdf_relief
    # 4. Mountains
    regions_final = {}
    for name, gdf in relief_regions.items():
        if "geometry_point" not in gdf.columns:
            gdf = add_geometry_point(gdf, method="centroid")
        gdf = flag_mountains(
            gdf,
            slope_field="mean_slope_deg",
            slope_thr=20,
            elev_thr=300,
            out_col="is_mountain"
        )
        gdf = add_distance_to_mountains(
            gdf,
            mountain_col="is_mountain",
            point_col="geometry_point",
            out_col="dist_to_mountain_m"
        )
        regions_final[name] = gdf
    #5. Distance to rivers
    regions_with_dist = {}
    for name, gdf in regions_final.items():
        gdf_dist = get_distance_to_rivers_parallel(
            geo_gdf= gdf,
            riv_gdf= main_rivers_web,
            geometry_col= "geometry",
            n_jobs= -1
        )
        regions_with_dist[name] = gdf_dist
    # regions_with_dist = {}

    # for name, gdf in regions_final.items():
    #     gdf_dist = get_distance_to_rivers_fast(
    #         geo_gdf=gdf,
    #         riv_gdf=main_rivers_web,
    #         geometry_col="geometry"
    #     )
    #     regions_with_dist[name] = gdf_dist
    # 6. River features
    regions_with_river_features = {}
    for name, gdf in regions_with_dist.items():
        if "geometry_point" not in gdf.columns:
            gdf = add_geometry_point(gdf, method="centroid")

        gdf_riv = compute_drainage_density(
            tiles_gdf=gdf,
            rivers_gdf=main_rivers_web,
            tile_geom_col="geometry",
            river_geom_col="geometry",
            metric_crs=METRIC_CRS,
            source_crs=AOI_CRS
        )
        gdf_riv = add_river_attrs_sjoin(tiles=gdf_riv,
                                        rivers=main_rivers_web,
                                        tile_pt_col="geometry_point",
                                        river_geom_col="geometry",
                                        river_rank_col="ORD_FLOW",
                                        river_area_col="UPLAND_SKM",
                                        metric_crs=METRIC_CRS)
        
        regions_with_river_features[name] = gdf_riv
    # 7. Update geometry_point (CRS)
    regions_updated = {}
    for name, gdf in regions_with_river_features.items():
        gdf = gdf.to_crs(METRIC_CRS)
        gdf = add_geometry_point(gdf, method="centroid")
        regions_updated[name] = gdf

    # ----------------------------------------------------------------------
    # MAIN: run with bar
    # ----------------------------------------------------------------------
    tasks = list(regions_updated.items())  # [(region_name, gdf), ...]

    with tqdm_joblib(tqdm(total=len(tasks), desc="Regions")) as progress:
        results = Parallel(n_jobs=min(N_JOBS, len(tasks)), verbose=0)(
            delayed(process_region)(name, gdf, RAW_DIR, WIN_SIZE, N_JOBS) for name, gdf in tasks
        )

    regions_with_terrain = dict(results)

    # ----------------------------------------------------------------------
    # validation
    # ----------------------------------------------------------------------
    for reg, gdf in regions_with_terrain.items():
        print(f"{reg}: TRI mean={gdf['tri'].mean():.2f}, TWI mean={gdf['twi'].mean():.2f}")
        # 8. Flatten

    regions_updated = {}
    for name, gdf in regions_with_terrain.items():
        gdf = gdf.to_crs(METRIC_CRS)
        gdf = add_geometry_point(gdf, method="centroid")
        regions_updated[name] = gdf

    flattened = {}
    for region, tiles in regions_updated.items():
        flat = flatten_tile_hits_with_negatives(tiles)
        flat = flat.drop(columns=["index_right", "name"], errors="ignore")
        flat["region"] = region
        flattened[region] = flat
    # 9. Load and normalize country polygons
    countries_gdf = load_country_shapes(crs=METRIC_CRS, use_bbox=False)
    if "ADMIN" in countries_gdf.columns:
        countries_gdf = countries_gdf.rename(columns={"ADMIN": "name"})
    elif "NAME_EN" in countries_gdf.columns:
        countries_gdf = countries_gdf.rename(columns={"NAME_EN": "name"})
    countries_gdf = countries_gdf[["name", "geometry"]]
    # 10. Assign country by spatial join on geometry_point
    regions_flat_with_country = {}
    for region, flat in flattened.items():
        flat = flat.set_geometry("geometry_point")
        if flat.crs is None:
            flat = flat.set_crs(METRIC_CRS, allow_override=True)
        else:
            flat = flat.to_crs(METRIC_CRS)
        joined = gpd.sjoin(flat, countries_gdf, how="left", predicate="within")
        out = flat.copy()
        out["country"] = joined["name"]
        regions_flat_with_country[region] = out
    # 11. Combine all regions into final GeoDataFrame

    all_tiles_flat = pd.concat(
        regions_flat_with_country.values(), ignore_index=True
    )
    all_tiles_flat = gpd.GeoDataFrame(
        all_tiles_flat,
        geometry="geometry",
        crs=METRIC_CRS
    )

    with open(PROCESSED_DIR / "all_tiles_flat.pkl", "wb") as f:
        pickle.dump(all_tiles_flat, f)
    print(f"✅ Saved to {PROCESSED_DIR / 'all_tiles_flat.pkl'}")

    return all_tiles_flat

if __name__ == "__main__":
    all_tiles_flat = get_rivers_and_mountains()
