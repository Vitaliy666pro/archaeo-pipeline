import os
import yaml
import pickle
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation

# Optional — if joblib is available, we'll additionally dump a binary model
try:
    import joblib
except ImportError:
    joblib = None

############################################################
# 1. Utility helpers                                        #
############################################################

def load_config(path: str = "config.yaml") -> dict:
    """Read a YAML config and return as dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_base_tiles(processed_dir: str) -> pd.DataFrame:
    """Load pickled tiles **without road‑mask** and filter elevation ≤ 350 m."""
    pkl = os.path.join(processed_dir, "all_tiles_without_roads.pkl")
    with open(pkl, "rb") as f:
        df_base: pd.DataFrame = pickle.load(f).reset_index(drop=True)
    return df_base[df_base["mean_elev_m"] <= 350].reset_index(drop=True)


def load_lidar_inventory(lidar_csv_path: str, crs: str = "EPSG:3857") -> gpd.GeoDataFrame:
    """Convert LiDAR tile inventory CSV → GeoDataFrame of bounding boxes."""
    li = pd.read_csv(lidar_csv_path)
    li_gdf = gpd.GeoDataFrame(
        li,
        geometry=li.apply(lambda r: box(r.min_lon, r.min_lat, r.max_lon, r.max_lat), axis=1),
        crs="EPSG:4326",
    ).to_crs(crs)
    return li_gdf


def mark_lidar_coverage(df_base: pd.DataFrame, li_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Add boolean column `lidar_cover` → tile intersects ANY LiDAR footprint."""
    gdf_base = gpd.GeoDataFrame(df_base, geometry="geometry", crs="EPSG:3857")
    lidar_union = li_gdf.geometry.union_all()
    df_base["lidar_cover"] = gdf_base.geometry.intersects(lidar_union)
    return df_base


def encode_country_region(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df, columns=["country", "region"], prefix=["country", "region"])


def compute_spatial_groups(df_enc: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    gdf = gpd.GeoDataFrame(df_enc, geometry="geometry", crs="EPSG:3857")
    centroids = gdf.geometry.centroid.to_crs("EPSG:4326")
    df_enc["lon"], df_enc["lat"] = centroids.x, centroids.y
    df_enc["lon_bin"] = pd.qcut(df_enc["lon"], q=n_bins, labels=False, duplicates="drop")
    df_enc["lat_bin"] = pd.qcut(df_enc["lat"], q=n_bins, labels=False, duplicates="drop")
    df_enc["spatial_group"] = df_enc["lon_bin"] * n_bins + df_enc["lat_bin"]
    return df_enc


def downsample(df: pd.DataFrame, neg_ratio: int = 2, seed: int = 42) -> pd.DataFrame:
    pos = df[df["has_geoglyph"] == 1]
    neg = df[df["has_geoglyph"] == 0].sample(len(pos) * neg_ratio, random_state=seed)
    return pd.concat([pos, neg]).sample(frac=1, random_state=seed)


def get_final_predictions(gdf_all: gpd.GeoDataFrame, li_gdf: gpd.GeoDataFrame, threshold: float) -> gpd.GeoDataFrame:
    """Intersect high‑probability tiles with LiDAR footprints to obtain final list."""
    tiles_pred = gpd.GeoDataFrame(
        gdf_all[gdf_all["pred_prob"] > threshold].copy(), geometry="geometry", crs="EPSG:3857"
    )

    joined = gpd.sjoin(
        tiles_pred,
        li_gdf[["filename", "geometry", "max_lat", "min_lat", "max_lon", "min_lon"]],
        how="inner",
        predicate="intersects",
    )

    result = (
        joined[
            [
                "tile_id",
                "pred_prob",
                "n_geoglyphs",
                "filename",
                "geometry",
                "max_lat",
                "min_lat",
                "max_lon",
                "min_lon",
            ]
        ]
        .drop_duplicates(subset=["tile_id", "filename"])
        .reset_index(drop=True)
    )

    # Add centroid lon/lat in WGS84
    result_wgs = result.to_crs("EPSG:4326")
    result_wgs["centroid"] = result_wgs.geometry.centroid
    result_wgs["lon"] = result_wgs.centroid.x
    result_wgs["lat"] = result_wgs.centroid.y
    return result_wgs.drop(columns=["centroid"])


def get_top_candidates(
    gdf_all: gpd.GeoDataFrame,
    top_n: int = 500,
    exclude_splits: tuple = ('train', 'val')
) -> gpd.GeoDataFrame:
    """
    Select the top-N candidate tiles by predicted probability,
    excluding any rows whose 'split' is in exclude_splits.

    Returns a GeoDataFrame in EPSG:4326 with columns:
      - tile_id
      - split
      - n_geoglyphs
      - pred_prob
      - lon, lat (centroid of each tile)
    """
    # 1) filter out unwanted splits
    df = gdf_all[~gdf_all['split'].isin(exclude_splits)].copy()

    # 2) sort by descending prediction probability and take top_n
    df = df.sort_values('pred_prob', ascending=False).head(top_n)

    # 3) ensure geometry column is present and in Web Mercator
    df = gpd.GeoDataFrame(df, geometry='geometry', crs=df.crs or "EPSG:3857")
    if df.crs.to_epsg() != 3857:
        df = df.to_crs("EPSG:3857")

    # 4) compute centroids and reproject to WGS84
    df['centroid'] = df.geometry.centroid
    df_wgs = df.to_crs("EPSG:4326")
    df_wgs['lon'] = df_wgs.centroid.x
    df_wgs['lat'] = df_wgs.centroid.y

    # 5) select and return the desired columns
    return df_wgs[[
        'tile_id',
        'split',
        'n_geoglyphs',
        'pred_prob',
        'lon',
        'lat'
    ]]

############################################################
# 2. Main pipeline                                         #
############################################################

def run_model(config_path: str = "config.yaml", threshold: float = 0.5):
    """Train LightGBM, export model + top‑500 candidates with LiDAR coverage."""

    # ── Reproducibility ────────────────────────────────────────────────────────
    np.random.seed(42)
    import random as _rnd

    _rnd.seed(42)
    os.environ["PYTHONHASHSEED"] = "42"

    # ── Load config & derive IO paths ─────────────────────────────────────────
    cfg = load_config(config_path)
    PROCESSED_DIR: str = cfg["processed_data_dir"]
    RAW_DIR: str = cfg["raw_data_dir"]
    RESULTS_DIR: str = cfg.get("results_dir", "results")  # fallback if key missing

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Data ingestion ────────────────────────────────────────────────────────
    lidar_csv = os.path.join(RAW_DIR, "datasets/cms_brazil_lidar_tile_inventory.csv")
    df_base = load_base_tiles(PROCESSED_DIR)
    li_gdf = load_lidar_inventory(lidar_csv)
    df_base = mark_lidar_coverage(df_base, li_gdf)

    df_enc = encode_country_region(df_base)
    df_enc = compute_spatial_groups(df_enc, n_bins=5)

    # ── Feature selection ─────────────────────────────────────────────────────
    base_num = [
        "drainage_density",
        "dist_to_mountain_m",
        "dist_to_river_m",
        "mean_elev_m",
        "mean_slope_deg",
        "ord_flow",
        "upland_skm",
        "drainage_density_m",
        "tri",
        "twi",
        "curv_plan",
        "curv_prof",
        "clay_0_5cm",
        "ph_h2o_0_5cm",
        "soc_0_5cm",
        "PC1",
        "PC2",
        "PC3",
        "PC4",
        "PC5",
        "PC6",
        "PC7",
    ]
    FEATURES = [c for c in base_num if c in df_enc.columns]

    # ── Train / val / test split ─────────────────────────────────────────────
    test_df = df_enc[df_enc["lidar_cover"]].reset_index(drop=True)
    rem_df = df_enc[~df_enc["lidar_cover"]].reset_index(drop=True)

    rem_train, rem_val = train_test_split(
        rem_df, test_size=0.2, stratify=rem_df["has_geoglyph"], random_state=42
    )

    train_df = downsample(rem_train, neg_ratio=2)
    val_df = downsample(rem_val, neg_ratio=2)

    train_df["split"], val_df["split"], test_df["split"] = "train", "val", "test"

    X_train, y_train, groups = (
        train_df[FEATURES],
        train_df["has_geoglyph"],
        train_df["spatial_group"],
    )
    X_val, y_val = val_df[FEATURES], val_df["has_geoglyph"]
    X_test, y_test = test_df[FEATURES], test_df["has_geoglyph"]

    # ── LightGBM config ───────────────────────────────────────────────────────
    lgb_params = dict(
        n_estimators=150,
        is_unbalance=True,
        random_state=42,
        reg_alpha=1.0,
        reg_lambda=1.0,
        min_gain_to_split=0.0,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        verbosity=-1,
    )

    # ── OOF GroupKFold to gauge stability ────────────────────────────────────
    oof = np.zeros(len(train_df))
    gkf = GroupKFold(n_splits=5)
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X_train, y_train, groups), 1):
        clf = lgb.LGBMClassifier(**lgb_params)
        clf.fit(
            X_train.iloc[tr_idx],
            y_train.iloc[tr_idx],
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[early_stopping(30), log_evaluation(0)],
        )
        oof[te_idx] = clf.predict_proba(X_train.iloc[te_idx])[:, 1]

    # ── Train final model on full train set ───────────────────────────────────
    final_clf = lgb.LGBMClassifier(**lgb_params)
    final_clf.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[early_stopping(30), log_evaluation(10)],
    )

    # ── Predict ───────────────────────────────────────────────────────────────
    val_df["pred_prob"] = final_clf.predict_proba(X_val)[:, 1]
    test_df["pred_prob"] = final_clf.predict_proba(X_test)[:, 1]
    train_df["pred_prob"] = oof

    df_all = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # ── Intersect with LiDAR & keep TOP‑500 ───────────────────────────────────
    gdf_all = gpd.GeoDataFrame(df_all, geometry="geometry", crs="EPSG:3857")
    
    all_candidates = get_final_predictions(gdf_all, li_gdf, threshold)
    short_list = (
        all_candidates.sort_values("pred_prob", ascending=False).head(100).reset_index(drop=True)
    )

    ########################################################
    # 3. Export artefacts                                   #
    ########################################################
    # 3.1 Model weights
    model_txt = os.path.join(RESULTS_DIR, "lightgbm_model.txt")
    final_clf.booster_.save_model(model_txt)

    if joblib is not None:
        model_pkl = os.path.join(RESULTS_DIR, "lightgbm_model.pkl")
        joblib.dump(final_clf, model_pkl, compress=3)

    # 3.2 Predictions
    out_csv_short = os.path.join(RESULTS_DIR, "short_list.csv")
    short_list.to_csv(out_csv_short, index=False)

    top500 = get_top_candidates(gdf_all, top_n=500)
    out_csv = os.path.join(RESULTS_DIR, "candidates_top500.csv")
    final.to_csv(out_csv, index=False)

    print("Saved short list:", out_csv_short)
    print("Saved model →", model_txt)
    print("Saved candidates →", out_csv)


#######################################################################
if __name__ == "__main__":
    run_model()
