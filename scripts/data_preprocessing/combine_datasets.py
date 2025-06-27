# #!/usr/bin/env python3
# """
# Script to load various CSV sources from raw_datasets/, convert and merge coordinates,
# return a pandas DataFrame and save combined records with bboxes to data/processed/.
# """
# import os
# import math
# import logging
# import json
# import pandas as pd
# import numpy as np
# from pyproj import Transformer
# from typing import List, Dict, Any

# # ------------------------------------------------------------
# # Configuration
# # ------------------------------------------------------------
# logging.basicConfig(
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     level=logging.INFO
# )
# log = logging.getLogger(__name__)

# # ------------------------------------------------------------
# # Coordinate transformation: UTM to WGS84
# # ------------------------------------------------------------
# def utm_to_latlon(x: float, y: float, zone: int, hemisphere: str = "south") -> (float, float):
#     epsg_code = 32600 + zone if hemisphere.lower() == "north" else 32700 + zone
#     transformer = Transformer.from_crs(f"EPSG:{epsg_code}", "EPSG:4326", always_xy=True)
#     lon, lat = transformer.transform(x, y)
#     return lat, lon

# # ------------------------------------------------------------
# # Readers for different CSV formats
# # ------------------------------------------------------------
# def read_mound_villages(path: str) -> pd.DataFrame:
#     df = pd.read_csv(path)
#     coords = []
#     for _, r in df.iterrows():
#         if pd.notna(r.get("UTM X (Easting)")) and pd.notna(r.get("UTM Y (Northing)")):
#             coords.append(utm_to_latlon(r["UTM X (Easting)"], r["UTM Y (Northing)"], 19))
#         else:
#             coords.append((np.nan, np.nan))
#     df[["latitude", "longitude"]] = pd.DataFrame(coords, index=df.index)
#     df["name"] = df["Site Name"].fillna("undefined").astype(str)
#     df["source"] = "Mound Villages"
#     return df[["name", "latitude", "longitude", "source"]]


# def read_casarabe(path: str) -> pd.DataFrame:
#     df = pd.read_csv(path)
#     coords = []
#     for _, r in df.iterrows():
#         if pd.notna(r.get("UTM X (Easting)")) and pd.notna(r.get("UTM Y (Northing)")):
#             coords.append(utm_to_latlon(r["UTM X (Easting)"], r["UTM Y (Northing)"], 20))
#         else:
#             coords.append((np.nan, np.nan))
#     df[["latitude", "longitude"]] = pd.DataFrame(coords, index=df.index)
#     df["name"] = df["Site Name"].fillna("undefined").astype(str)
#     df["source"] = "Casarabe Sites"
#     return df[["name", "latitude", "longitude", "source"]]


# def read_geoglyphs(path: str) -> pd.DataFrame:
#     df = pd.read_csv(path)
#     df["latitude"] = pd.to_numeric(df.get("latitude"), errors="coerce")
#     df["longitude"] = pd.to_numeric(df.get("longitude"), errors="coerce")
#     df["name"] = df.get("name").fillna("undefined").astype(str)
#     df["source"] = "Amazon Geoglyphs"
#     return df[["name", "latitude", "longitude", "source"]]


# def read_submit(path: str) -> pd.DataFrame:
#     df = pd.read_csv(path)
#     if -180 < df["x"].iloc[0] < 180 and -90 < df["y"].iloc[0] < 90:
#         df["latitude"], df["longitude"] = df["y"], df["x"]
#     else:
#         df["latitude"], df["longitude"] = np.nan, np.nan
#         for i, r in df.iterrows():
#             for zone in [18, 19, 20, 21]:
#                 lat, lon = utm_to_latlon(r["x"], r["y"], zone)
#                 if -90 < lat < 90 and -180 < lon < 180:
#                     df.at[i, "latitude"], df.at[i, "longitude"] = lat, lon
#                     break
#     df["name"] = (df.get("type", pd.Series()).fillna("submit").str.replace(r"\s+", "_", regex=True)
#                    + "_" + df["latitude"].round(2).astype(str)
#                    + "_" + df["longitude"].round(2).astype(str))
#     df["source"] = "Archaeological Survey"
#     return df[["name", "latitude", "longitude", "source"]]


# def read_science(path: str) -> pd.DataFrame:
#     df = pd.read_csv(path)
#     df["latitude"] = pd.to_numeric(df.get("Latitude"), errors="coerce")
#     df["longitude"] = pd.to_numeric(df.get("Longitude"), errors="coerce")
#     df["name"] = df.get("Site", df.get("Site Name")).fillna("undefined").astype(str)
#     df["source"] = "Science Data"
#     return df[["name", "latitude", "longitude", "source"]]

# # ------------------------------------------------------------
# # Main processing
# # ------------------------------------------------------------
# def process_datasets() -> pd.DataFrame:
#     raw_dir = os.path.join(os.getcwd(), "data/raw/datasets")
#     processed_dir = os.path.join(os.getcwd(), "data/processed")

#     dfs = [
#         read_mound_villages(os.path.join(raw_dir, "mound_villages_acre.csv")),
#         read_casarabe(os.path.join(raw_dir, "casarabe_sites_utm.csv")),
#         read_geoglyphs(os.path.join(raw_dir, "amazon_geoglyphs_sites.csv")),
#         read_submit(os.path.join(raw_dir, "submit.csv")),
#         read_science(os.path.join(raw_dir, "science.ade2541_data_s2.csv"))
#     ]
#     df_all = pd.concat(dfs, ignore_index=True).dropna(subset=["latitude", "longitude"])
#     log.info(f"Всего точек из CSV: {len(df_all)}")

#     # entries for CSV
#     entries = [{"name": r["name"], "coordinates": [r["longitude"], r["latitude"]], "source": r["source"]}
#                for _, r in df_all.iterrows()]

#     # bboxes
#     def point_to_bbox(lon, lat, side=500):
#         hlat = side / 111320.0
#         hlon = side / (111320.0 * math.cos(math.radians(lat)))
#         return [lon-hlon, lat-hlat, lon+hlon, lat+hlat]
#     records = [{**e, "bbox": point_to_bbox(*e["coordinates"])} for e in entries]
#     log.info(f"BBox для CSV: {len(records)} точек")

#     # prepared JSON
#     prep_path = os.path.join(raw_dir, "coords_prepared.json")
#     with open(prep_path, 'r', encoding='utf-8') as f:
#         prep = json.load(f)
#     def center(b): return [(b[0]+b[2])/2, (b[1]+b[3])/2]
#     prep_entries = [{"name": it.get("name"), "coordinates": center(it.get("bbox", [])), "bbox": it.get("bbox", [])}
#                     for it in prep]
#     log.info(f"Записей из JSON: {len(prep_entries)}")

#     combined = records + prep_entries
#     df_out = pd.DataFrame(combined)
#     out_csv = os.path.join(processed_dir, "all_coords.csv")
#     df_out.to_csv(out_csv, index=False)
#     log.info(f"Сохранено всего: {len(df_out)} записей в {out_csv}")

#     return df_out

# if __name__ == "__main__":
#     process_datasets()


#!/usr/bin/env python3
"""
Script to load various CSV sources from raw_datasets/, convert and merge coordinates,
return a pandas DataFrame and save combined records with bboxes to data/processed/.
"""
import os
import math
import logging
import json
import pandas as pd
import numpy as np
import yaml
import sys
from pathlib import Path
from pyproj import Transformer
from typing import List, Dict, Any

# ------------------------------------------------------------
# Configuration & logging
# ------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)
log = logging.getLogger(__name__)

# ------------------------------------------------------------
# Metadata-logger import
# ------------------------------------------------------------
base = Path.cwd()
cfg = yaml.safe_load((base / 'config.yaml').read_text())
scripts_dir = cfg.get('scripts_dir', 'scripts')
sys.path.insert(0, str(base / scripts_dir / 'general_utils'))
from utils import print_dataset_info

# ------------------------------------------------------------
# Coordinate transformation: UTM to WGS84
# ------------------------------------------------------------

def utm_to_latlon(x: float, y: float, zone: int, hemisphere: str = "south") -> (float, float):
    epsg_code = 32600 + zone if hemisphere.lower() == "north" else 32700 + zone
    transformer = Transformer.from_crs(f"EPSG:{epsg_code}", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(x, y)
    return lat, lon

# ------------------------------------------------------------
# Readers for different CSV formats
# ------------------------------------------------------------

def read_mound_villages(path: str) -> pd.DataFrame:
    print_dataset_info('mound_villages')
    df = pd.read_csv(path)
    coords = [
        utm_to_latlon(r['UTM X (Easting)'], r['UTM Y (Northing)'], 19)
        if pd.notna(r.get('UTM X (Easting)')) and pd.notna(r.get('UTM Y (Northing)'))
        else (np.nan, np.nan)
        for _, r in df.iterrows()
    ]
    df[['latitude','longitude']] = pd.DataFrame(coords, index=df.index)
    df['name'] = df['Site Name'].fillna('undefined').astype(str)
    df['source'] = 'Mound Villages'
    return df[['name','latitude','longitude','source']]


def read_casarabe(path: str) -> pd.DataFrame:
    print_dataset_info('casarabe_sites')
    df = pd.read_csv(path)
    coords = [
        utm_to_latlon(r['UTM X (Easting)'], r['UTM Y (Northing)'], 20)
        if pd.notna(r.get('UTM X (Easting)')) and pd.notna(r.get('UTM Y (Northing)'))
        else (np.nan, np.nan)
        for _, r in df.iterrows()
    ]
    df[['latitude','longitude']] = pd.DataFrame(coords, index=df.index)
    df['name'] = df['Site Name'].fillna('undefined').astype(str)
    df['source'] = 'Casarabe Sites'
    return df[['name','latitude','longitude','source']]


def read_geoglyphs(path: str) -> pd.DataFrame:
    print_dataset_info('geoglyphs')
    df = pd.read_csv(path)
    df['latitude'] = pd.to_numeric(df.get('latitude'), errors='coerce')
    df['longitude'] = pd.to_numeric(df.get('longitude'), errors='coerce')
    df['name'] = df.get('name').fillna('undefined').astype(str)
    df['source'] = 'Amazon Geoglyphs'
    return df[['name','latitude','longitude','source']]


def read_submit(path: str) -> pd.DataFrame:
    print_dataset_info('submit')
    df = pd.read_csv(path)
    if -180 < df['x'].iloc[0] < 180 and -90 < df['y'].iloc[0] < 90:
        df['latitude'], df['longitude'] = df['y'], df['x']
    else:
        df['latitude'], df['longitude'] = np.nan, np.nan
        for i, r in df.iterrows():
            for zone in [18,19,20,21]:
                lat, lon = utm_to_latlon(r['x'], r['y'], zone)
                if -90 < lat < 90 and -180 < lon < 180:
                    df.at[i,'latitude'], df.at[i,'longitude'] = lat, lon
                    break
    df['name'] = (
        df.get('type', pd.Series()).fillna('submit').str.replace(r'\s+','_', regex=True)
        + '_' + df['latitude'].round(2).astype(str)
        + '_' + df['longitude'].round(2).astype(str)
    )
    df['source'] = 'Archaeological Survey'
    return df[['name','latitude','longitude','source']]


def read_science(path: str) -> pd.DataFrame:
    print_dataset_info('science_data')
    df = pd.read_csv(path)
    df['latitude'] = pd.to_numeric(df.get('Latitude'), errors='coerce')
    df['longitude'] = pd.to_numeric(df.get('Longitude'), errors='coerce')
    df['name'] = df.get('Site', df.get('Site Name')).fillna('undefined').astype(str)
    df['source'] = 'Science Data'
    return df[['name','latitude','longitude','source']]

# ------------------------------------------------------------
# Main processing
# ------------------------------------------------------------

def process_datasets() -> pd.DataFrame:
    raw_dir = os.path.join(os.getcwd(), 'data/raw/datasets')
    processed_dir = os.path.join(os.getcwd(), 'data/processed')

    dfs = [
        read_mound_villages(os.path.join(raw_dir,'mound_villages_acre.csv')),
        read_casarabe(os.path.join(raw_dir,'casarabe_sites_utm.csv')),
        read_geoglyphs(os.path.join(raw_dir,'amazon_geoglyphs_sites.csv')),
        read_submit(os.path.join(raw_dir,'submit.csv')),
        read_science(os.path.join(raw_dir,'science.ade2541_data_s2.csv'))
    ]
    df_all = pd.concat(dfs, ignore_index=True).dropna(subset=['latitude','longitude'])
    log.info(f"number of points in CSV: {len(df_all)}")

    # entries
    entries = [
        {'name':r['name'],'coordinates':[r['longitude'],r['latitude']],'source':r['source']}
        for _,r in df_all.iterrows()
    ]

    # bboxes
    def point_to_bbox(lon, lat, side=500):
        hl = side/111320.0
        hw = side/(111320.0*math.cos(math.radians(lat)))
        return [lon-hw, lat-hl, lon+hw, lat+hl]
    records = [{**e,'bbox':point_to_bbox(*e['coordinates'])} for e in entries]
    log.info(f"bbox for CSV: {len(records)} points")

    # prepared JSON
    print_dataset_info('coords_prepared')
    prep_path = os.path.join(raw_dir,'coords_prepared.json')
    with open(prep_path,'r',encoding='utf-8') as f:
        prep = json.load(f)
    def center(b): return [(b[0]+b[2])/2,(b[1]+b[3])/2]
    prep_entries = [
        {'name':item.get('name'),'coordinates':center(item.get('bbox',[])),'bbox':item.get('bbox',[]),'source':'coords_prepared'}
        for item in prep
    ]
    log.info(f"rows JSON: {len(prep_entries)}")

    combined = records + prep_entries
    df_out = pd.DataFrame(combined)
    out_csv = os.path.join(processed_dir,'all_coords.csv')
    df_out.to_csv(out_csv,index=False)
    log.info(f"saved: {len(df_out)} rows in {out_csv}")

    return df_out

if __name__=='__main__':
    process_datasets()

