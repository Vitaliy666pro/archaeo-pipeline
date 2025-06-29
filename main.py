"""
Main entry point to run the full archaeopipeline end-to-end.
"""
import logging

from scripts.data_preprocessing.unpack_data import unpack_archives
from scripts.data_preprocessing.combine_datasets import process_datasets
from scripts.data_preprocessing.download_external_datasets import download_all_kaggle_datasets

from scripts.feature_engine.get_tiles_with_regions_and_sites import get_tiles_with_reg_and_sites
from scripts.feature_engine.get_rivers_and_mountains import get_rivers_and_mountains
from scripts.feature_engine.get_soil_features import add_soil_features_to_tiles
from scripts.feature_engine.get_emb_pca import get_emb_pca
from scripts.feature_engine.cut_roads import cut_roads

from scripts.model.train_model import run_model

from scripts.verification.download_predicted_s1_s2 import retrieve_tiles


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("🚀 Starting full pipeline")

    logging.info("1) Unpacking raw data archives…")
    unpack_archives()

    logging.info("2) Combining and cleaning datasets…")
    combined_df = process_datasets()

    logging.info("3) Downloading external Kaggle datasets…")
    download_all_kaggle_datasets()

    logging.info("4) Generating tiles with regions and sites…")
    get_tiles_with_reg_and_sites()

    logging.info("5) Extracting rivers and mountain features…")
    get_rivers_and_mountains()

    logging.info("6) Adding soil features to tiles…")
    add_soil_features_to_tiles()

    logging.info("7) Attaching embeddings + PCA reduction…")
    gdf_emb = get_emb_pca()

    logging.info("8) Cutting out roads from tiles…")
    cut_roads()

    logging.info("9) Training the model…")
    run_model()

    logging.info("10) Downloading candidate S1/S2 predictions…")
    retrieve_tiles()

    logging.info("✅ Pipeline complete!")


if __name__ == "__main__":
    main()
