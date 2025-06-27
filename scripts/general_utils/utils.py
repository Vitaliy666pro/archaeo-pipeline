# scripts/general_utils/utils.py
"""
Utility functions to fetch and print dataset metadata before loading.
"""
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Load metadata from project root.
# Assumes datasets_metadata.json лежит в корне проекта.
META_PATH = Path(__file__).resolve().parents[2] / 'datasets_metadata.json'
try:
    with open(META_PATH, 'r', encoding='utf-8') as f:
        DATASETS_META = json.load(f)
except FileNotFoundError:
    log.error(f"Dataset metadata file not found at {META_PATH}")
    DATASETS_META = {}
except json.JSONDecodeError as e:
    log.error(f"Error parsing metadata JSON: {e}")
    DATASETS_META = {}

def get_dataset_meta(name: str) -> dict:
    """
    Retrieve metadata dictionary for a given dataset key.
    Returns an empty dict if no metadata is found.
    """
    return DATASETS_META.get(name, {})

def print_dataset_info(name: str):
    """
    Log and print metadata information for the given dataset key.
    """
    meta = get_dataset_meta(name)
    if not meta:
        log.warning(f"No metadata found for dataset '{name}'")
        print(f"Warning: No metadata found for dataset '{name}'")
        return

    file = meta.get('file', 'unknown')
    desc = meta.get('description', '')
    citation = meta.get('citation', '')

    # # Логируем
    # log.info(f"Loading dataset '{name}' from file: {file}")
    # log.info(f"Description: {desc}")
    # if citation:
    #     log.info(f"Citation: {citation}")

    # И выводим на экран
    print(f"Loading dataset: {name}")
    print(f"  File:        {file}")
    print(f"  Description: {desc}")
    if citation:
        print(f"  Citation:    {citation}")
