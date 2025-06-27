#!/usr/bin/env python3
"""
Script to unpack all .7z archives located in the `data/` directory and extract them into `data/raw/`.
Located at scripts/data_preprocessing/unpack_data.py
"""
import sys
from pathlib import Path
import py7zr
import yaml


def unpack_archives(config_file: str = "config.yaml"):
    """
    Read data_dir and raw_data_dir from config, unpack all .7z from data_dir to raw_data_dir.

    Args:
        config_file (str): Path to YAML config file.
    """
    # Load configuration
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = yaml.safe_load(config_path.read_text())

    # Directories
    data_dir = (Path.cwd() / cfg.get("data_dir", "data")).resolve()
    raw_dir = (Path.cwd() / cfg.get("raw_data_dir", "data/raw")).resolve()

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Find and unpack .7z archives
    archives = list(data_dir.glob("*.7z"))
    if not archives:
        print(f"No .7z archives found in {data_dir}.")
        return

    for archive in archives:
        print(f"Extracting {archive.name} to {raw_dir}...")
        with py7zr.SevenZipFile(archive, mode="r") as z:
            z.extractall(path=raw_dir)

    print(f"All archives unpacked into {raw_dir}.")


if __name__ == '__main__':
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    try:
        unpack_archives(cfg_file)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
