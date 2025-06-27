#!/usr/bin/env python3

"""
Script to unpack all .7z archives in the raw data directory defined in config.yaml.
"""
import sys
from pathlib import Path
import py7zr
import yaml


def unpack_archives_from_config(config_file: str = "config.yaml"):
    """
    Load the raw_data_dir from the given config file and unpack all .7z archives there.

    Args:
        config_file (str): Path to the YAML config file.
    """
    # --- Load configuration ---
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = yaml.safe_load(config_path.read_text())

    # Resolve raw data directory
    raw_dir = (Path.cwd() / cfg.get("raw_data_dir", "data/raw")).resolve()
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    # Find and unpack .7z archives
    archives = list(raw_dir.glob("*.7z"))
    if not archives:
        print("No .7z archives to unpack.")
        return

    for archive in archives:
        print(f"Extracting {archive.name}...")
        with py7zr.SevenZipFile(archive, mode="r") as z:
            z.extractall(path=raw_dir)

    print("All archives unpacked.")


if __name__ == '__main__':
    # Optional: allow passing a different config path
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    try:
        unpack_archives_from_config(cfg_file)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
