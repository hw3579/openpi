#!/usr/bin/env python3
"""Cleanup legacy cache/global/round_* directories.

Usage:
  python -m scripts.cleanup_cache --cache-dir ./cache/federated_disk
"""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.tools_fl_fed import cleanup_cache_global_rounds


def main() -> None:
    parser = argparse.ArgumentParser(description="Cleanup legacy cache/global/round_* directories")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./cache/federated_disk",
        help="Root cache directory (default: ./cache/federated_disk)",
    )
    args = parser.parse_args()

    removed = cleanup_cache_global_rounds(args.cache_dir)
    print(f"Removed {removed} legacy round_* directories under {Path(args.cache_dir) / 'global'}")


if __name__ == "__main__":
    main()
