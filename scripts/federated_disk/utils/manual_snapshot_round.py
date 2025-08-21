"""Manually snapshot current global params.npz as a specific FL round.

This writes an Orbax checkpoint step under
  ./checkpoints/<config-name>/<snapshot-exp>/<round>/

It loads params from:
  ./cache/federated_disk/global/current/params.npz (by default)

Usage (examples):
  uv run python scripts/federated_disk/utils/manual_snapshot_round.py --round 6
  uv run python scripts/federated_disk/utils/manual_snapshot_round.py --round 6 \
      --source ./cache/federated_disk/global/current/params.npz
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from typing import Any, Dict


ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Prefer scripts module path for helpers
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from etils import epath  # type: ignore
import numpy as np  # noqa: F401
import orbax.checkpoint as ocp  # type: ignore

from scripts.tools_fl_fed import load_npz_dict as _load_npz_dict  # type: ignore


def _read_defaults_from_pyproject(pyproject_path: pathlib.Path) -> tuple[str, str]:
    """Read (config-name, snapshot-exp) from pyproject.toml.

    Fallbacks: ("pi0_libero_0813_fl", "flwr").
    """
    cfg_name = "pi0_libero_0813_fl"
    snap_exp = "flwr"
    try:
        try:
            import tomllib as _tomllib  # Python 3.11+
        except Exception:  # pragma: no cover
            import tomli as _tomllib  # type: ignore
        with open(pyproject_path, "rb") as f:
            data = _tomllib.load(f)
        rc = (
            data.get("tool", {})
            .get("flwr", {})
            .get("app", {})
            .get("config", {})
        )
        cfg_name = str(rc.get("config-name", cfg_name))
        snap_exp = str(rc.get("snapshot-exp", snap_exp))
    except Exception:
        pass
    return cfg_name, snap_exp


def manual_snapshot(
    *,
    round_id: int,
    source_npz: pathlib.Path,
    config_name: str,
    snapshot_exp: str,
) -> pathlib.Path:
    # Resolve paths
    source_npz = source_npz.resolve()
    if not source_npz.exists():
        raise FileNotFoundError(f"Source NPZ not found: {source_npz}")

    snap_base = pathlib.Path("./checkpoints") / config_name / snapshot_exp
    snap_base = snap_base.resolve()
    snap_base.mkdir(parents=True, exist_ok=True)

    # If step already exists with metadata, do not overwrite
    step_dir = snap_base / str(int(round_id))
    metadata_exists = any((step_dir / name).exists() for name in (
        "_METADATA",
        "_checkpoint_metadata",
        "_CHECKPOINT_METADATA",
        "METADATA",
        "metadata",
    ))
    if metadata_exists:
        print(f"[manual-snapshot] Step {round_id} already exists with metadata: {step_dir}")
        return step_dir

    # Load params dict from NPZ
    params: Dict[str, Any] = _load_npz_dict(source_npz)

    # Setup a minimal CheckpointManager for 'params' only
    manager = ocp.CheckpointManager(
        epath.Path(snap_base),
        item_handlers={
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=None,
            keep_period=None,
            create=True,
            enable_async_checkpointing=False,
            cleanup_tmp_directories=True,
            single_host_load_and_broadcast=True,
            enable_background_delete=True,
        ),
    )

    # Save under the requested step; mirror server structure: {"params": params}
    items = {
        "params": {"params": params},
    }
    manager.wait_until_finished()
    manager.save(int(round_id), items)
    manager.wait_until_finished()
    out_dir = snap_base / str(int(round_id))
    print(f"[manual-snapshot] Saved round {round_id} -> {out_dir}")

    # Best-effort: append a JSONL log record for traceability
    try:
        logs_dir = pathlib.Path("./logs") / snapshot_exp
        logs_dir.mkdir(parents=True, exist_ok=True)
        server_log = logs_dir / "server.jsonl"
        with open(server_log, "a", encoding="utf-8") as f:
            json.dump(
                {
                    "event": "manual_snapshot_saved",
                    "ts": time.time(),
                    "round": int(round_id),
                    "snapshot_dir": str(out_dir),
                    "source_npz": str(source_npz),
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")
    except Exception:
        pass

    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Manually snapshot current global params as a given round")
    parser.add_argument("--round", type=int, required=True, help="Global round number to save as")
    parser.add_argument(
        "--source",
        type=str,
        default="./cache/federated_disk/global/current/params.npz",
        help="Path to source params.npz",
    )
    parser.add_argument("--config-name", type=str, default=None, help="Config name (defaults to pyproject)")
    parser.add_argument("--snapshot-exp", type=str, default=None, help="Snapshot exp (defaults to pyproject)")

    args = parser.parse_args()

    pyproj = ROOT / "pyproject.toml"
    d_cfg, d_exp = _read_defaults_from_pyproject(pyproj)
    cfg_name = args.config_name or d_cfg
    snap_exp = args.snapshot_exp or d_exp

    manual_snapshot(
        round_id=int(args.round),
        source_npz=pathlib.Path(args.source),
        config_name=cfg_name,
        snapshot_exp=snap_exp,
    )


if __name__ == "__main__":
    main()
