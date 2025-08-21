"""Plot client loss curves over local steps across rounds with aggregation dividers.

Usage (as script):
  uv run python -m scripts.federated_disk.plot_loss --logs-dir logs --clients 0 1 \
      --metric avg_loss --local-steps 5 --output logs/fed_loss.png

Notes:
  - Reads JSONL files like logs/client_<id>.jsonl with lines containing events:
      {"event": "step", "round": int, "client_id": int, "step": int,
       "loss": float, "avg_loss": float, ...}
  - X axis uses cumulative local steps across rounds: e.g., if local-steps=5,
    then round 1 covers [1..5], round 2 covers [6..10], etc.
  - If --local-steps is not provided, it will infer per-round max step from logs
    (per client) and place dividers at cumulative sums of inferred steps of the
    reference client (the smallest client id among selected).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


@dataclass
class StepRecord:
    round: int
    step: int
    loss: float
    avg_loss: float
    ts: float


def _load_client_steps(file_path: Path) -> List[StepRecord]:
    recs: List[StepRecord] = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                if obj.get("event") != "step":
                    continue
                try:
                    recs.append(
                        StepRecord(
                            round=int(obj.get("round", 0)),
                            step=int(obj.get("step", 0)),
                            loss=float(obj.get("loss", obj.get("avg_loss", 0.0))),
                            avg_loss=float(obj.get("avg_loss", obj.get("loss", 0.0))),
                            ts=float(obj.get("ts", 0.0)),
                        )
                    )
                except Exception:
                    # Skip malformed rows
                    continue
    except FileNotFoundError:
        pass
    # sort by (round, step, ts)
    recs.sort(key=lambda r: (r.round, r.step, r.ts))
    return recs


def _infer_round_steps(recs: List[StepRecord]) -> Dict[int, int]:
    """Infer per-round max step for a client from its step records."""
    per_round: Dict[int, int] = {}
    for r in recs:
        per_round[r.round] = max(per_round.get(r.round, 0), r.step)
    return per_round


def _build_cumulative_series(
    recs: List[StepRecord],
    metric: str,
    local_steps: Optional[int] = None,
) -> Tuple[List[int], List[float], Dict[int, int]]:
    """Return (x, y, per_round_steps) for a client's curve.

    If local_steps is provided, the x index uses fixed local_steps per round.
    Otherwise, infer per-round steps from records.
    """
    if metric not in ("loss", "avg_loss"):
        metric = "avg_loss"

    if local_steps is None:
        per_round_steps = _infer_round_steps(recs)
    else:
        # Use fixed steps for all rounds observed
        rounds = sorted({r.round for r in recs})
        per_round_steps = {rnd: int(local_steps) for rnd in rounds}

    # Build prefix sums for cumulative x indexing
    rounds_sorted = sorted(per_round_steps.keys())
    prefix = {0: 0}
    total = 0
    for rnd in rounds_sorted:
        # assume rounds are 1-based; fill gaps if any
        if rnd - 1 in prefix:
            total = prefix[rnd - 1] + per_round_steps.get(rnd - 1, 0)
        else:
            # if missing, continue from last known total
            total = total + per_round_steps.get(rnd - 1, 0)
        prefix[rnd] = total

    xs: List[int] = []
    ys: List[float] = []
    for r in recs:
        base = prefix.get(r.round, 0)
        x = base + max(1, r.step)
        y = getattr(r, metric)
        xs.append(x)
        ys.append(y)
    return xs, ys, per_round_steps


def plot_clients_loss(
    logs_dir: str | os.PathLike,
    clients: Optional[List[int]] = None,
    metric: str = "avg_loss",
    local_steps: Optional[int] = None,
    output: Optional[str] = None,
    dpi: int = 150,
    show: bool = False,
) -> Path:
    """Plot loss curves for specified clients on one figure.

    Returns the saved image path (when output provided) or a default path under logs.
    """
    logs_path = Path(logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)

    # Discover client files
    all_files = sorted(glob.glob(str(logs_path / "client_*.jsonl")))
    if not all_files:
        raise FileNotFoundError(f"No client logs found under: {logs_path}")

    selected: List[Tuple[int, Path]] = []
    for fp in all_files:
        name = Path(fp).name
        try:
            cid = int(name.split("_")[1].split(".")[0])
        except Exception:
            continue
        if clients is None or cid in clients:
            selected.append((cid, Path(fp)))

    if not selected:
        raise RuntimeError("No matching client logs for the requested client IDs")

    # Load and build series per client
    plt.figure(figsize=(10, 6), dpi=dpi)
    # Use a reference client (smallest id) to place vertical dividers if local_steps is None
    ref_cid = min(cid for cid, _ in selected)
    ref_round_steps: Dict[int, int] = {}
    max_x = 0

    for cid, fp in selected:
        recs = _load_client_steps(fp)
        if not recs:
            continue
        xs, ys, per_round = _build_cumulative_series(recs, metric=metric, local_steps=local_steps)
        max_x = max(max_x, max(xs) if xs else 0)
        if cid == ref_cid:
            ref_round_steps = per_round
        plt.plot(xs, ys, label=f"client_{cid}", linewidth=0.8)

    # Add vertical dashed lines between rounds
    if local_steps is not None:
        step = int(local_steps)
        if step > 0:
            vlines = list(range(step + 0, max_x + 1, step))
            # do not draw at zero
            for x in vlines:
                if x <= 0:
                    continue
                plt.axvline(x=x + 0.5, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)
    else:
        # use reference client's inferred per-round steps to set boundaries
        cumul = 0
        for rnd in sorted(ref_round_steps.keys()):
            cumul += ref_round_steps[rnd]
            if cumul > 0:
                plt.axvline(x=cumul + 0.5, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)

    plt.xlabel("Local Step (cumulative across rounds)")
    plt.ylabel(metric)
    plt.title("Client Loss Curves with Server Aggregations")
    plt.legend()
    plt.grid(True, alpha=0.2)

    if output is None:
        output_path = logs_path / "fed_loss.png"
    else:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path)
    if show:
        plt.show()
    plt.close()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot federated client loss curves")
    parser.add_argument("--snapshot-exp", type=str, default="flwr", help="Experiment subfolder under logs/")
    parser.add_argument("--logs-dir", type=str, default=None, help="Directory containing client_*.jsonl (overrides --snapshot-exp if set)")
    parser.add_argument("--clients", type=int, nargs="*", default=None, help="Client IDs to include (default: all)")
    parser.add_argument("--metric", type=str, default="loss", choices=["loss", "avg_loss"], help="Metric to plot")
    parser.add_argument("--local-steps", type=int, default=None, help="Fixed local steps per round; if omitted, infer from logs")
    parser.add_argument("--output", type=str, default=None, help="Output image path (default: logs/fed_loss.png)")
    parser.add_argument("--dpi", type=int, default=1200)
    parser.add_argument("--show", action="store_true", help="Show the figure interactively")

    args = parser.parse_args()
    logs_dir = args.logs_dir if args.logs_dir else str(Path("logs") / args.snapshot_exp)
    out = plot_clients_loss(
        logs_dir=logs_dir,
        clients=args.clients,
        metric=args.metric,
        local_steps=args.local_steps,
        output=args.output,
        dpi=args.dpi,
        show=args.show,
    )
    print(f"Saved figure -> {out}")


if __name__ == "__main__":
    main()
