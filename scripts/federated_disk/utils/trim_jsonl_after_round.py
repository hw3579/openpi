"""Trim JSONL logs after a given round to avoid resume log mixing.

This utility scans JSONL files under a logs directory (defaulting to
  ./logs/<snapshot-exp>
where <snapshot-exp> is read from pyproject.toml) and removes any JSON records
containing a numeric field `round` strictly greater than the specified value.

It is robust to malformed JSONL where multiple JSON objects were concatenated
on the same line without newlines; it uses a brace-balanced scanner to extract
top-level JSON objects, then re-emits a clean, one-object-per-line JSONL file.

Usage:
  uv run python scripts/federated_disk/utils/trim_jsonl_after_round.py --round 6
  uv run python scripts/federated_disk/utils/trim_jsonl_after_round.py --round 6 \
      --logs-dir ./logs/flwr_iid_0820 --dry-run

Options:
  --round N             Keep entries with round <= N; drop entries with round > N.
  --logs-dir PATH       Logs directory; if omitted, defaults to logs/<snapshot-exp> from pyproject.toml.
  --snapshot-exp NAME   Override snapshot-exp to resolve default logs dir.
  --dry-run             Do not modify files; only print summary.
  --drop-noround        Also drop entries that have no 'round' field.
  --no-backup           Do not create a .bak timestamped backup before overwrite.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import pathlib
import sys
from typing import Generator, Iterable, Optional


ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read_snapshot_exp_from_pyproject(pyproject_path: pathlib.Path) -> str:
    """Read snapshot-exp from pyproject.toml; fallback to 'flwr'."""
    snap_exp = "flwr"
    try:
        try:
            import tomllib as _tomllib  # py311+
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
        snap_exp = str(rc.get("snapshot-exp", snap_exp))
    except Exception:
        pass
    return snap_exp


def _iter_json_objects(text: str) -> Generator[str, None, None]:
    """Yield top-level JSON object strings from arbitrary text.

    Uses a brace-balanced scanner with string/escape handling to split concatenated
    JSON objects like: {..}{..}{..} possibly with whitespace between them.
    """
    i = 0
    n = len(text)
    depth = 0
    start = None
    in_str = False
    esc = False
    while i < n:
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        yield text[start : i + 1]
                        start = None
        i += 1


def _parse_jsonl_loose(text: str) -> Iterable[dict]:
    """Parse possibly malformed JSONL, yielding dict objects.

    Strategy:
      1) Try line-by-line json.loads for fast path.
      2) For any line that fails, run brace-balanced extraction and parse each.
    """
    any_failed = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj
                continue
        except Exception:
            any_failed = True
        # Fallback: split concatenated objects
        for frag in _iter_json_objects(line):
            try:
                obj = json.loads(frag)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj
    # If nothing yielded and we had failures, try scanning entire text as a last resort
    if any_failed:
        for frag in _iter_json_objects(text):
            try:
                obj = json.loads(frag)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _to_int_or_none(val) -> Optional[int]:
    try:
        return int(val)
    except Exception:
        return None


def _process_file(path: pathlib.Path, *, max_round: int, dry_run: bool, drop_noround: bool, backup: bool) -> tuple[int, int, int]:
    """Trim a single JSONL file.

    Returns: (kept, dropped, total)
    """
    text = path.read_text(encoding="utf-8", errors="ignore")
    objs = list(_parse_jsonl_loose(text))
    total = len(objs)
    kept_objs = []
    dropped = 0
    for obj in objs:
        r = _to_int_or_none(obj.get("round"))
        if r is None:
            if drop_noround:
                dropped += 1
                continue
            kept_objs.append(obj)
        else:
            if r <= max_round:
                kept_objs.append(obj)
            else:
                dropped += 1

    kept = len(kept_objs)
    if dry_run:
        return kept, dropped, total

    # Write backup
    if backup and path.exists():
        ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_path = path.with_suffix(path.suffix + f".bak.{ts}")
        try:
            backup_path.write_text(text, encoding="utf-8")
        except Exception:
            pass

    # Overwrite with normalized JSONL
    with open(path, "w", encoding="utf-8") as f:
        for obj in kept_objs:
            f.write(json.dumps(obj, ensure_ascii=False))
            f.write("\n")

    return kept, dropped, total


def main() -> None:
    ap = argparse.ArgumentParser(description="Trim JSONL logs after a given round.")
    ap.add_argument("--round", type=int, required=True, help="Keep entries with round <= N; drop round > N")
    ap.add_argument("--logs-dir", type=str, default=None, help="Logs directory (defaults to logs/<snapshot-exp>)")
    ap.add_argument("--snapshot-exp", type=str, default=None, help="Override snapshot-exp to locate default logs dir")
    ap.add_argument("--dry-run", action="store_true", help="Only print summary; do not modify files")
    ap.add_argument("--drop-noround", action="store_true", help="Also drop entries without 'round' field")
    ap.add_argument("--no-backup", dest="backup", action="store_false", help="Do not create .bak backup")
    ap.set_defaults(backup=True)

    args = ap.parse_args()
    max_round = int(args.round)

    logs_dir = pathlib.Path(args.logs_dir) if args.logs_dir else None
    if logs_dir is None:
        snap_exp = args.snapshot_exp or _read_snapshot_exp_from_pyproject(ROOT / "pyproject.toml")
        logs_dir = pathlib.Path("./logs") / snap_exp
    logs_dir = logs_dir.resolve()

    if not logs_dir.exists():
        print(f"[trim-jsonl] Logs dir not found: {logs_dir}")
        sys.exit(2)

    # Target files: server.jsonl and client_*.jsonl
    candidates = []
    for name in os.listdir(logs_dir):
        if not name.endswith(".jsonl"):
            continue
        if name == "server.jsonl" or name.startswith("client_"):
            candidates.append(logs_dir / name)

    if not candidates:
        print(f"[trim-jsonl] No JSONL files found in {logs_dir}")
        return

    print(f"[trim-jsonl] Processing {len(candidates)} files in {logs_dir} (round <= {max_round}, dry_run={args.dry_run})")
    total_kept = total_dropped = total_total = 0
    for file_path in sorted(candidates):
        kept, dropped, total = _process_file(
            file_path,
            max_round=max_round,
            dry_run=bool(args.dry_run),
            drop_noround=bool(args.drop_noround),
            backup=bool(args.backup),
        )
        total_kept += kept
        total_dropped += dropped
        total_total += total
        print(f"  - {file_path.name}: kept={kept}, dropped={dropped}, total={total}")

    print(f"[trim-jsonl] Done. All files processed. Summary: kept={total_kept}, dropped={total_dropped}, total={total_total}")


if __name__ == "__main__":
    main()
