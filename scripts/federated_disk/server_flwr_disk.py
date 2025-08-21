"""OpenPI Flower Server (Disk-based): exchange weights via file paths, minimal cache.

Updated workflow (space-saving):
- Keep a single current global npz in ./cache/federated_disk/global/current/params.npz for clients
- Every round overwrite that file after aggregation (no per-round accumulation in cache)
- Persist every N rounds (snapshot-interval) to ./checkpoints/<config>/<exp>/<round>/ via Orbax for inference/restore
This avoids transmitting large tensors in memory and reduces disk usage while keeping periodic checkpoints.
"""

from __future__ import annotations

import os
import json
import gc
import pathlib
from typing import Any, Dict, List, Optional, Tuple
import time

# Force server to CPU to avoid占用显存；同时禁用 XLA 预分配
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from flwr.common import Context, FitIns, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.server import ServerConfig

import numpy as np
import jax
import jax.numpy as jnp
from flax import traverse_util
from scripts.tools_fl_fed import (
    flatten_params as _flatten_params,
    save_npz_dict as _save_npz_dict,
    load_npz_dict as _load_npz_dict,
    save_params_npz as _save_params_npz,
)
import warnings

# Suppress specific deprecation warnings from JAX/Flax internals during FL runs
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r".*linear_util\\.wrap_init.*DebugInfo.*",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r".*shape requires ndarray or scalar arguments.*",
)

# OpenPI imports
import sys
_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from openpi.training import config as _config
from openpi.models import model as _model
from openpi.shared import download
from etils import epath
from openpi.training import checkpoints as _checkpoints

import orbax.checkpoint as ocp  # Only used to read initial checkpoints via _model.restore_params
import flax.nnx as nnx
from tqdm import tqdm  # type: ignore


def _get_fed_opt_from_toml() -> bool:
    """Read pyproject.toml to check if fed_opt is enabled.

    Returns:
        True if fed_opt is enabled, False otherwise (default).
    """
    pyproj = _ROOT / "pyproject.toml"
    try:
        try:
            import tomllib as _tomllib  # Python 3.11+
        except Exception:  # pragma: no cover
            import tomli as _tomllib  # type: ignore
        with open(pyproj, "rb") as f:
            data = _tomllib.load(f)
        config_sec = (
            data.get("tool", {})
            .get("flwr", {})
            .get("app", {})
            .get("config", {})
        )
        return config_sec.get("fed-opt", False)  # Default to False
    except Exception as e:  # pragma: no cover
        print(f"[DiskServer] Failed to read fed_opt from TOML: {e}")
        return False


class DiskFedAvg(FedAvg):
    """FedAvg variant that exchanges only checkpoint paths.

    Aggregation reads params from disk in a streaming fashion to keep memory low.
    """

    def __init__(
        self,
        *,
        config_name: str = "pi0_libero_0813_fl",
        cache_dir: str | pathlib.Path = "./cache/federated_disk",
        target_total_rounds: Optional[int] = None,
        delete_client_files: bool = True,
        # Legacy knobs (no-op)
        keep_last_global: int = 1,
        keep_all_globals: bool = False,
        agg_precision: str = "fp16",
        store_precision: str = "fp16",
        snapshot_interval: int = 0,
        snapshot_dir: str | pathlib.Path | None = None,
        snapshot_exp: str = "flwr",
        resume: bool = False,
        resume_use_cache: bool = False,
        resume_from_round: Optional[int] = None,
        **kwargs,
    ) -> None:
        # Initialize FedAvg first
        super().__init__(**kwargs)

        # Paths and basic flags
        self._config_name = config_name
        self._cache_dir = pathlib.Path(cache_dir).resolve()
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._delete_client_files = bool(delete_client_files)
        # Kept for compatibility; no functional effect
        self._keep_last_global = int(keep_last_global)
        self._keep_all_globals = bool(keep_all_globals)
        self._initialized = False
        self._current_global_path = None
        # Target global round to reach (used to no-op if already reached)
        self._target_total_rounds = int(target_total_rounds) if target_total_rounds is not None else None

        # Data types / precision
        self._agg_dtype = np.float16 if str(agg_precision).lower() == "fp16" else np.float32
        sp = str(store_precision).lower()
        self._store_dtype = np.float16 if sp == "fp16" else np.float32

        # Resume
        self._resume = bool(resume)
        # For safety: even if provided, we no longer allow cache fallback when resuming
        self._resume_use_cache = False
        if resume_use_cache:
            print("[DiskServer] Note: resume-use-cache is ignored; resume only restores from checkpoints")
        self._resume_from_round = int(resume_from_round) if resume_from_round is not None else None
        self._round_offset = 0

        # Snapshot settings
        self._snapshot_interval = int(snapshot_interval)
        default_snap_dir = pathlib.Path("./checkpoints") / self._config_name / str(snapshot_exp)
        self._snapshot_dir = pathlib.Path(snapshot_dir or default_snap_dir).resolve()
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._snapshot_manager = None
        # Remember snapshot experiment name for scoping logs
        self._snapshot_exp = str(snapshot_exp)

        # Logging (scope under logs/<snapshot_exp> to avoid mixing)
        self._logs_dir = (pathlib.Path("./logs") / self._snapshot_exp).resolve()
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        self._server_log = self._logs_dir / "server.jsonl"
        # If a legacy log exists at ./logs/server.jsonl, move/merge it into the scoped folder
        try:
            legacy_log = pathlib.Path("./logs").resolve() / "server.jsonl"
            if legacy_log.exists() and legacy_log != self._server_log:
                import shutil
                if not self._server_log.exists():
                    shutil.move(str(legacy_log), str(self._server_log))
                else:
                    # Append legacy contents then remove legacy file
                    with open(legacy_log, "r", encoding="utf-8") as src, open(self._server_log, "a", encoding="utf-8") as dst:
                        for line in src:
                            dst.write(line)
                    try:
                        legacy_log.unlink(missing_ok=True)
                    except TypeError:
                        legacy_log.unlink()
            # Best-effort: place flwr.log under the same exp subdirectory
            try:
                root_flwr_log = (_ROOT / "flwr.log").resolve()
                scoped_flwr_log = (self._logs_dir / "flwr.log").resolve()
                if root_flwr_log.exists():
                    # If scoped file doesn't exist, create a symlink to avoid disrupting writers
                    if not scoped_flwr_log.exists():
                        try:
                            os.symlink(str(root_flwr_log), str(scoped_flwr_log))
                        except FileExistsError:
                            pass
                        except OSError:
                            # Fallback: copy once
                            import shutil as _shutil
                            _shutil.copy2(str(root_flwr_log), str(scoped_flwr_log))
                else:
                    # Also check legacy ./logs/flwr.log and move it into scoped folder
                    legacy_flwr_log = (pathlib.Path("./logs").resolve() / "flwr.log").resolve()
                    if legacy_flwr_log.exists():
                        import shutil as _shutil
                        _shutil.move(str(legacy_flwr_log), str(scoped_flwr_log))
            except Exception:
                pass
        except Exception:
            pass
        # Fail-fast sticky flag
        self._fatal_failure = False

    def _append_jsonl(self, obj: dict) -> None:
        try:
            with open(self._server_log, "a", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False)
                f.write("\n")
        except Exception:
            pass

    def _current_global_file(self) -> pathlib.Path:
        return self._cache_dir / "global" / "current" / "params.npz"

    # _client_round_file was used in earlier versions; not needed anymore

    def _find_latest_snapshot_round(self) -> Optional[int]:
        d = self._snapshot_dir
        if not d.exists():
            return None
        rounds: List[int] = []
        for ch in d.iterdir():
            if ch.is_dir():
                try:
                    r = int(ch.name)
                except Exception:
                    continue
                # Accept various metadata filenames from different orbax versions
                meta_ok = False
                for name in ("_METADATA", "_checkpoint_metadata", "_CHECKPOINT_METADATA", "METADATA", "metadata"):
                    if (ch / name).exists():
                        meta_ok = True
                        break
                if meta_ok:
                    rounds.append(r)
        return max(rounds) if rounds else None

    # Legacy cleanup removed: server no longer performs cache/global/round_* deletions

    def _initialize_pretrained(self) -> None:
        if self._initialized:
            return
        # Try resume first from checkpoints
        if self._resume:
            # Prefer explicit resume_from_round if provided and valid
            round_to_use: Optional[int] = None
            if self._resume_from_round is not None:
                cand = int(self._resume_from_round)
                snap_dir = self._snapshot_dir / str(cand)
                meta_ok = any(
                    (snap_dir / name).exists()
                    for name in ("_METADATA", "_checkpoint_metadata", "_CHECKPOINT_METADATA", "METADATA", "metadata")
                )
                if meta_ok:
                    round_to_use = cand
            if round_to_use is None:
                latest_round = self._find_latest_snapshot_round()
                if latest_round is not None:
                    round_to_use = int(latest_round)

            if round_to_use is not None:
                try:
                    snap_dir = self._snapshot_dir / str(round_to_use)
                    # Restore from the 'params' item directory inside the checkpoint step dir
                    params = _model.restore_params(
                        str(snap_dir / "params"), restore_type=np.ndarray, dtype=jnp.float16
                    )
                    cur = self._current_global_file()
                    self._current_global_path = _save_params_npz(cur, params, dtype=self._store_dtype)
                    self._round_offset = int(round_to_use)
                    self._initialized = True
                    print(f"[DiskServer] Resumed from checkpoint round {round_to_use}: {snap_dir}")
                    try:
                        self._append_jsonl(
                            {
                                "event": "resume",
                                "ts": time.time(),
                                "from_round": int(round_to_use),
                                "snapshot_dir": str(snap_dir),
                                "current_global": str(self._current_global_path),
                            }
                        )
                    except Exception:
                        pass
                    return
                except Exception as e:
                    # Fail fast when resume cannot restore from checkpoints
                    raise RuntimeError(
                        f"Resume requested but failed to restore from checkpoint round {round_to_use}: {e}"
                    )
            # No valid snapshot found: fallback to cache or pretrained so we can continue
            print(
                f"[DiskServer] Resume requested but no valid checkpoint found under {self._snapshot_dir}. "
                f"Falling back to cache/global or pretrained init."
            )
            # Try to reuse existing global current file in cache if present
            try:
                cur = self._current_global_file()
                cur.parent.mkdir(parents=True, exist_ok=True)
                if cur.exists():
                    self._current_global_path = cur
                    self._round_offset = 0
                    self._initialized = True
                    try:
                        self._append_jsonl(
                            {
                                "event": "resume_fallback_cache",
                                "ts": time.time(),
                                "current_global": str(cur),
                            }
                        )
                    except Exception:
                        pass
                    print(f"[DiskServer] Resume fallback: reuse cached global at {cur}")
                    return
            except Exception as e:
                print(f"[DiskServer] Resume fallback check failed: {e}")
            # If no cache global, fall through to pretrained init below

        # Initialize from pretrained weights in config or start empty
        print(f"[DiskServer] Initializing pretrained weights: {self._config_name}")
        cfg = _config.get_config(self._config_name)
        params_path = getattr(cfg.weight_loader, "params_path", None)
        if params_path:
            # Support remote URLs (e.g., s3://...) by first downloading to local cache
            try:
                local_params_path = str(download.maybe_download(params_path))
            except Exception as e:
                raise FileNotFoundError(
                    f"Failed to download pretrained params from {params_path}: {e}"
                )
            params = _model.restore_params(
                local_params_path, restore_type=np.ndarray, dtype=jnp.float16
            )
            # Intersect with current model state's shapes to avoid mismatches (e.g., state_dim 32 vs 7)
            try:
                model = cfg.model.create(jax.random.key(0))
                _, state = nnx.split(model)
                target = state.to_pure_dict()
                pflat = traverse_util.flatten_dict(params)
                tflat = traverse_util.flatten_dict(target)
                if all(k[-1] == "value" for k in tflat):
                    tflat = {k[:-1]: v for k, v in tflat.items()}
                if all(k[-1] == "value" for k in pflat):
                    pflat = {k[:-1]: v for k, v in pflat.items()}
                inter = {}
                for k, tv in tflat.items():
                    lv = pflat.get(k, None)
                    if lv is not None and np.shape(lv) == np.shape(tv):
                        inter[k] = lv
                params = traverse_util.unflatten_dict(inter)
            except Exception as e:
                print(
                    f"[DiskServer] Warning: failed to intersect params by shape, using raw: {e}"
                )
            save_path = self._current_global_file()
            save_path = _save_params_npz(save_path, params, dtype=self._store_dtype)
            self._current_global_path = save_path
            total = sum(
                np.asarray(v).size for v in traverse_util.flatten_dict(params).values()
            )
            mb = (
                sum(
                    np.asarray(v).nbytes
                    for v in traverse_util.flatten_dict(params).values()
                )
                / 1024
                / 1024
            )
            print(
                f"[DiskServer] Saved pretrained params to {save_path} (src: {local_params_path}) "
                f"({total:,} params, {mb:.1f} MB)"
            )
            # Log init event
            try:
                self._append_jsonl(
                    {
                        "event": "init_pretrained",
                        "ts": time.time(),
                        "params_src": str(local_params_path),
                        "current_global": str(save_path),
                    }
                )
            except Exception:
                pass
        else:
            # No weights; start with empty
            save_path = self._current_global_file()
            _save_params_npz(save_path, {}, dtype=self._store_dtype)
            self._current_global_path = save_path
            print("[DiskServer] No pretrained params in config; starting empty")
            try:
                self._append_jsonl(
                    {
                        "event": "init_empty",
                        "ts": time.time(),
                        "current_global": str(save_path),
                    }
                )
            except Exception:
                pass

        self._initialized = True

    # Strategy hooks
    def initialize_parameters(self, client_manager):
        self._initialize_pretrained()
        # We do not send parameters through the wire; return empty and rely on config paths
        return ndarrays_to_parameters([])

    def configure_fit(self, server_round: int, parameters, client_manager):
        # Ensure resume/init completed so _round_offset is known
        self._initialize_pretrained()
        # Extra safety: if last round triggered fail-fast, terminate here
        if getattr(self, "_fatal_failure", False):
            print("[DiskServer][FATAL] Previous round had failures; terminating.")
            try:
                self._append_jsonl(
                    {"event": "configure_fit_abort_after_failure", "ts": time.time(), "round": int(server_round)}
                )
            except Exception:
                pass
            os._exit(2)
        real_round = self._round_offset + server_round
        # If we've already met/exceeded the target total rounds, do not schedule clients
        if self._target_total_rounds is not None and real_round > self._target_total_rounds:
            print(
                f"[DiskServer] Round {server_round} (global {real_round}): target {self._target_total_rounds} reached; no-op"
            )
            try:
                self._append_jsonl(
                    {
                        "event": "no_op",
                        "ts": time.time(),
                        "round": int(real_round),
                        "reason": "target_total_rounds_reached",
                    }
                )
            except Exception:
                pass
            return []
        print(f"[DiskServer] Round {server_round} (global {real_round}): configuring fit")
        if self._current_global_path is None:
            raise RuntimeError("Global params path not initialized")

        sample_size, min_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_clients)

        cfg = {
            "server_round": real_round,
            "params_path": str(self._current_global_path),  # .npz file
        }
        # Send empty Parameters to avoid tensor broadcast
        ins = FitIns(ndarrays_to_parameters([]), cfg)
        # Log sampled clients for this round
        try:
            sampled = [cid for cid in clients]
        except Exception:
            sampled = []
        self._append_jsonl(
            {
                "event": "configure_fit",
                "ts": time.time(),
                "round": int(real_round),
                "params_path": str(self._current_global_path),
                "clients": sampled,
            }
        )
        return [(cid, ins) for cid in clients]

    def aggregate_fit(self, server_round: int, results, failures):
        real_round = self._round_offset + server_round
        print(
            f"[DiskServer] Round {server_round} (global {real_round}): aggregating {len(results)} results, {len(failures)} failures"
        )
        # Fail-fast: if any client failed in this round, stop the whole program immediately
        if failures and len(failures) > 0:
            try:
                details = []
                for f in failures:
                    try:
                        details.append(str(f))
                    except Exception:
                        details.append("<unknown failure>")
                print("[DiskServer][FATAL] Client failure(s) detected during aggregate_fit. Exiting.")
                self._append_jsonl(
                    {
                        "event": "aggregate_fit_fail_fast",
                        "ts": time.time(),
                        "round": int(real_round),
                        "failures": details,
                    }
                )
                self._fatal_failure = True
            except Exception:
                pass
            # Ensure immediate termination even if the framework catches exceptions
            os._exit(2)
        if not results:
            return None, {}

        # Fast aggregation: batch load and parallel processing
        import concurrent.futures
        import threading
        
        client_data = []  # [(path, examples, client_proxy), ...]
        client_param_paths: List[pathlib.Path] = []

        # Phase 1: Collect paths and examples (fast)
        for client_proxy, fit_res in results:
            metrics: Dict[str, Any] = getattr(fit_res, "metrics", None) or {}
            p = metrics.get("saved_params_path")
            if not p:
                print(f"[DiskServer] Warning: client {client_proxy} returned no 'saved_params_path'")
                continue
            
            # Recover num_examples from sidecar if needed
            reported_examples = getattr(fit_res, "num_examples", 0) or 0
            if (metrics.get("skipped") is True) and (reported_examples == 0):
                try:
                    meta_file = pathlib.Path(p).parent / "meta.json"
                    if meta_file.exists():
                        with open(meta_file, "r", encoding="utf-8") as mf:
                            md = json.load(mf)
                            reported_examples = int(md.get("examples", 0) or 0)
                except Exception:
                    pass
            
            client_data.append((pathlib.Path(p), reported_examples, client_proxy))
            client_param_paths.append(pathlib.Path(p))

        if not client_data:
            print("[DiskServer] No valid client paths; skip update")
            return ndarrays_to_parameters([]), {}

        # Phase 2+3: Serial per-component streaming aggregation with progress (lower peak memory)
        fed_opt_enabled = _get_fed_opt_from_toml()

        def aggregate_component(component: str | None, label: str) -> dict:
            keys = None
            acc = None
            total_w = 0.0
            bar = tqdm(total=len(client_data), desc=f"{label}", leave=False, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}")
            for (p, examples, proxy) in client_data:
                w = float(examples)
                if w <= 0:
                    bar.update(1)
                    continue
                try:
                    loaded = _load_npz_dict(p)
                    pure = loaded
                    if component is not None:
                        if not isinstance(loaded, dict) or component not in loaded:
                            bar.update(1)
                            continue
                        pure = loaded.get(component, {})
                    k, a = _flatten_params(pure)
                except Exception as e:
                    print(f"[DiskServer] Load/flatten failed for {proxy} [{label}]: {e}")
                    bar.update(1)
                    continue
                if keys is None:
                    keys = k
                    acc = []
                    for i, arr in enumerate(a):
                        if np.issubdtype(arr.dtype, np.floating):
                            acc.append(arr.astype(self._agg_dtype, copy=False) * np.array(w, dtype=self._agg_dtype))
                        else:
                            acc.append(arr.copy())
                        if (i % 2) == 0:
                            bar.set_postfix(client=str(proxy), layer=f"{i+1}/{len(a)}")
                    total_w = w
                else:
                    key_to_idx = {kk: i for i, kk in enumerate(keys)}
                    for i, (kk, arr) in enumerate(zip(k, a)):
                        j = key_to_idx.get(kk, None)
                        if j is None:
                            continue
                        if np.issubdtype(arr.dtype, np.floating):
                            acc[j] += arr.astype(self._agg_dtype, copy=False) * np.array(w, dtype=self._agg_dtype)
                        if (i % 2) == 0:
                            bar.set_postfix(client=str(proxy), layer=f"{i+1}/{len(a)}")
                    total_w += w
                bar.update(1)
            # Close the bar explicitly to avoid leftover lines
            try:
                bar.close()
            except Exception:
                pass
            if keys is None or acc is None or total_w == 0.0:
                return {}
            sw = np.array(total_w, dtype=self._agg_dtype)
            out = []
            for arr in acc:
                out.append(arr / sw if np.issubdtype(arr.dtype, np.floating) else arr)
            return traverse_util.unflatten_dict({k: v for k, v in zip(keys, out)})

        # Aggregate params first
        aggregated_params = aggregate_component(None if not fed_opt_enabled else "params", "params")
        if not aggregated_params:
            print("[DiskServer] No valid client arrays; skip update")
            return ndarrays_to_parameters([]), {}
        aggregated_dict = {"params": aggregated_params}

        # Then optimizer and EMA serially (to bound memory)
        if fed_opt_enabled:
            aggregated_opt_state = aggregate_component("opt_state", "opt")
            if aggregated_opt_state:
                aggregated_dict["opt_state"] = aggregated_opt_state
            aggregated_ema_params = aggregate_component("ema_params", "ema")
            if aggregated_ema_params:
                aggregated_dict["ema_params"] = aggregated_ema_params

        # Save the aggregated result
        new_current = self._current_global_file()
        if fed_opt_enabled and len(aggregated_dict) > 1:
            _save_npz_dict(new_current, aggregated_dict, dtype=self._store_dtype)
            print(f"[FedOpt] Saved aggregated training state: {new_current}")
        else:
            _save_params_npz(new_current, aggregated_params, dtype=self._store_dtype)
            print(f"[DiskServer] Updated current global params: {new_current}")

        self._current_global_path = new_current
        # Log aggregate summary
        try:
            client_paths = [str(p) for p in client_param_paths]
        except Exception:
            client_paths = []
        self._append_jsonl(
            {
                "event": "aggregate_fit",
                "ts": time.time(),
                "round": int(real_round),
                "num_results": int(len(results)),
                "num_failures": int(len(failures)),
                "client_param_paths": client_paths,
                "new_global_path": str(new_current),
            }
        )

        # Optional: save an Orbax snapshot every N rounds for inference reuse
        if self._snapshot_interval > 0 and (real_round % self._snapshot_interval == 0):
            try:
                if self._snapshot_manager is None:
                    self._snapshot_manager = ocp.CheckpointManager(
                        epath.Path(self._snapshot_dir),
                        item_handlers={
                            "assets": _checkpoints.CallbackHandler(),
                            "params": ocp.PyTreeCheckpointHandler(),
                        },
                        options=ocp.CheckpointManagerOptions(
                            max_to_keep=None,  # Keep all snapshots for FL rounds
                            keep_period=None,
                            create=False,
                            enable_async_checkpointing=False,
                            cleanup_tmp_directories=True,
                            single_host_load_and_broadcast=True,
                            enable_background_delete=True,
                        ),
                    )
                # Ensure directory exists
                epath.Path(self._snapshot_dir).mkdir(parents=True, exist_ok=True)
                # Prepare assets save callback consistent with training.save_state
                cfg = _config.get_config(self._config_name)
                data_config = cfg.data.create(cfg.assets_dirs, cfg.model)

                def save_assets(directory: epath.Path):
                    try:
                        from openpi.shared import normalize as _normalize
                        if data_config.norm_stats is not None and data_config.asset_id is not None:
                            _normalize.save(directory / data_config.asset_id, data_config.norm_stats)
                    except Exception as e:
                        print(f"[DiskServer] Snapshot assets save skipped: {e}")

                items = {
                    "assets": save_assets,
                    "params": {"params": aggregated_params},
                }
                self._snapshot_manager.wait_until_finished()
                self._snapshot_manager.save(real_round, items)
                self._snapshot_manager.wait_until_finished()
                snap_dir = self._snapshot_dir / str(real_round)
                print(
                    f"[DiskServer] Snapshot saved (round {real_round}) -> {snap_dir}"
                )
            except Exception as e:
                print(f"[DiskServer] Snapshot failed at round {server_round}: {e}")

        # Cleanup client files for this round
        if self._delete_client_files:
            for p in client_param_paths:
                try:
                    base = p.parent  # .../round_xxxxx/params.npz -> .../round_xxxxx
                    if base.exists():
                        import shutil
                        shutil.rmtree(base)
                except Exception as e:
                    print(f"[DiskServer] Cleanup failed for {p}: {e}")

        # No per-round globals kept in cache; only keep the single current file.

        # Aggressive memory cleanup to reduce Ray OOM risk before scheduling next round
        try:
            for _name in (
                'loaded_clients','loaded_opt_states','loaded_ema_params','aggregated_params','aggregated_dict'
            ):
                if _name in locals():
                    locals()[_name] = None
        except Exception:
            pass
        gc.collect()
        try:
            jax.clear_caches()
        except Exception:
            pass

        # We still need to return Parameters for framework consistency; keep empty to avoid memory
        return ndarrays_to_parameters([]), {"global_params_path": str(new_current)}

    def _aggregate_component(self, loaded_data: List[Tuple], component_name: str) -> dict:
        """Fast aggregation for a single component (params, opt_state, or ema_params).
        
        Args:
            loaded_data: List of (keys, arrays, examples) tuples for the component
            component_name: Name of the component being aggregated (for logging)
            
        Returns:
            Dictionary containing the aggregated component
        """
        if not loaded_data:
            return {}
            
        # Use first client as template
        keys_order, template_arrays, first_examples = loaded_data[0]
        key_to_idx = {kk: i for i, kk in enumerate(keys_order)}
        
        # Pre-allocate accumulators with proper dtype
        acc_arrays = []
        for arr in template_arrays:
            if np.issubdtype(arr.dtype, np.floating):
                # Initialize with first client's weighted contribution
                acc_arrays.append(
                    arr.astype(self._agg_dtype, copy=False) * np.array(first_examples, dtype=self._agg_dtype)
                )
            else:
                # Non-floating: keep first copy
                acc_arrays.append(arr.copy())
        
        total_w = float(first_examples)

        # Vectorized accumulation for remaining clients
        for k, a, examples in loaded_data[1:]:
            weight = np.array(examples, dtype=self._agg_dtype)
            for kk, arr in zip(k, a):
                if kk not in key_to_idx:
                    continue
                i = key_to_idx[kk]
                if np.issubdtype(arr.dtype, np.floating):
                    # In-place accumulation for speed
                    acc_arrays[i] += arr.astype(self._agg_dtype, copy=False) * weight
            total_w += float(examples)

        if total_w == 0:
            print(f"[DiskServer] Zero total weight for {component_name}; returning empty")
            return {}

        # Vectorized final normalization
        sw = np.array(total_w, dtype=self._agg_dtype)
        out_arrays: List[np.ndarray] = []
        for arr in acc_arrays:
            if np.issubdtype(arr.dtype, np.floating):
                # In-place division when possible to avoid extra allocation
                if arr.dtype == self._agg_dtype:
                    arr /= sw
                    out_arrays.append(arr)
                else:
                    out_arrays.append(arr / sw)
            else:
                out_arrays.append(arr)

        return traverse_util.unflatten_dict({k: v for k, v in zip(keys_order, out_arrays)})

    def configure_evaluate(self, server_round: int, parameters, client_manager):
        print(f"[DiskServer] Round {server_round}: evaluation disabled")
        return []

    def aggregate_evaluate(self, server_round: int, results, failures):
        print(f"[DiskServer] Round {server_round}: evaluation aggregation skipped")
        return None, {}


def server_fn(context: Context) -> ServerAppComponents:
    # Read minimal settings
    config_name = context.run_config.get("config-name", "pi0_libero_0813_fl")
    num_rounds = int(context.run_config.get("num-server-rounds", 10))
    min_fit_clients = int(context.run_config.get("min-fit-clients", 1))
    min_available_clients = int(context.run_config.get("min-available-clients", 1))
    fraction_fit = float(context.run_config.get("fraction-fit", 1.0))
    cache_dir = context.run_config.get("cache-dir", "./cache/federated_disk")
    delete_client_files = bool(context.run_config.get("delete-client-files", True))
    # Legacy config keys kept for compatibility (no effect)
    keep_last_global = int(context.run_config.get("keep-last-global", 2))
    keep_all_globals = bool(context.run_config.get("keep-all-globals", True))
    agg_precision = context.run_config.get("agg-precision", "fp16")
    store_precision = context.run_config.get("store-precision", "fp16")
    snapshot_interval = int(context.run_config.get("snapshot-interval", 0))
    snapshot_dir = context.run_config.get("snapshot-dir", None)
    snapshot_exp = context.run_config.get("snapshot-exp", "flwr")
    resume = bool(context.run_config.get("resume", False))
    # resume-use-cache is ignored by strategy (resume only from checkpoints)
    resume_use_cache = False
    resume_from_round = context.run_config.get("resume-from-round", None)
    try:
        resume_from_round = int(resume_from_round) if resume_from_round is not None else None
    except Exception:
        resume_from_round = None

    print("[DiskServer] Configuration:")
    print(f"  - config: {config_name}")
    print(f"  - target_total_rounds: {num_rounds}")
    print(f"  - fraction_fit: {fraction_fit}")
    print(f"  - cache_dir: {cache_dir}")

    strategy = DiskFedAvg(
        config_name=config_name,
        cache_dir=cache_dir,
        target_total_rounds=num_rounds,
        delete_client_files=delete_client_files,
        keep_last_global=keep_last_global,
    keep_all_globals=keep_all_globals,
        agg_precision=agg_precision,
    store_precision=store_precision,
        snapshot_interval=snapshot_interval,
        snapshot_dir=snapshot_dir,
        snapshot_exp=snapshot_exp,
    resume=resume,
    resume_use_cache=resume_use_cache,
    resume_from_round=resume_from_round,
        min_fit_clients=min_fit_clients,
        min_available_clients=min_available_clients,
        fraction_fit=fraction_fit,
        fraction_evaluate=0.0,
        min_evaluate_clients=0,
        initial_parameters=None,
    )

    # Adjust num_rounds when resuming so that the total number of global rounds equals num-server-rounds
    effective_rounds = num_rounds
    if resume:
        try:
            # Determine round offset (already-trained global rounds)
            if resume_from_round is not None:
                round_offset = int(resume_from_round)
            else:
                # Auto-detect latest snapshot step as round offset
                snap_base = pathlib.Path(snapshot_dir or (pathlib.Path("./checkpoints") / config_name / str(snapshot_exp)))
                round_offset = None
                if snap_base.exists():
                    candidates = []
                    for ch in snap_base.iterdir():
                        if ch.is_dir():
                            try:
                                r = int(ch.name)
                            except Exception:
                                continue
                            for name in ("_METADATA", "_checkpoint_metadata", "_CHECKPOINT_METADATA", "METADATA", "metadata"):
                                if (ch / name).exists():
                                    candidates.append(r)
                                    break
                    round_offset = max(candidates) if candidates else None
            if round_offset is not None and round_offset >= 0:
                effective_rounds = max(0, num_rounds - round_offset)
                print(f"[DiskServer] Resuming: detected round_offset={round_offset}; computed effective_rounds={effective_rounds}")
        except Exception as e:
            print(f"[DiskServer] Resume rounds adjustment failed: {e}")
    # If nothing left to do, run a no-op round (strategy will skip) to let Flower exit cleanly
    config = ServerConfig(num_rounds=effective_rounds if effective_rounds > 0 else 1)
    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)


if __name__ == "__main__":
    print("OpenPI Disk-based Flower Server - use 'flwr run' to start")
