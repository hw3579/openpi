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

        # Logging
        self._logs_dir = pathlib.Path("./logs").resolve()
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        self._server_log = self._logs_dir / "server.jsonl"

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
            # No valid snapshot found: do not fallback to cache or pretrained
            raise RuntimeError(
                f"Resume requested but no valid checkpoint found under {self._snapshot_dir}. "
                f"Please create snapshots or disable resume."
            )

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
        if not results:
            return None, {}
        # real_round already computed above

        # Streaming weighted sum
        acc_arrays: List[np.ndarray] | None = None
        keys_order: List[Tuple[str, ...]] | None = None
        total_w = 0.0
        client_param_paths: List[pathlib.Path] = []

        for client_proxy, fit_res in results:
            metrics: Dict[str, Any] = getattr(fit_res, "metrics", None) or {}
            p = metrics.get("saved_params_path")
            if not p:
                print(f"[DiskServer] Warning: client {client_proxy} returned no 'saved_params_path'")
                continue
            # If client reported skipped, we may need to recover num_examples from sidecar or fallbacks
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
            try:
                params = _load_npz_dict(p)
            except Exception as e:
                print(f"[DiskServer] Failed to load client params from {p}: {e}")
                continue

            k, a = _flatten_params(params)
            # Establish key order on first successful client
            if acc_arrays is None:
                keys_order = k
                acc_arrays = []
                for arr in a:
                    if np.issubdtype(arr.dtype, np.floating):
                        acc_arrays.append(
                            arr.astype(self._agg_dtype, copy=False)
                            * np.array(reported_examples, dtype=self._agg_dtype)
                        )
                    else:
                        # Non-floating: keep first copy and ignore in weighted sum
                        acc_arrays.append(arr)
            else:
                # Align by intersection of keys
                key_to_idx = {kk: i for i, kk in enumerate(keys_order)}
                for kk, arr in zip(k, a):
                    if kk not in key_to_idx:
                        continue
                    i = key_to_idx[kk]
                    if np.issubdtype(arr.dtype, np.floating):
                        acc_arrays[i] += (
                            arr.astype(self._agg_dtype, copy=False)
                            * np.array(reported_examples, dtype=self._agg_dtype)
                        )
                    else:
                        # keep first value (already in acc_arrays)
                        pass

            total_w += float(reported_examples)
            client_param_paths.append(pathlib.Path(p))

        if acc_arrays is None or total_w == 0 or keys_order is None:
            print("[DiskServer] No valid client arrays or zero weight; skip update")
            return ndarrays_to_parameters([]), {}

        # Finalize average for floating arrays; saver will cast to configured store dtype
        out_arrays: List[np.ndarray] = []
        sw = np.array(total_w, dtype=self._agg_dtype)
        for arr in acc_arrays:
            if np.issubdtype(arr.dtype, np.floating):
                out_arrays.append(arr / sw)
            else:
                out_arrays.append(arr)

        new_params = traverse_util.unflatten_dict({k: v for k, v in zip(keys_order, out_arrays)})
        # Overwrite single current global file in cache
        new_current = self._current_global_file()
        _save_params_npz(new_current, new_params, dtype=self._store_dtype)
        self._current_global_path = new_current
        print(f"[DiskServer] Updated current global params: {new_current}")
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
                    "params": {"params": new_params},
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

        gc.collect()
        try:
            jax.clear_caches()
        except Exception:
            pass

        # We still need to return Parameters for framework consistency; keep empty to avoid memory
        return ndarrays_to_parameters([]), {"global_params_path": str(new_current)}

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
