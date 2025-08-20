"""OpenPI Flower Client (Disk-based): receives global params path, trains, saves to disk, returns path.

Goals
- Avoid transmitting large ndarrays; only exchange file paths
- Load params using Orbax helpers from openpi.models.model.restore_params
- Save params to ./cache/federated_disk/client_<id>/round_<r>/params
- Deterministic IID uniform split across clients with a TOML-controlled split seed
"""
from __future__ import annotations

# Early suppress of specific JAX DeprecationWarning before any heavy imports
import warnings as _early_warnings
_early_warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r".*linear_util\\.wrap_init.*DebugInfo.*",
)

import os
import json
import pathlib
import dataclasses
import gc
from typing import List
import time

# Client memory hygiene for JAX
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")
# Do NOT force CPU by default. If previously set to CPU but GPUs are exposed, clear it to allow GPU.
if os.environ.get("JAX_PLATFORMS", "") == "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "-1"):
    os.environ.pop("JAX_PLATFORMS", None)

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
import warnings
import contextlib

# Suppress specific deprecation warning from JAX internals
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r".*linear_util\\.wrap_init.*DebugInfo.*",
)
import warnings

# Suppress specific deprecation warnings from JAX/Flax internals
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r".*linear_util\\.wrap_init.*DebugInfo.*",
)
from tqdm import tqdm  # type: ignore

# Suppress noisy deprecation warnings from JAX/Flax internals (does not hide other warnings)
import warnings
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

import numpy as np
import jax
import jax.numpy as jnp

# Enable JAX persistent compilation cache (shared across Ray client processes)
try:
    from jax.experimental import compilation_cache as _jax_cc  # type: ignore
    _cc_dir = os.environ.get(
        "JAX_PERSISTENT_CACHE_DIR",
        str(pathlib.Path(__file__).resolve().parents[2] / "cache" / "jax_persistent"),
    )
    pathlib.Path(_cc_dir).mkdir(parents=True, exist_ok=True)
    _jax_cc.initialize_cache(_cc_dir)
    # Optional: print cache dir once for debugging
    print(f"[DiskClient] JAX persistent compilation cache: {_cc_dir}")
except Exception as _e:
    # Safe to continue without persistent cache
    pass
from flax import traverse_util
from scripts.tools_fl_fed import (
    load_npz_dict as _load_npz_dict,
    save_npz_dict as _save_npz_dict,
)

# OpenPI imports
import sys
_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import openpi.training.config as _config
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.data_loader as dl
import openpi.models.model as _model
import train as train_mod
# Note: orbax is not used in the disk-exchange path on client


class IIDSubsetDataset:
    def __init__(self, base, indices: List[int]):
        self._base = base
        self._indices = indices
    def __getitem__(self, idx: int):
        return self._base[self._indices[idx]]
    def __len__(self) -> int:
        return len(self._indices)


# build_iid_indices is now imported from scripts.tools_fl_fed
def _get_split_seed_from_toml(config_name: str) -> int | None:
    """Read split seed from pyproject.toml under [tool.flwr.app.iid.<config_name>].

    Example layout:
    [tool.flwr.app.iid.pi0_libero_0813_fl]
    split_seed = 20250820
    """
    root = pathlib.Path(__file__).resolve().parents[2]
    pyproj = root / "pyproject.toml"
    try:
        try:
            import tomllib as _tomllib  # Python 3.11+
        except Exception:  # pragma: no cover
            import tomli as _tomllib  # type: ignore
        with open(pyproj, "rb") as f:
            data = _tomllib.load(f)
        sec = (
            data.get("tool", {})
            .get("flwr", {})
            .get("app", {})
            .get("iid", {})
        )
        cfgsec = sec.get(config_name)
        if isinstance(cfgsec, dict):
            val = cfgsec.get("split_seed")
            if isinstance(val, int):
                return int(val)
            # allow numeric string
            if isinstance(val, str) and val.strip().isdigit():
                return int(val.strip())
        return None
    except Exception:
        return None


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
        print(f"[DiskClient] Failed to read fed_opt from TOML: {e}")
        return False


def _build_uniform_indices(n_samples: int, total_clients: int, client_id: int, seed: int) -> list[int]:
    """Uniform partition: random permute by seed, then contiguous split.

    Ensures non-overlapping, near-equal sized shards across clients.
    """
    total_clients = max(1, int(total_clients))
    client_id = int(client_id)
    if client_id < 0 or client_id >= total_clients:
        raise ValueError(f"client_id {client_id} out of range for total_clients {total_clients}")
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(int(n_samples))
    base = int(n_samples) // total_clients
    rem = int(n_samples) % total_clients
    start = client_id * base + min(client_id, rem)
    length = base + (1 if client_id < rem else 0)
    end = start + length
    sel = perm[start:end]
    # Keep order stable for dataloader if needed
    return list(map(int, np.sort(sel)))


# flatten is provided by tools for saving pure dicts


def _save_params_npz(file_path: pathlib.Path | str, state: training_utils.TrainState, *, dtype: np.dtype) -> str:
    # Read fed_opt setting from TOML
    fed_opt_enabled = _get_fed_opt_from_toml()
    
    if fed_opt_enabled:
        # Save full training state (params + opt_state + ema_params)
        save_dict = {}
        
        # Always save model parameters (EMA if available, otherwise regular params)
        params_pure = (state.ema_params or state.params).to_pure_dict()
        params_pure = jax.tree.map(lambda x: getattr(x, "value", x), params_pure)
        save_dict["params"] = params_pure
        
        # Save optimizer state
        if state.opt_state is not None:
            opt_state_pure = jax.tree.map(lambda x: getattr(x, "value", x), state.opt_state)
            save_dict["opt_state"] = opt_state_pure
        
        # Save EMA parameters if they exist and are different from regular params
        if state.ema_params is not None:
            ema_pure = state.ema_params.to_pure_dict()
            ema_pure = jax.tree.map(lambda x: getattr(x, "value", x), ema_pure)
            save_dict["ema_params"] = ema_pure
        
        return _save_npz_dict(file_path, save_dict, dtype=dtype)
    else:
        # Standard mode: only save model parameters
        pure = (state.ema_params or state.params).to_pure_dict()
        pure = jax.tree.map(lambda x: getattr(x, "value", x), pure)
        return _save_npz_dict(file_path, pure, dtype=dtype)


def _apply_params_from(path: str, state: training_utils.TrainState) -> training_utils.TrainState:
    # Support .npz global params sent by server
    file = pathlib.Path(path)
    if file.suffix == ".npz":
        loaded = _load_npz_dict(file)
    else:
        loaded = _model.restore_params(path, restore_type=jax.Array, dtype=None)
    
    # Check if this is fed_opt format (contains multiple keys like params, opt_state, ema_params)
    fed_opt_enabled = _get_fed_opt_from_toml()
    is_fed_opt_format = (
        isinstance(loaded, dict) and 
        any(key in loaded for key in ["params", "opt_state", "ema_params"])
    )
    
    if fed_opt_enabled and is_fed_opt_format:
        # Fed_opt mode: restore full training state
        new_state = state
        
        # Apply model parameters if present
        if "params" in loaded:
            new_state = _apply_model_params(loaded["params"], new_state)
        
        # Apply optimizer state if present
        if "opt_state" in loaded and loaded["opt_state"] is not None:
            # Validate shape compatibility and apply
            try:
                # Simple shape validation for opt_state
                import jax
                current_opt_flat = jax.tree_util.tree_flatten(state.opt_state)[0]
                loaded_opt_flat = jax.tree_util.tree_flatten(loaded["opt_state"])[0]
                
                if len(current_opt_flat) == len(loaded_opt_flat):
                    # Basic check passed, apply optimizer state
                    new_state = dataclasses.replace(new_state, opt_state=loaded["opt_state"])
                    print(f"[FedOpt] Applied optimizer state")
                else:
                    print(f"[FedOpt] Optimizer state shape mismatch, keeping local state")
            except Exception as e:
                print(f"[FedOpt] Failed to apply optimizer state: {e}")
        
        # Apply EMA parameters if present
        if "ema_params" in loaded and loaded["ema_params"] is not None:
            try:
                import flax.nnx as nnx
                # Convert loaded EMA params to proper format
                model = nnx.merge(new_state.model_def, new_state.params)
                graphdef, nnx_state = nnx.split(model)
                nnx_state.replace_by_pure_dict(loaded["ema_params"])
                ema_params = nnx.state(nnx.merge(graphdef, nnx_state))
                new_state = dataclasses.replace(new_state, ema_params=ema_params)
                print(f"[FedOpt] Applied EMA parameters")
            except Exception as e:
                print(f"[FedOpt] Failed to apply EMA parameters: {e}")
        
        return new_state
    else:
        # Standard mode: only apply model parameters
        return _apply_model_params(loaded, state)


def _apply_model_params(loaded: dict, state: training_utils.TrainState) -> training_utils.TrainState:
    """Apply model parameters to training state (extracted from original _apply_params_from)."""
    import flax.nnx as nnx
    model = nnx.merge(state.model_def, state.params)
    graphdef, nnx_state = nnx.split(model)

    # Intersect by shape: only load tensors whose shapes match the current model state
    target_pure = nnx_state.to_pure_dict()
    tflat = traverse_util.flatten_dict(target_pure)
    lflat = traverse_util.flatten_dict(loaded)
    oflat = {}
    # Prefer staging array creation on CPU to avoid early GPU OOM
    try:
        _cpu0 = jax.devices("cpu")[0]
    except Exception:
        _cpu0 = None
    for k, tval in tflat.items():
        lval = lflat.get(k, None)
        if lval is not None and np.shape(lval) == np.shape(tval):
            # Cast to target dtype using jax.numpy; place on CPU first to reduce VRAM pressure
            target_dtype = getattr(tval, "dtype", None)
            _ctx = jax.default_device(_cpu0) if _cpu0 is not None else contextlib.nullcontext()
            try:
                with _ctx:
                    arr_host = (
                        jnp.asarray(lval, dtype=target_dtype)
                        if target_dtype is not None
                        else jnp.asarray(lval)
                    )
                oflat[k] = arr_host
            except Exception:
                # Fallback without dtype if casting fails
                with _ctx:
                    oflat[k] = jnp.asarray(lval)
        else:
            # Keep existing initialized value when shape mismatches
            oflat[k] = tval
    merged = traverse_util.unflatten_dict(oflat)

    nnx_state.replace_by_pure_dict(merged)
    model = nnx.merge(graphdef, nnx_state)
    new_params = nnx.state(model)
    return dataclasses.replace(state, params=new_params)


class OpenPIFlowerDiskClient(NumPyClient):
    def __init__(
        self,
        config_name: str,
        total_clients: int,
        client_id: int,
        local_steps: int,
        virtual_clients: int = 1,
        batch_size: int | None = None,
        num_workers: int = 2,
        fsdp_devices: int = 1,
        out_cache_dir: str | None = None,
        store_precision: str = "fp16",
    ) -> None:
        self.cfg = _config.get_config(config_name)
        self.cfg = dataclasses.replace(
            self.cfg,
            exp_name=f"flwr_disk_client_{client_id}",
            wandb_enabled=False,
            fsdp_devices=fsdp_devices,
            batch_size=(batch_size if batch_size is not None else self.cfg.batch_size),
            num_workers=num_workers,
        )
        self.total_clients = total_clients
        self.client_id = client_id
        self.local_steps = local_steps
        self.virtual_clients = max(1, int(virtual_clients))
        self.initialized = False
        self.mesh = None
        self.state = None
        self.state_sharding = None
        self.replicated_sharding = None
        self.data_sharding = None
        self.data_iter = None
        self.n_samples = 0
        self.out_cache_dir = pathlib.Path(out_cache_dir or "./cache/federated_disk").resolve()
        self.out_cache_dir.mkdir(parents=True, exist_ok=True)
        # Logs directory for JSONL step-wise metrics (resume-friendly append)
        self.logs_dir = pathlib.Path("./logs").resolve()
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = self.logs_dir / f"client_{self.client_id}.jsonl"
        # Storage precision for mid-pipeline NPZ exchange
        sp = str(store_precision).lower()
        self._store_dtype = np.float16 if sp == "fp16" else np.float32

    def _append_jsonl(self, obj: dict) -> None:
        try:
            with open(self._log_file, "a", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False)
                f.write("\n")
        except Exception:
            pass

    def _lazy_init(self):
        if self.initialized:
            return
        try:
            print(f"[DiskClient {self.client_id}] JAX backend: {jax.default_backend()}, devices: {jax.devices()}")
        except Exception:
            pass
        self.mesh = sharding.make_mesh(self.cfg.fsdp_devices)
        self.data_sharding = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        self.replicated_sharding = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec())
        data_config = self.cfg.data.create(self.cfg.assets_dirs, self.cfg.model)
        base_dataset = dl.create_dataset(data_config, self.cfg.model)
        self.n_samples = len(base_dataset)
        # Read split seed from TOML if present, else fallback to cfg.seed
        split_seed = _get_split_seed_from_toml(self.cfg.name) or int(self.cfg.seed)
        # Build deterministic uniform shard for this client
        idx = _build_uniform_indices(self.n_samples, self.total_clients, self.client_id, split_seed)
        # Log split meta once for determinism auditing
        try:
            self._append_jsonl(
                {
                    "event": "iid_split",
                    "ts": time.time(),
                    "client_id": int(self.client_id),
                    "total_clients": int(self.total_clients),
                    "split_seed": int(split_seed),
                    "n_samples": int(self.n_samples),
                    "count": int(len(idx)),
                }
            )
        except Exception:
            pass
        dataset = dl.transform_dataset(IIDSubsetDataset(base_dataset, idx), data_config, skip_norm_stats=False)
        local_bs = self.cfg.batch_size // jax.device_count()
        if self.cfg.batch_size % jax.device_count() != 0:
            raise ValueError(f"Batch size {self.cfg.batch_size} must be divisible by device count {jax.device_count()}")
        self.loader = dl.TorchDataLoader(
            dataset,
            local_batch_size=local_bs,
            sharding=self.data_sharding,
            shuffle=True,
            num_batches=None,
            num_workers=self.cfg.num_workers,
            seed=self.cfg.seed + self.client_id,
        )
        self.data_iter = iter(self.loader)

        init_rng = jax.random.key(self.cfg.seed)
        self.train_rng = jax.random.key(self.cfg.seed + 1000 + self.client_id)
        self.state, self.state_sharding = train_mod.init_train_state(self.cfg, init_rng, self.mesh, resume=False)
        self.ptrain_step = jax.jit(
            lambda rng, state, batch: train_mod.train_step(self.cfg, rng, state, batch),
            in_shardings=(self.replicated_sharding, self.state_sharding, self.data_sharding),
            out_shardings=(self.state_sharding, self.replicated_sharding),
            donate_argnums=(1,),
        )
        self.initialized = True

    def fit(self, parameters, config):
        # Best-effort pre-fit cleanup if上一轮遗留
        try:
            gc.collect(); jax.clear_caches()
        except Exception:
            pass
        rnd = int(config.get("server_round", 0))
        # Fast-path: if this round's params already exist (from a previous partial run), skip heavy init/training
        try:
            out_path = self.out_cache_dir / f"client_{self.client_id}" / f"round_{rnd:05d}" / "params.npz"
            if out_path.exists():
                # Prefer sidecar meta for examples; fallback to JSONL; else approximate
                examples = None
                meta_file = out_path.parent / "meta.json"
                try:
                    if meta_file.exists():
                        with open(meta_file, "r", encoding="utf-8") as mf:
                            md = json.load(mf)
                            examples = int(md.get("examples", 0) or 0)
                except Exception:
                    examples = None
                if examples is None:
                    try:
                        if self._log_file.exists():
                            with open(self._log_file, "r", encoding="utf-8") as f:
                                for line in f:
                                    try:
                                        rec = json.loads(line)
                                    except Exception:
                                        continue
                                    if (
                                        isinstance(rec, dict)
                                        and rec.get("event") == "round_end"
                                        and int(rec.get("round", -1)) == rnd
                                        and int(rec.get("client_id", -1)) == int(self.client_id)
                                    ):
                                        examples = int(rec.get("examples", 0) or 0)
                    except Exception:
                        examples = None
                if examples is None:
                    examples = int(self.cfg.batch_size) * int(self.local_steps)
                # Log skip event
                self._append_jsonl(
                    {
                        "event": "round_skip_existing",
                        "ts": time.time(),
                        "round": int(rnd),
                        "client_id": int(self.client_id),
                        "saved_params_path": str(out_path),
                        "examples": int(examples),
                    }
                )
                print(f"[DiskClient {self.client_id}] Skip training for round {rnd}: reuse {out_path}")
                return [], int(examples), {"saved_params_path": str(out_path), "skipped": True}
        except Exception:
            pass
        self._lazy_init()
        # Try to release previous XLA caches to reduce fragmentation between clients
        try:
            gc.collect()
            jax.clear_caches()
        except Exception:
            pass
        gpath = config.get("params_path")
        # Log round start for continuity across resume
        self._append_jsonl(
            {
                "event": "round_start",
                "ts": time.time(),
                "round": int(rnd),
                "client_id": int(self.client_id),
                "params_path": gpath,
            }
        )
        if not gpath:
            print(f"[DiskClient {self.client_id}] No params_path provided; skip")
            return [], 0, {"error": "no_params_path"}
        # Apply global parameters
        try:
            self.state = _apply_params_from(gpath, self.state)
        except Exception as e:
            print(f"[DiskClient {self.client_id}] Failed to apply params from {gpath}: {e}")
            return [], 0, {"error": str(e)}
        # Ensure placements follow compiled shardings
        try:
            self.state = jax.device_put(self.state, self.state_sharding)
            self.train_rng = jax.device_put(self.train_rng, self.replicated_sharding)
            self.state = dataclasses.replace(
                self.state, step=jax.device_put(self.state.step, self.replicated_sharding)
            )
        except Exception:
            pass

        # Train locally for local_steps with progress bar
        examples = 0
        losses = []
        vc_info = f"VC 1/{self.virtual_clients}" if self.virtual_clients > 1 else "VC 1/1"
        desc = f"[Client {self.client_id} | {vc_info}] Round {rnd} | Training"
        with tqdm(
            range(self.local_steps),
            desc=desc,
            leave=False,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        ) as pbar:
            for step_idx in pbar:
                raw = next(self.data_iter)
                batch = (_model.Observation.from_dict(raw), raw["actions"])
                with sharding.set_mesh(self.mesh):
                    self.state, info = self.ptrain_step(self.train_rng, self.state, batch)
                examples += self.cfg.batch_size
                try:
                    cur_loss = float(jax.device_get(info.get("loss", np.nan)))
                except Exception:
                    cur_loss = float("nan")
                if not np.isnan(cur_loss):
                    losses.append(cur_loss)
                avg_loss = float(np.mean(losses)) if losses else float("nan")
                pbar.set_postfix(
                    {
                        "loss": (f"{cur_loss:.4f}" if not np.isnan(cur_loss) else "nan"),
                        "avg": (f"{avg_loss:.4f}" if not np.isnan(avg_loss) else "nan"),
                        "ex": examples,
                    }
                )
                # Append JSON log per step (resume-safe append)
                self._append_jsonl(
                    {
                        "event": "step",
                        "ts": time.time(),
                        "round": int(rnd),
                        "client_id": int(self.client_id),
                        "step": int(step_idx),
                        "examples": int(examples),
                        "loss": (None if np.isnan(cur_loss) else float(cur_loss)),
                        "avg_loss": (None if np.isnan(avg_loss) else float(avg_loss)),
                    }
                )

        # Save params to disk and return path for server aggregation
        out_path = self.out_cache_dir / f"client_{self.client_id}" / f"round_{rnd:05d}" / "params.npz"
        path_str = _save_params_npz(out_path, self.state, dtype=self._store_dtype)
        # Write sidecar meta for resume-aware aggregation
        try:
            meta = {
                "round": int(rnd),
                "client_id": int(self.client_id),
                "examples": int(examples),
                "ts": time.time(),
            }
            with open(out_path.parent / "meta.json", "w", encoding="utf-8") as mf:
                json.dump(meta, mf, ensure_ascii=False)
        except Exception:
            pass
        print(f"[DiskClient {self.client_id}] Saved trained params -> {path_str}")
        # Log round-end summary
        self._append_jsonl(
            {
                "event": "round_end",
                "ts": time.time(),
                "round": int(rnd),
                "client_id": int(self.client_id),
                "examples": int(examples),
                "saved_params_path": path_str,
            }
        )
        # Post-fit cleanup: aggressively drop device references and caches so下一客户端能获得显存
        try:
            # Ensure pending work finished before deletion
            def _block(x):
                try:
                    return x.block_until_ready()
                except Exception:
                    return x
            if self.state is not None:
                jax.tree.map(_block, self.state)
            # Drop heavy references
            self.loader = None
            self.data_iter = None
            self.ptrain_step = None
            self.state = None
            self.state_sharding = None
            self.data_sharding = None
            self.replicated_sharding = None
            self.mesh = None
        except Exception:
            pass
        try:
            gc.collect(); jax.clear_caches()
        except Exception:
            pass
        return [], int(examples), {"saved_params_path": path_str}

    def get_parameters(self, config=None):
        # No direct tensor transmission; keep empty
        return []

    def evaluate(self, parameters, config):
        print(f"[DiskClient {self.client_id}] Skipping evaluation")
        return 0.0, 0, {"skipped": True}


def client_fn(context: Context) -> NumPyClient:
    config_name = context.run_config.get("config-name", "pi0_libero_0813_fl")
    total_clients = context.run_config.get("total-clients", 1)
    client_id = context.run_config.get("client-id", 0)
    # Prefer id from node_config when running multi-client simulations
    try:
        nc = getattr(context, "node_config", {}) or {}
        auto_cid = nc.get("node_id") or nc.get("cid") or nc.get("partition-id")
        if auto_cid is not None:
            client_id = int(str(auto_cid)) if str(auto_cid).isdigit() else client_id
    except Exception:
        pass
    local_steps = context.run_config.get("local-steps", 5)
    virtual_clients = context.run_config.get("virtual-clients", 1)
    batch_size = context.run_config.get("batch-size", None)
    num_workers = context.run_config.get("num-workers", 2)
    fsdp_devices = context.run_config.get("fsdp-devices", 1)
    out_cache_dir = context.run_config.get("cache-dir", "./cache/federated_disk")
    store_precision = context.run_config.get("store-precision", "fp16")

    client = OpenPIFlowerDiskClient(
        config_name=config_name,
        total_clients=total_clients,
        client_id=client_id,
        local_steps=local_steps,
        virtual_clients=virtual_clients,
        batch_size=batch_size,
        num_workers=num_workers,
    fsdp_devices=fsdp_devices,
    out_cache_dir=out_cache_dir,
    store_precision=store_precision,
    )
    return client.to_client()


app = ClientApp(client_fn=client_fn)


if __name__ == "__main__":
    print("OpenPI Disk-based Flower Client - use 'flwr run' to start")
