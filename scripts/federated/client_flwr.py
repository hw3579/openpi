"""
Simplified Flower NumPyClient for OpenPI training.

- Sequential virtual-client simulation on a single GPU.
- Conservative environment flags to reduce OOM from cuDNN/XLA memory usage.
- Basic parameter transmission with optional fixed-size chunking for large models.
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import traceback
from typing import Iterator, Tuple, List

# # Conservative defaults to reduce GPU OOM from cuDNN autotune/workspace and XLA prealloc.
# # Only set if user hasn't provided them in the environment.
# os.environ.setdefault("JAX_CUDNN_AUTOTUNE_DEFAULT", "0")  # avoid trying large-workspace algos
# os.environ.setdefault("JAX_ASYNC_DISPATCH", "false")      # fail early in-step, clearer OOM
# os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
# # Keep existing XLA_FLAGS and append a safe autotune level if not already present.
# if "--xla_gpu_autotune_level" not in os.environ.get("XLA_FLAGS", ""):
#     os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_gpu_autotune_level=0").strip()

import flwr as fl
import jax
import jax.numpy as jnp
import numpy as np
from flax import traverse_util
import flax.nnx as nnx
from tqdm import tqdm  # type: ignore

# from fixed_size_chunked_transmission import FixedSizeChunkManager

import openpi.training.config as _config
import openpi.training.sharding as sharding
import openpi.training.data_loader as dl
import openpi.training.utils as training_utils
import openpi.models.model as _model

# Import train helpers
import sys
import pathlib
_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import train as train_mod  # type: ignore

print("version of flwr:", fl.__version__)


class IIDSubsetDataset:
    def __init__(self, base, indices: List[int]):
        self._base = base
        self._indices = indices
    def __getitem__(self, idx: int):
        return self._base[self._indices[idx]]
    def __len__(self) -> int:
        return len(self._indices)


def build_iid_indices(n: int, total_clients: int, client_id: int, seed: int) -> List[int]:
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    chunk = max(1, n // max(1, total_clients))
    start = min(client_id * chunk, n)
    end = n if client_id >= total_clients - 1 else min((client_id + 1) * chunk, n)
    return perm[start:end].tolist()


def state_to_numpy_list(state: training_utils.TrainState) -> Tuple[List[np.ndarray], List[Tuple[str, ...]]]:
    """Convert training state to list of numpy arrays with proper sanitization."""
    pure = state.params.to_pure_dict()
    pure = jax.tree.map(lambda x: getattr(x, "value", x), pure)
    flat = traverse_util.flatten_dict(pure)
    keys = sorted(flat.keys())
    arrays: List[np.ndarray] = []
    
    for i, k in enumerate(keys):
        # Get value from JAX device
        val = jax.device_get(flat[k])
        np_arr = np.asarray(val)
        
        # Debug problematic parameters
        if np_arr.dtype.kind == "V" or np_arr.dtype == object:
            print(f"Debug: Parameter {i} (key={k}) has problematic dtype: {np_arr.dtype}")
            print(f"  Shape: {np_arr.shape}, Size: {np_arr.size}")
            if hasattr(np_arr.dtype, 'names') and np_arr.dtype.names:
                print(f"  Structured fields: {np_arr.dtype.names}")
        
        # Ensure proper dtype and format for serialization
        if np_arr.dtype.kind == "c":  # complex
            np_arr = np_arr.real.astype(np.float32)
        elif np.issubdtype(np_arr.dtype, np.floating):
            if np_arr.dtype not in (np.float32, np.float64):
                np_arr = np_arr.astype(np.float32)
        elif np.issubdtype(np_arr.dtype, np.integer):
            if np_arr.dtype not in (np.int32, np.int64):
                np_arr = np_arr.astype(np.int32)
        
        # Ensure contiguous layout
        if not np_arr.flags.c_contiguous:
            np_arr = np.ascontiguousarray(np_arr)
        
        # Check for non-finite values
        if np.issubdtype(np_arr.dtype, np.floating):
            if not np.all(np.isfinite(np_arr)):
                print(f"Warning: Parameter {k} contains non-finite values")
                np_arr = np.where(np.isfinite(np_arr), np_arr, 0.0)
        
        arrays.append(np_arr)
    
    return arrays, keys


def numpy_list_to_state(arrays: List[np.ndarray], state: training_utils.TrainState) -> training_utils.TrainState:
    cur_pure = state.params.to_pure_dict()
    cur_pure = jax.tree.map(lambda x: getattr(x, "value", x), cur_pure)
    cur_flat = traverse_util.flatten_dict(cur_pure)
    keys = sorted(cur_flat.keys())
    if len(arrays) != len(keys):
        raise ValueError(f"Parameter length mismatch: got {len(arrays)} vs expected {len(keys)}")
    rebuilt_flat = {}
    for k, a in zip(keys, arrays):
        ref = cur_flat[k]
        a_jnp = jnp.asarray(a, dtype=ref.dtype)
        if ref.shape != a_jnp.shape:
            raise ValueError(f"Shape mismatch for {k}: got {a_jnp.shape}, expected {ref.shape}")
        rebuilt_flat[k] = a_jnp
    nested = traverse_util.unflatten_dict(rebuilt_flat)
    model = nnx.merge(state.model_def, state.params)
    graphdef, nnx_state = nnx.split(model)
    nnx_state.replace_by_pure_dict(nested)
    model = nnx.merge(graphdef, nnx_state)
    new_params = nnx.state(model)
    return dataclasses.replace(state, params=new_params)


def sanitize_and_validate_parameters(arrays: List[np.ndarray]) -> List[np.ndarray]:
    """Ensure arrays are valid for network transmission."""
    out: List[np.ndarray] = []
    for i, arr in enumerate(arrays):
        try:
            np_arr = np.asarray(arr)
            
            # Check for problematic dtypes
            if np_arr.dtype == object:
                print(f"Warning: Parameter {i} has object dtype, converting to float32")
                np_arr = np_arr.astype(np.float32)
            elif np_arr.dtype.kind in ("U", "S"):  # Unicode/string
                print(f"Warning: Parameter {i} has string dtype, converting to float32")
                np_arr = np.zeros_like(np_arr, dtype=np.float32)
            elif np_arr.dtype.kind == "V":  # Void (structured)
                print(f"Warning: Parameter {i} has structured dtype {np_arr.dtype}, converting to float32")
                # For structured arrays, try to extract numeric fields or flatten
                if np_arr.dtype.names:  # Named fields
                    # Extract first numeric field if available
                    numeric_fields = [name for name in np_arr.dtype.names 
                                    if np.issubdtype(np_arr.dtype.fields[name][0], np.number)]
                    if numeric_fields:
                        np_arr = np_arr[numeric_fields[0]].astype(np.float32)
                    else:
                        # No numeric fields, create zeros with same shape
                        np_arr = np.zeros(np_arr.shape, dtype=np.float32)
                else:
                    # No named fields, try to view as bytes and reshape
                    byte_view = np_arr.view(np.uint8)
                    float_size = int(np.ceil(byte_view.size / 4) * 4)  # Round up to float32 boundary
                    padded = np.zeros(float_size, dtype=np.uint8)
                    padded[:byte_view.size] = byte_view.ravel()
                    np_arr = padded.view(np.float32)
            
            # Handle complex numbers by taking real part
            if np_arr.dtype.kind == "c":
                np_arr = np_arr.real.astype(np.float32)
            
            # Normalize floating point types to supported ones
            if np.issubdtype(np_arr.dtype, np.floating):
                if np_arr.dtype not in (np.float32, np.float64):
                    np_arr = np_arr.astype(np.float32)
            elif np.issubdtype(np_arr.dtype, np.integer):
                # Convert to appropriate int type
                if np_arr.dtype not in (np.int32, np.int64):
                    np_arr = np_arr.astype(np.int32)
            elif np.issubdtype(np_arr.dtype, np.bool_):
                # Keep bool as is
                pass
            else:
                # Unknown dtype, convert to float32
                print(f"Warning: Parameter {i} has unknown dtype {np_arr.dtype}, converting to float32")
                np_arr = np_arr.astype(np.float32)
            
            # Ensure contiguous memory layout
            if not np_arr.flags.c_contiguous:
                np_arr = np.ascontiguousarray(np_arr)
            
            # Ensure it's actually a numpy array
            if type(np_arr) != np.ndarray:
                np_arr = np.array(np_arr)
            
            # Final validation - check for non-finite values
            if np.issubdtype(np_arr.dtype, np.floating):
                if not np.all(np.isfinite(np_arr)):
                    print(f"Warning: Parameter {i} contains non-finite values (inf/nan)")
                    # Replace inf/nan with zeros
                    np_arr = np.where(np.isfinite(np_arr), np_arr, 0.0)
            
            out.append(np_arr)
            
        except Exception as e:
            print(f"Warning: Failed to sanitize parameter {i} ({type(e).__name__}: {e}), using zeros")
            # Create a zero array with reasonable shape
            try:
                if hasattr(arr, 'shape'):
                    np_arr = np.zeros(arr.shape, dtype=np.float32)
                else:
                    np_arr = np.zeros((1,), dtype=np.float32)
                out.append(np_arr)
            except:
                # Last resort: single zero
                out.append(np.array([0.0], dtype=np.float32))
    
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--server-address", type=str, default="127.0.0.1:8080")
    p.add_argument("--config-name", type=str, default="pi0_libero_0813_fl")
    p.add_argument("--exp-name", type=str, default="flwr_client")
    p.add_argument("--total-clients", type=int, default=1)
    p.add_argument("--client-id", type=int, default=0)
    p.add_argument("--virtual-clients", type=int, default=1)
    p.add_argument("--local-steps", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fsdp-devices", type=int, default=1)
    p.add_argument("--disable-wandb", action="store_true", default=True)
    return p.parse_args()


class OpenPIFlowerClient(fl.client.NumPyClient):
    """Simplified Flower client using Flower 1.20's automatic large model transmission."""
    
    def __init__(self, cfg: _config.TrainConfig, total_clients: int, client_id: int, local_steps: int, *, num_workers: int, virtual_clients: int = 1):
        self.cfg = cfg
        self.total_clients = total_clients
        self.client_id = client_id
        self.local_steps = local_steps
        self.num_workers = num_workers
        self.virtual_clients = max(1, int(virtual_clients))
        self.initialized = False
        
        # Round tracking
        self._round_counter = 0
        
        # Fixed-size chunk manager (for future use)
        # self._fixed_size_chunk_manager = FixedSizeChunkManager()

    def _lazy_init(self):
        if self.initialized:
            return
        # Mesh/sharding
        self.mesh = sharding.make_mesh(self.cfg.fsdp_devices)
        self.data_sharding = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        self.replicated_sharding = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec())

        # Dataset
        self.data_config = self.cfg.data.create(self.cfg.assets_dirs, self.cfg.model)
        self.base_dataset = dl.create_dataset(self.data_config, self.cfg.model)
        self.n_samples = len(self.base_dataset)
        indices = build_iid_indices(self.n_samples, self.total_clients, self.client_id, seed=self.cfg.seed)
        subset = IIDSubsetDataset(self.base_dataset, indices)
        dataset = dl.transform_dataset(subset, self.data_config, skip_norm_stats=False)

        local_bs = self.cfg.batch_size // jax.device_count()
        if self.cfg.batch_size % jax.device_count() != 0:
            raise ValueError(f"Batch size {self.cfg.batch_size} must be divisible by device count {jax.device_count()}")
        logging.info(
            "[FL client] devices=%s, device_count=%d, global_batch=%d, local_batch=%d",
            ",".join([d.device_kind for d in jax.devices()]),
            jax.device_count(),
            self.cfg.batch_size,
            local_bs,
        )
        self.loader = dl.TorchDataLoader(
            dataset,
            local_batch_size=local_bs,
            sharding=self.data_sharding,
            shuffle=True,
            num_batches=None,
            num_workers=self.num_workers,
            seed=self.cfg.seed + self.client_id,
        )
        self.data_iter = iter(self.loader)

        # Train state and jit
        init_rng = jax.random.key(self.cfg.seed)
        train_rng = jax.random.key(self.cfg.seed + 10_000 + self.client_id)
        self.state, self.state_sharding = train_mod.init_train_state(self.cfg, init_rng, self.mesh, resume=False)
        self.ptrain_step = jax.jit(
            lambda rng, state, batch: train_mod.train_step(self.cfg, rng, state, batch),
            in_shardings=(self.replicated_sharding, self.state_sharding, self.data_sharding),
            out_shardings=(self.state_sharding, self.replicated_sharding),
            donate_argnums=(1,),
        )
        self.train_rng = train_rng
        self.initialized = True

    # Flower API
    def get_parameters(self, config):
        """Return empty list as server controls parameter distribution."""
        return []

    def set_parameters(self, parameters):
        """Apply received parameters to model state."""
        self._lazy_init()
        if not parameters:  # Empty parameters
            return
            
        current_arrays, _ = state_to_numpy_list(self.state)
        
        if len(parameters) == len(current_arrays):
            # Full parameter update - normal case
            print(f"[FL client] Applying full parameter update ({len(parameters)} parameters)")
            self.state = numpy_list_to_state(parameters, self.state)
            
        elif len(parameters) < len(current_arrays):
            # Partial parameter update (e.g., for testing)
            print(f"[FL client] Partial update: received {len(parameters)}, expected {len(current_arrays)}")
            print(f"[FL client] Updating only the first {len(parameters)} parameters")
            updated_arrays = current_arrays.copy()
            for i, param in enumerate(parameters):
                if i < len(updated_arrays) and updated_arrays[i].shape == param.shape:
                    updated_arrays[i] = param
            self.state = numpy_list_to_state(updated_arrays, self.state)
            
        else:
            print(f"[FL client] Warning: received more parameters ({len(parameters)}) than expected ({len(current_arrays)})")
            # Truncate to expected length
            self.state = numpy_list_to_state(parameters[:len(current_arrays)], self.state)
    
    def _iter_for_client(self, logical_client_id: int) -> Iterator:
        indices = build_iid_indices(self.n_samples, self.virtual_clients, logical_client_id, seed=self.cfg.seed)
        subset = IIDSubsetDataset(self.base_dataset, indices)
        dataset = dl.transform_dataset(subset, self.data_config, skip_norm_stats=False)
        local_bs = self.cfg.batch_size // jax.device_count()
        loader = dl.TorchDataLoader(
            dataset,
            local_batch_size=local_bs,
            sharding=self.data_sharding,
            shuffle=True,
            num_batches=None,
            num_workers=self.num_workers,
            seed=self.cfg.seed + logical_client_id,
        )
        return iter(loader)

    def _local_train_once(self) -> Tuple[List[np.ndarray], int, float]:
        if self.virtual_clients <= 1:
            losses = []
            examples = 0
            pbar = tqdm(range(self.local_steps), desc="[train] steps", leave=False, position=0)
            for _ in pbar:
                raw = next(self.data_iter)
                batch = (_model.Observation.from_dict(raw), raw["actions"])  # match train.py
                self.state, info = self.ptrain_step(self.train_rng, self.state, batch)
                cur_loss = float(jax.device_get(info["loss"]))
                losses.append(cur_loss)
                try:
                    pbar.set_postfix({"loss": f"{cur_loss:.4f}"})
                except Exception:
                    pass
                examples += self.cfg.batch_size
            try:
                pbar.close()
            except Exception:
                pass
            arrays, _ = state_to_numpy_list(self.state)
            return arrays, examples, (float(np.mean(losses)) if losses else 0.0)


        # 拿一份“基准权重”用于 reset
        base_arrays, _ = state_to_numpy_list(self.state)

        # 在线累加器（和参数同形状）
        agg_arrays = [np.zeros_like(a, dtype=np.float32) for a in base_arrays]
        total_weight = 0.0
        shard_losses: List[float] = []

        with tqdm(range(self.virtual_clients), desc=f"[Client {self.client_id}] Virtual clients") as pbar_v:
            for v in pbar_v:
                # reset 到基准权重（注意：不要保留多份；用已有的 base_arrays 重建 state）
                self.state = numpy_list_to_state(base_arrays, self.state)

                data_iter = self._iter_for_client(v)
                losses = []
                for _ in range(self.local_steps):
                    raw = next(data_iter)
                    batch = (_model.Observation.from_dict(raw), raw["actions"])
                    self.state, info = self.ptrain_step(self.train_rng, self.state, batch)
                    losses.append(float(jax.device_get(info["loss"])))

                # 取该虚拟 client 的更新并加权累加到 agg
                arrays, _ = state_to_numpy_list(self.state)
                w = float(self.cfg.batch_size * self.local_steps)
                inv_total = w  # 只需用于显示
                total_weight += w

                # 在线加权：agg = agg + arrays * (w/总权重)，但为避免每次重缩放，我们做“先累加，最后统一 / total_weight”
                for p in range(len(arrays)):
                    # 尽量避免新建中间副本
                    agg_arrays[p] += arrays[p].astype(np.float32, copy=False) * (w)

                shard_losses.append(float(np.mean(losses)) if losses else 0.0)
                pbar_v.set_postfix({"vc": f"{v+1}/{self.virtual_clients}", "loss": f"{shard_losses[-1]:.4f}"})

        # 统一归一化
        if total_weight > 0:
            for p in range(len(agg_arrays)):
                agg_arrays[p] /= total_weight

        # 更新回 state
        if agg_arrays:
            self.state = numpy_list_to_state(agg_arrays, self.state)

        return agg_arrays, int(total_weight), float(np.mean(shard_losses)) if shard_losses else 0.0
        # # Multi-virtual: sequential shards; reuse the same state object to avoid double residency
        # base_arrays, _ = state_to_numpy_list(self.state)
        # shard_arrays_list: List[List[np.ndarray]] = []
        # shard_weights: List[int] = []
        # shard_losses: List[float] = []
        # pbar_v = tqdm(range(self.virtual_clients), desc="[train] virtual", leave=True, position=0)
        # for v in pbar_v:
        #     logging.info(f"[FL client] start virtual shard {v+1}/{self.virtual_clients}")
        #     # Reset params to base snapshot in-place (new buffers, but we reuse the same holder self.state)
        #     self.state = numpy_list_to_state(base_arrays, self.state)
        #     data_iter = self._iter_for_client(v)
        #     losses = []
        #     pbar_s = tqdm(range(self.local_steps), desc=f"[vc {v+1}] steps", leave=False, position=1)
        #     for _ in pbar_s:
        #         raw = next(data_iter)
        #         batch = (_model.Observation.from_dict(raw), raw["actions"])  # match train.py
        #         self.state, info = self.ptrain_step(self.train_rng, self.state, batch)
        #         cur_loss = float(jax.device_get(info["loss"]))
        #         losses.append(cur_loss)
        #         try:
        #             pbar_s.set_postfix({"loss": f"{cur_loss:.4f}"})
        #         except Exception:
        #             pass
        #     try:
        #         pbar_s.close()
        #     except Exception:
        #         pass
        #     arrays, _ = state_to_numpy_list(self.state)
        #     shard_arrays_list.append(arrays)
        #     shard_weights.append(self.cfg.batch_size * self.local_steps)
        #     shard_losses.append(float(np.mean(losses)) if losses else 0.0)
        # try:
        #     pbar_v.close()
        # except Exception:
        #     pass

        # total_weight = float(sum(shard_weights))
        # num_params = len(shard_arrays_list[0]) if shard_arrays_list else 0
        # agg_arrays: List[np.ndarray] = []
        # for p in range(num_params):
        #     acc = None
        #     for s_idx, arrays in enumerate(shard_arrays_list):
        #         w = shard_weights[s_idx] / total_weight
        #         contrib = arrays[p] * w
        #         acc = contrib if acc is None else acc + contrib
        #     agg_arrays.append(acc)

        # if agg_arrays:
        #     self.state = numpy_list_to_state(agg_arrays, self.state)
        # return agg_arrays, int(total_weight), (float(np.mean(shard_losses)) if shard_losses else 0.0)

    def fit(self, parameters, config):
        """Train locally and return updated parameters."""
        try:
            self._lazy_init()
            self._round_counter += 1
            
            # Apply received parameters if any
            if parameters:
                print(f"[FL client] Round {self._round_counter}: Received {len(parameters)} parameters for download")
                self.set_parameters(parameters)
            
            # Train locally
            print(f"[FL client] Round {self._round_counter}: Starting local training with {self.virtual_clients} virtual clients, {self.local_steps} steps each")
            trained_arrays, examples, mean_loss = self._local_train_once()
            trained_arrays = sanitize_and_validate_parameters(trained_arrays)
            
            print(f"[FL client] Round {self._round_counter}: Training complete. Loss: {mean_loss:.4f}, Examples: {examples}")
            print(f"[FL client] Round {self._round_counter}: Returning {len(trained_arrays)} parameters")
            
            # Log parameter info for debugging
            total_params = sum(arr.size for arr in trained_arrays)
            total_bytes = sum(arr.nbytes for arr in trained_arrays)
            print(f"[FL client] Round {self._round_counter}: Total parameters: {total_params:,}, Total size: {total_bytes/1024/1024:.1f} MB")
            
            # Check for potential serialization issues
            for i, arr in enumerate(trained_arrays[:]):  # Check all params
                print(f"[FL client] Round {self._round_counter}: Param {i}: shape={arr.shape}, dtype={arr.dtype}, size={arr.nbytes/1024/1024:.1f}MB")
                if not np.all(np.isfinite(arr)) and np.issubdtype(arr.dtype, np.floating):
                    print(f"[FL client] Round {self._round_counter}: WARNING: Param {i} contains non-finite values!")
            
            # Always return full parameters to maintain federated learning semantics
            # Even if transmission may fail due to size limits, we maintain FL integrity
            if total_bytes > 1024 * 1024 * 1024:  # 1GB
                print(f"[FL client] Round {self._round_counter}: Model size ({total_bytes/1024/1024:.1f} MB) exceeds 1GB limit")
                print(f"[FL client] Round {self._round_counter}: Attempting full parameter transmission")
            
            return trained_arrays, int(examples), {"loss": float(mean_loss), "round": self._round_counter}
            
        except Exception as e:
            print(f"[FL client] Round {self._round_counter}: Error in fit(): {type(e).__name__}: {e}")
            traceback.print_exc()
            # Return empty arrays to avoid crash
            return [], 0, {"loss": 0.0, "error": str(e), "round": self._round_counter}

    def evaluate(self, parameters, config):
        """Evaluate model performance."""
        self._lazy_init()
        if parameters:
            self.set_parameters(parameters)
        
        # Quick evaluation on one batch
        raw = next(self.data_iter)
        batch = (_model.Observation.from_dict(raw), raw["actions"])
        _, info = self.ptrain_step(self.train_rng, self.state, batch)
        loss = float(jax.device_get(info["loss"]))
        
        return loss, self.cfg.batch_size, {"loss": loss}


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    cfg = _config.get_config(args.config_name)
    cfg = dataclasses.replace(
        cfg,
        exp_name=args.exp_name,
        wandb_enabled=(False if args.disable_wandb else cfg.wandb_enabled),
        fsdp_devices=args.fsdp_devices,
        batch_size=(args.batch_size if args.batch_size is not None else cfg.batch_size),
        num_workers=args.num_workers,
    )
    
    client = OpenPIFlowerClient(
        cfg=cfg,
        total_clients=args.total_clients,
        client_id=args.client_id,
        local_steps=args.local_steps,
        num_workers=args.num_workers,
        virtual_clients=args.virtual_clients,
    )
    
    # Start Flower client - Flower 1.20 handles large models automatically
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client,
    )


if __name__ == "__main__":
    main()
