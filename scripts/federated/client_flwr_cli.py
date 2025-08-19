"""OpenPI Flower Client: A Flower / JAX app for federated learning."""

import os
import json
from pympler import asizeof

# 仅客户端：按需分配显存，避免预占导致的再分配 OOM；使用 platform 分配器减少大块占用
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# 移除可能遗留的内存比例限制
# os.environ.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)
# 可选：降低编译并行度以进一步降低峰值显存
# os.environ.setdefault("XLA_FLAGS", "--xla_gpu_force_compilation_parallelism=1 --xla_gpu_autotune_level=1")
# 若被分配到 GPU，但继承了 JAX_PLATFORMS=cpu，移除之
if os.environ.get("CUDA_VISIBLE_DEVICES") and os.environ.get("JAX_PLATFORMS") == "cpu":
    os.environ.pop("JAX_PLATFORMS", None)

import jax
import jax.numpy as jnp
import numpy as np
import dataclasses
import logging
from typing import List, Tuple, Iterator
from tqdm import tqdm

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flax import traverse_util
import flax.nnx as nnx

# OpenPI imports
import openpi.training.config as _config
import openpi.training.sharding as sharding
import openpi.training.data_loader as dl
import openpi.training.utils as training_utils
import openpi.models.model as _model
import openpi.training.weight_loaders as weight_loaders
import openpi.training.checkpoints as _checkpoints

# Import train helpers
import sys
import pathlib
_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import train as train_mod


class IIDSubsetDataset:
    """IID subset of dataset for federated learning."""
    def __init__(self, base, indices: List[int]):
        self._base = base
        self._indices = indices
    
    def __getitem__(self, idx: int):
        return self._base[self._indices[idx]]
    
    def __len__(self) -> int:
        return len(self._indices)


def build_iid_indices(n: int, total_clients: int, client_id: int, seed: int) -> List[int]:
    """Build IID indices for client data split."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    chunk = max(1, n // max(1, total_clients))
    start = min(client_id * chunk, n)
    end = n if client_id >= total_clients - 1 else min((client_id + 1) * chunk, n)
    return perm[start:end].tolist()


def state_to_numpy_list(state: training_utils.TrainState) -> Tuple[List[np.ndarray], List[Tuple[str, ...]]]:
    """Convert training state to list of numpy arrays.

    For transport efficiency, compress float tensors to float16. Non-float
    tensors are kept as-is. This matches the server's transport format.
    """
    pure = state.params.to_pure_dict()
    pure = jax.tree.map(lambda x: getattr(x, "value", x), pure)
    flat = traverse_util.flatten_dict(pure)
    keys = sorted(flat.keys())
    arrays: List[np.ndarray] = []
    
    for k in keys:
        ref = flat[k]
        # 对浮点参数先在设备上转换为 fp16 再拉到主机，避免产生大的 fp32 主机副本
        try:
            if hasattr(ref, "dtype") and jnp.issubdtype(ref.dtype, jnp.floating):
                host_val = jax.device_get(jnp.asarray(ref, dtype=jnp.float16))
            else:
                host_val = jax.device_get(ref)
        except Exception:
            host_val = jax.device_get(ref)

        np_arr = np.asarray(host_val)
        original_shape = np_arr.shape
        original_dtype = np_arr.dtype
        
        # Handle different dtypes carefully（此时浮点一般已是 float16）
        if np_arr.dtype.kind in ("V", "O"):  # Void or Object dtype
            print(f"Warning: Parameter {k} has problematic dtype {np_arr.dtype}, converting to float32")
            # This should be rare, but handle it carefully
            np_arr = np.zeros(original_shape, dtype=np.float32)
        elif np_arr.dtype.kind == "c":  # complex
            print(f"Info: Converting complex parameter {k} to float32 (real part)")
            np_arr = np_arr.real.astype(np.float32)
        elif np.issubdtype(np_arr.dtype, np.floating):
            # Ensure float16 for transport
            if np_arr.dtype != np.float16:
                np_arr = np_arr.astype(np.float16)
        elif np.issubdtype(np_arr.dtype, np.integer):
            # Keep integer types, but ensure they're supported
            if np_arr.dtype not in (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64):
                np_arr = np_arr.astype(np.int32)
        
        # Verify shape is preserved
        if np_arr.shape != original_shape:
            print(f"Error: Shape mismatch for {k}: original {original_shape} != converted {np_arr.shape}")
            # Try to reshape back
            try:
                np_arr = np_arr.reshape(original_shape)
            except ValueError as e:
                print(f"Failed to reshape {k}: {e}")
                # Last resort: create zeros with correct shape
                np_arr = np.zeros(original_shape, dtype=np.float32)
        
        # Ensure contiguous memory layout
        if not np_arr.flags.c_contiguous:
            np_arr = np.ascontiguousarray(np_arr)
        
        # Final validation
        if not np.all(np.isfinite(np_arr)) and np.issubdtype(np_arr.dtype, np.floating):
            print(f"Warning: Parameter {k} contains non-finite values")
            np_arr = np.where(np.isfinite(np_arr), np_arr, 0.0)
        
        arrays.append(np_arr)
    
    return arrays, keys


def numpy_list_to_state(arrays: List[np.ndarray], state: training_utils.TrainState) -> training_utils.TrainState:
    """Convert numpy arrays back to training state."""
    cur_pure = state.params.to_pure_dict()
    cur_pure = jax.tree.map(lambda x: getattr(x, "value", x), cur_pure)
    cur_flat = traverse_util.flatten_dict(cur_pure)
    keys = sorted(cur_flat.keys())
    
    if len(arrays) != len(keys):
        raise ValueError(f"Parameter length mismatch: got {len(arrays)} vs expected {len(keys)}")
    
    rebuilt_flat = {}
    for k, a in zip(keys, arrays):
        ref = cur_flat[k]
        # Normal path: array may arrive as float16 (transport). Cast to ref dtype.
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


def numpy_list_with_keys_to_state(
    arrays: List[np.ndarray], server_keys: List[Tuple[str, ...]] | List[str], state: training_utils.TrainState
) -> training_utils.TrainState:
    """Apply numpy arrays to state.params using provided flat keys from server.

    - server_keys may be a list of tuples or strings with '/' separators.
    - Arrays are cast to the reference dtype and validated for shape.
    - Only matching keys are updated; unknown keys are skipped with a warning.
    """
    # Normalize server keys to tuple[str, ...]
    norm_keys: List[Tuple[str, ...]] = []
    for k in server_keys:
        if isinstance(k, tuple):
            norm_keys.append(tuple(k))
        elif isinstance(k, list):
            norm_keys.append(tuple(k))
        elif isinstance(k, str):
            norm_keys.append(tuple([p for p in k.split("/") if p]))
        else:
            raise ValueError(f"Unsupported key type from server: {type(k)} -> {k}")

    cur_pure = state.params.to_pure_dict()
    cur_pure = jax.tree.map(lambda x: getattr(x, "value", x), cur_pure)
    cur_flat = traverse_util.flatten_dict(cur_pure)

    updated_flat = dict(cur_flat)
    applied = 0
    for k, a in zip(norm_keys, arrays):
        if k not in cur_flat:
            # Skip unknown keys silently but track
            # print(f"[Client] Unknown key from server, skipping: {k}")
            continue
        ref = cur_flat[k]
        a_jnp = jnp.asarray(a, dtype=ref.dtype)
        if ref.shape != a_jnp.shape:
            # Shape mismatch; skip with warning
            print(f"[Client] Shape mismatch for {k}: got {a_jnp.shape}, expected {ref.shape}; skipping")
            continue
        updated_flat[k] = a_jnp
        applied += 1

    if applied == 0:
        print("[Client] Warning: No parameters applied from server keys; falling back to existing state")
        return state

    nested = traverse_util.unflatten_dict(updated_flat)
    model = nnx.merge(state.model_def, state.params)
    graphdef, nnx_state = nnx.split(model)
    nnx_state.replace_by_pure_dict(nested)
    model = nnx.merge(graphdef, nnx_state)
    new_params = nnx.state(model)
    return dataclasses.replace(state, params=new_params)


# Define Flower Client
class OpenPIFlowerClient(NumPyClient):
    """OpenPI Flower client using new CLI architecture."""

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
    pretrained_trainstate_dir: str | None = None,
    ) -> None:
        # Configuration - always skip weight loading as server provides pretrained weights
        self.cfg = _config.get_config(config_name)
        self.cfg = dataclasses.replace(
            self.cfg,
            exp_name=f"flwr_client_{client_id}",
            wandb_enabled=False,  # Disable wandb for FL
            fsdp_devices=fsdp_devices,
            batch_size=(batch_size if batch_size is not None else self.cfg.batch_size),
            num_workers=num_workers,
            weight_loader=weight_loaders.NoOpWeightLoader(),  # Always skip weight loading
        )

        self.total_clients = total_clients
        self.client_id = client_id
        self.local_steps = local_steps
        self.virtual_clients = max(1, int(virtual_clients))
        self.initialized = False
        self._round_counter = 0
        self._num_params = None
        self._pretrained_trainstate_dir = pretrained_trainstate_dir

        # Create client display prefix
        self.client_prefix = f"🤖 Client-{self.client_id:02d}"
        if self.total_clients > 1:
            self.client_prefix += f"/{self.total_clients:02d}"

        print(f"[OpenPI Client {self.client_id}] Initialized with config: {config_name}")
        print(f"[OpenPI Client {self.client_id}] Weight loading disabled - will receive pretrained weights from server")
        print(
            f"[OpenPI Client {self.client_id}] Virtual clients: {self.virtual_clients}, "
            f"Local steps: {self.local_steps}"
        )
    
    def _lazy_init(self):
        """Initialize training components on first use."""
        if self.initialized:
            return
            
        print(f"[OpenPI Client {self.client_id}] Initializing training components...")
        
        # Mesh/sharding setup
        self.mesh = sharding.make_mesh(self.cfg.fsdp_devices)
        self.data_sharding = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        self.replicated_sharding = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec())

        # Dataset setup
        self.data_config = self.cfg.data.create(self.cfg.assets_dirs, self.cfg.model)
        self.base_dataset = dl.create_dataset(self.data_config, self.cfg.model)
        self.n_samples = len(self.base_dataset)
        
        # Create IID data split for this client
        indices = build_iid_indices(self.n_samples, self.total_clients, self.client_id, seed=self.cfg.seed)
        subset = IIDSubsetDataset(self.base_dataset, indices)
        dataset = dl.transform_dataset(subset, self.data_config, skip_norm_stats=False)

        # Data loader
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

        # Training state and JIT compilation
        init_rng = jax.random.key(self.cfg.seed)
        train_rng = jax.random.key(self.cfg.seed + 10_000 + self.client_id)

        if self._pretrained_trainstate_dir:
            print(f"[OpenPI Client {self.client_id}] Restoring pretrained TrainState (opt_state/EMA) from {self._pretrained_trainstate_dir}")
            # 仅生成形状与分片信息
            train_state_shape, self.state_sharding = train_mod.init_train_state(
                self.cfg, init_rng, self.mesh, resume=True
            )
            # 构建 checkpoint manager 并恢复
            mngr = _checkpoints.ocp.CheckpointManager(
                pathlib.Path(self._pretrained_trainstate_dir).resolve(),
                item_handlers={
                    "assets": _checkpoints.CallbackHandler(),
                    "train_state": _checkpoints.ocp.PyTreeCheckpointHandler(),
                    "params": _checkpoints.ocp.PyTreeCheckpointHandler(),
                },
                options=_checkpoints.ocp.CheckpointManagerOptions(
                    max_to_keep=1,
                    create=False,
                    single_host_load_and_broadcast=True,
                    enable_background_delete=True,
                ),
            )
            # 恢复到模板形状（包含优化器与 EMA）
            self.state = _checkpoints.restore_state(mngr, train_state_shape, data_loader=None)
        else:
            print(f"[OpenPI Client {self.client_id}] Initializing fresh TrainState (opt_state/EMA) locally")
            # 在 CPU 上禁用 JIT 初始化，避免显存峰值
            cpu_devs = jax.devices("cpu")
            cpu0 = cpu_devs[0] if cpu_devs else None
            if cpu0 is not None:
                with jax.disable_jit(), jax.default_device(cpu0):
                    self.state, self.state_sharding = train_mod.init_train_state(
                        self.cfg, init_rng, self.mesh, resume=False
                    )
            else:
                self.state, self.state_sharding = train_mod.init_train_state(
                    self.cfg, init_rng, self.mesh, resume=False
                )

        self.ptrain_step = jax.jit(
            lambda rng, state, batch: train_mod.train_step(self.cfg, rng, state, batch),
            in_shardings=(self.replicated_sharding, self.state_sharding, self.data_sharding),
            out_shardings=(self.state_sharding, self.replicated_sharding),
            donate_argnums=(1,),
        )
        # 推迟 GPU 放置到 fit 阶段（应用完服务端参数后），避免重复分配显存
        self.train_rng = train_rng

        # 仅统计键数量（不触发 device_get），用于参数长度校验
        try:
            pure = self.state.params.to_pure_dict()
            pure = jax.tree.map(lambda x: getattr(x, "value", x), pure)
            flat = traverse_util.flatten_dict(pure)
            self._num_params = len(flat)
        except Exception:
            self._num_params = None

        self.initialized = True

        print(f"[OpenPI Client {self.client_id}] Initialization complete")
        print(f"[OpenPI Client {self.client_id}] Dataset size: {len(subset)} samples")
        print(f"[OpenPI Client {self.client_id}] Batch size: {self.cfg.batch_size} (local: {local_bs})")

    def _iter_for_client(self, logical_client_id: int) -> Iterator:
        """Create data iterator for a specific virtual client."""
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
            num_workers=self.cfg.num_workers,
            seed=self.cfg.seed + logical_client_id,
        )
        return iter(loader)

    def _local_train(self, server_round: int = 0) -> Tuple[List[np.ndarray], int, float]:
        """Perform local training with virtual clients."""
        if self.virtual_clients <= 1:
            # Single virtual client training
            losses = []
            examples = 0
            
            def _ensure_data_sharding(x):
                # 仅在需要时放置到设备，避免重复拷贝
                try:
                    if isinstance(x, jax.Array):
                        return x if x.sharding == self.data_sharding else jax.device_put(x, self.data_sharding)
                    return jax.device_put(x, self.data_sharding) if hasattr(x, "shape") else x
                except Exception:
                    return x
            
            with tqdm(
                range(self.local_steps), 
                desc=f"{self.client_prefix} | Round {server_round} | Training", 
                leave=False,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
            ) as pbar:
                for step in pbar:
                    raw = next(self.data_iter)
                    batch = (_model.Observation.from_dict(raw), raw["actions"])
                    # 仅在需要时放置 batch 到 data_sharding
                    batch = jax.tree.map(_ensure_data_sharding, batch)
                    with sharding.set_mesh(self.mesh):
                        self.state, info = self.ptrain_step(self.train_rng, self.state, batch)
                    
                    cur_loss = float(jax.device_get(info["loss"]))
                    losses.append(cur_loss)
                    examples += self.cfg.batch_size
                    
                    # Enhanced progress display
                    avg_loss = np.mean(losses)
                    pbar.set_postfix({
                        "loss": f"{cur_loss:.4f}",
                        "avg_loss": f"{avg_loss:.4f}",
                        "examples": examples
                    })
            
            arrays, _ = state_to_numpy_list(self.state)
            return arrays, examples, float(np.mean(losses)) if losses else 0.0
        
        # Multi-virtual client training with aggregation
        base_arrays, _ = state_to_numpy_list(self.state)
        shard_arrays_list: List[List[np.ndarray]] = []
        shard_weights: List[int] = []
        shard_losses: List[float] = []

        with tqdm(
            range(self.virtual_clients),
            desc=f"{self.client_prefix} | Round {server_round} | Virtual Clients",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        ) as pbar_v:
            for v in pbar_v:
                # Reset to base state
                self.state = numpy_list_to_state(base_arrays, self.state)
                self.state = jax.device_put(self.state, self.state_sharding)
                try:
                    self.state = dataclasses.replace(
                        self.state, step=jax.device_put(self.state.step, self.replicated_sharding)
                    )
                except Exception:
                    pass
                data_iter = self._iter_for_client(v)
                losses = []

                # Inner training loop for each virtual client
                with tqdm(
                    range(self.local_steps),
                    desc=f"  └─ VC-{v+1} Training",
                    leave=False,
                    bar_format="    {l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}",
                ) as pbar_inner:
                    for step in pbar_inner:
                        raw = next(data_iter)
                        batch = (_model.Observation.from_dict(raw), raw["actions"])
                        batch = jax.tree.map(_ensure_data_sharding, batch)
                        with sharding.set_mesh(self.mesh):
                            self.state, info = self.ptrain_step(self.train_rng, self.state, batch)

                        cur_loss = float(jax.device_get(info["loss"]))
                        losses.append(cur_loss)

                        # Update inner progress
                        pbar_inner.set_postfix({"loss": f"{cur_loss:.4f}"})

                arrays, _ = state_to_numpy_list(self.state)
                shard_arrays_list.append(arrays)
                shard_weights.append(self.cfg.batch_size * self.local_steps)
                shard_losses.append(float(np.mean(losses)) if losses else 0.0)

                # Update virtual client progress
                avg_loss = np.mean(shard_losses)
                pbar_v.set_postfix(
                    {"vc": f"{v+1}/{self.virtual_clients}", "loss": f"{shard_losses[-1]:.4f}", "avg_loss": f"{avg_loss:.4f}"}
                )

        # Aggregate virtual client results
        total_weight = float(sum(shard_weights))
        num_params = len(shard_arrays_list[0]) if shard_arrays_list else 0
        agg_arrays: List[np.ndarray] = []

        for p in range(num_params):
            acc = None
            for s_idx, arrays in enumerate(shard_arrays_list):
                w = shard_weights[s_idx] / total_weight
                contrib = arrays[p] * w
                acc = contrib if acc is None else acc + contrib
            agg_arrays.append(acc)

        if agg_arrays:
            self.state = numpy_list_to_state(agg_arrays, self.state)
            self.state = jax.device_put(self.state, self.state_sharding)
            try:
                self.state = dataclasses.replace(self.state, step=jax.device_put(self.state.step, self.replicated_sharding))
            except Exception:
                pass

        return agg_arrays, int(total_weight), float(np.mean(shard_losses)) if shard_losses else 0.0

    def fit(self, parameters, config):
        """Train the model locally and return updated parameters."""
        try:
            self._lazy_init()
            import ipdb; ipdb.set_trace()  # Debug
            self._round_counter += 1
            

            server_round = config.get("server_round", self._round_counter)
            print(f"[OpenPI Client {self.client_id}] Round {server_round}: Starting fit")
            
            # Apply received parameters from server (should always have them after first round)
            if parameters:
                print(f"[OpenPI Client {self.client_id}] Applying {len(parameters)} received parameters from server")
                # Prefer applying by keys if provided by server
                server_keys_json = config.get("param_keys")
                if server_keys_json:
                    try:
                        server_keys = json.loads(server_keys_json)
                        self.state = numpy_list_with_keys_to_state(parameters, server_keys, self.state)
                        applied_by_keys = True
                    except Exception as e:
                        print(f"[OpenPI Client {self.client_id}] Failed to apply by keys: {e}; falling back to order")
                        applied_by_keys = False
                else:
                    applied_by_keys = False

                if not applied_by_keys:
                    expected = self._num_params
                    if expected is None or len(parameters) == expected:
                        self.state = numpy_list_to_state(parameters, self.state)
                    else:
                        print(
                            f"[OpenPI Client {self.client_id}] Parameter count mismatch: received {len(parameters)}, expected {expected}"
                        )
                        raise ValueError(
                            f"Parameter count mismatch: server sent {len(parameters)}, client expects {expected}"
                        )
                print(f"[OpenPI Client {self.client_id}] Successfully applied server parameters")
            else:
                print(f"[OpenPI Client {self.client_id}] Warning: No parameters received from server")
            
            # 训练前确保 state/rng 放置到期望分片（在应用完服务端参数之后再执行，避免重复分配）
            try:
                self.state = jax.device_put(self.state, self.state_sharding)
                self.train_rng = jax.device_put(self.train_rng, self.replicated_sharding)
                self.state = dataclasses.replace(
                    self.state, step=jax.device_put(self.state.step, self.replicated_sharding)
                )
            except Exception:
                pass
            # Local training with round info
            print(f"[OpenPI Client {self.client_id}] Starting local training ({self.virtual_clients} virtual clients, {self.local_steps} steps)")
            trained_arrays, examples, mean_loss = self._local_train(server_round)
            
            # Log training results
            total_params = sum(arr.size for arr in trained_arrays)
            total_bytes = sum(arr.nbytes for arr in trained_arrays)
            
            print(f"[OpenPI Client {self.client_id}] Round {server_round} complete:")
            print(f"  - Loss: {mean_loss:.4f}")
            print(f"  - Examples: {examples}")
            print(f"  - Parameters: {total_params:,} ({total_bytes/1024/1024:.1f} MB)")

            # 直接返回训练后的参数数组（state_to_numpy_list 已统一为 float16/连续内存）
            out_arrays = trained_arrays

            # 可选：单轮后释放内存（在 local-simulation 下可避免驻留峰值）
            if config.get("release-after-fit") or config.get("release_after_fit"):
                try:
                    # 释放大对象引用，让 GC 能回收
                    self.data_iter = None
                    self.loader = None
                    self.base_dataset = None
                    self.data_config = None
                    self.mesh = None
                    self.state_sharding = None
                    self.data_sharding = None
                    self.replicated_sharding = None
                    self.ptrain_step = None
                    self.state = None
                except Exception:
                    pass
                try:
                    jax.clear_caches()
                except Exception:
                    pass
                import gc as _gc
                _gc.collect()
                print(f"[OpenPI Client {self.client_id}] Released local state after fit to reduce memory")

            # 可选：单轮后强制退出进程，激进回收（仿真下有效）。
            # 注意：该行为会终止当前 Ray actor，下一客户端会在新进程中启动。
            if config.get("exit-after-fit") or config.get("exit_after_fit"):
                try:
                    import threading, time, os as _os
                    def _delayed_exit():
                        time.sleep(1.0)  # 确保 RPC 返回完成
                        _os._exit(0)
                    threading.Thread(target=_delayed_exit, daemon=True).start()
                    print(f"[OpenPI Client {self.client_id}] Scheduled process exit after fit (1s)")
                except Exception:
                    pass

            return out_arrays, int(examples), {"loss": float(mean_loss), "round": server_round}
            
        except Exception as e:
            print(f"[OpenPI Client {self.client_id}] Error in fit: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return [], 0, {"loss": 0.0, "error": str(e)}

    def get_parameters(self, config=None):
        """Return empty list; server broadcasts initial parameters. Avoids early device_get."""
        return []

    def evaluate(self, parameters, config):
        """Skip evaluation for VLA models."""
        print(f"[OpenPI Client {self.client_id}] Skipping evaluation (VLA model)")
        return 0.0, 0, {"skipped": True}


def client_fn(context: Context) -> NumPyClient:
    """Create and configure the Flower client."""
    
    # Read configuration from context
    config_name = context.run_config.get("config-name", "pi0_libero_0813_fl")
    total_clients = context.run_config.get("total-clients", 1)
    client_id = context.run_config.get("client-id", 0)
    # Prefer dynamic client identity from node_config if available
    try:
        nc = getattr(context, "node_config", None)
        if isinstance(nc, dict):
            for k in [
                "partition-id",
                "node-id",
                "node_id",
                "client-id",
                "client_id",
                "cid",
            ]:
                if k in nc:
                    client_id = int(nc[k])
                    print(f"[OpenPI Client] Using client-id from node_config[{k}] = {client_id}")
                    break
    except Exception:
        pass
    local_steps = context.run_config.get("local-steps", 5)
    virtual_clients = context.run_config.get("virtual-clients", 1)
    batch_size = context.run_config.get("batch-size", None)
    num_workers = context.run_config.get("num-workers", 2)
    fsdp_devices = context.run_config.get("fsdp-devices", 1)
    
    print(f"[OpenPI Client] Configuration:")
    print(f"  - Config: {config_name}")
    print(f"  - Client ID: {client_id} (out of {total_clients} total clients)")
    print(f"  - Total clients: {total_clients}")
    print(f"  - Local steps: {local_steps}")
    print(f"  - Virtual clients: {virtual_clients}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - FSDP devices: {fsdp_devices}")
    
    # Enhanced client identification
    client_prefix = f"🤖 Client-{client_id:02d}"
    if total_clients > 1:
        client_prefix += f"/{total_clients:02d}"
    
    # Create and return client instance
    # 预训练 train_state 目录（可选）
    pretrained_trainstate_dir = context.run_config.get("pretrained-train-state", None) or context.run_config.get("pretrained_train_state", None) or context.run_config.get("pretrained_dir", None)

    client = OpenPIFlowerClient(
        config_name=config_name,
        total_clients=total_clients,
        client_id=client_id,
        local_steps=local_steps,
        virtual_clients=virtual_clients,
        batch_size=batch_size,
        num_workers=num_workers,
        fsdp_devices=fsdp_devices,
        pretrained_trainstate_dir=pretrained_trainstate_dir,
    )
    
    return client.to_client()


# Flower ClientApp
app = ClientApp(client_fn=client_fn)


if __name__ == "__main__":
    print("OpenPI Flower Client - Use 'flwr run' to start this app")
    print("Example: flwr run . --run-config client-id=0 total-clients=2 local-steps=10")
