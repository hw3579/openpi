"""OpenPI Flower Server: A Flower / JAX app for federated learning."""

import os
from pympler import asizeof
import json
import gc

os.environ["RAY_BACKEND_LOG_LEVEL"] = "ERROR"

# 强制服务端（driver）CPU-only，必须在导入 jax 之前设置
# os.environ.setdefault("JAX_PLATFORMS", "cpu")         # 仅用 CPU 平台
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")     # 屏蔽 GPU
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # 禁止预分配


from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays, FitIns, EvaluateIns
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
import numpy as np
import jax
import jax.numpy as jnp
import dataclasses
from typing import List, Optional, Dict, Any, Tuple
from flax import traverse_util

# OpenPI imports
import openpi.training.config as _config
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.models.model as _model

# Import train helpers
import sys
import pathlib
_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import train as train_mod  # still available if needed elsewhere
import openpi.shared.nnx_utils as nnx_utils


def state_to_numpy_list(state: training_utils.TrainState) -> Tuple[List[np.ndarray], List[Tuple[str, ...]]]:
    """Convert training state to list of numpy arrays.

    For transport efficiency, compress floating-point arrays to float16.
    Non-floating arrays (e.g., ints/bools) are kept as-is. Any bfloat16 is
    converted to float16 (np doesn't support bf16 well).
    """
    pure = state.params.to_pure_dict()
    pure = jax.tree.map(lambda x: getattr(x, "value", x), pure)
    flat = traverse_util.flatten_dict(pure)
    keys = sorted(flat.keys())
    arrays: List[np.ndarray] = []
    
    for k in keys:
        val = jax.device_get(flat[k])
        np_arr = np.asarray(val)
        original_shape = np_arr.shape
        original_dtype = np_arr.dtype
        
        # Handle different dtypes carefully
        if str(original_dtype) == 'bfloat16':
            # Convert bfloat16 to float16 for transport
            print(f"[Server] Converting bfloat16 parameter {k} to float16")
            np_arr = np_arr.astype(np.float16)
        elif np_arr.dtype.kind in ("V", "O"):  # Void or Object dtype
            print(f"[Server] Warning: Parameter {k} has problematic dtype {np_arr.dtype}, converting to float32")
            np_arr = np.zeros(original_shape, dtype=np.float32)
        elif np_arr.dtype.kind == "c":  # complex
            print(f"[Server] Info: Converting complex parameter {k} to float32 (real part)")
            np_arr = np_arr.real.astype(np.float32)
        elif np.issubdtype(np_arr.dtype, np.floating):
            # Compress to float16 for transport
            if np_arr.dtype != np.float16:
                np_arr = np_arr.astype(np.float16)
        elif np.issubdtype(np_arr.dtype, np.integer):
            if np_arr.dtype not in (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64):
                np_arr = np_arr.astype(np.int32)
        
        # Ensure contiguous memory layout
        if not np_arr.flags.c_contiguous:
            np_arr = np.ascontiguousarray(np_arr)
        
        # Final validation
        if not np.all(np.isfinite(np_arr)) and np.issubdtype(np_arr.dtype, np.floating):
            print(f"[Server] Warning: Parameter {k} contains non-finite values")
            np_arr = np.where(np.isfinite(np_arr), np_arr, 0.0)
        
        arrays.append(np_arr)
    
    return arrays, keys


class OpenPIFedAvg(FedAvg):
    """Custom FedAvg strategy for OpenPI with server-side pretrained weight initialization."""

    def __init__(self, config_name: str = "pi0_libero_0813_fl", max_parallel_clients: int | None = None, agg_precision: str = "fp16", exit_after_fit: bool | None = None, release_after_fit: bool | None = None, **kwargs):
        super().__init__(**kwargs)
        self._global_params: Optional[List[np.ndarray]] = None
        self._config_name = config_name
        self._initialized = False
        self._param_keys: Optional[list[tuple[str, ...]]] = None
        self._max_parallel_clients = max_parallel_clients
        # 聚合精度：fp16 或 fp32（默认 fp16 降低内存）。
        self._agg_dtype = np.float16 if str(agg_precision).lower() == "fp16" else np.float32
        # 将这些标志下发给客户端，控制单轮完成后的内存回收/进程退出
        self._exit_after_fit = bool(exit_after_fit) if exit_after_fit is not None else False
        self._release_after_fit = bool(release_after_fit) if release_after_fit is not None else False

    def _initialize_pretrained_weights(self):
        """Initialize pretrained weights on server side by flat-loading numpy fp16.

        Avoids building the JAX model/state to keep RSS minimal. Only keeps
        a list of numpy float16 arrays and a parallel list of param keys.
        """
        if self._initialized:
            return
        print(f"[OpenPI Server] Initializing pretrained weights with config: {self._config_name}")
        cfg = _config.get_config(self._config_name)
        print(f"[OpenPI Server] Using weight loader: {type(cfg.weight_loader).__name__}")

        params_path = getattr(cfg.weight_loader, "params_path", None)
        skip_patterns = getattr(cfg.weight_loader, "skip_patterns", None)

        arrays: List[np.ndarray] = []
        keys: list[tuple[str, ...]] = []

        if params_path:
            from openpi.shared import download
            import re
            try:
                ckpt_path = download.maybe_download(params_path)
                loaded = _model.restore_params(ckpt_path, restore_type=np.ndarray, dtype=jnp.float16)
                flat = traverse_util.flatten_dict(loaded)
                compiled = [re.compile(p) for p in (skip_patterns or ())]
                for k, v in flat.items():
                    if compiled and any(p.search("/".join(k)) for p in compiled):
                        continue
                    np_arr = np.asarray(v, dtype=np.float16)
                    if not np_arr.flags.c_contiguous:
                        np_arr = np.ascontiguousarray(np_arr)
                    arrays.append(np_arr)
                    keys.append(k)
                order = sorted(range(len(keys)), key=lambda i: keys[i])
                self._global_params = [arrays[i] for i in order]
                self._param_keys = [keys[i] for i in order]
                total_params = sum(a.size for a in self._global_params)
                total_bytes = sum(a.nbytes for a in self._global_params)
                print(
                    f"[OpenPI Server] Loaded checkpoint (flat): {len(self._global_params)} tensors, "
                    f"{total_params:,} params, {total_bytes/1024/1024:.1f} MB (fp16 floats)"
                )
                # Release temporaries
                loaded = None
                flat = None
            finally:
                gc.collect()
                try:
                    jax.clear_caches()
                except Exception:
                    pass
        else:
            print("[OpenPI Server] Warning: weight_loader has no params_path; broadcasting empty params")
            self._global_params = []
            self._param_keys = None


        self._initialized = True

    def initialize_parameters(self, client_manager):
        """Initialize with pretrained parameters instead of empty parameters."""
        self._initialize_pretrained_weights()
        if self._global_params:
            print(f"[OpenPI Server] Broadcasting pretrained weights to clients")
            return ndarrays_to_parameters(self._global_params)
        else:
            print(f"[OpenPI Server] No pretrained weights available, sending empty parameters")
            return ndarrays_to_parameters([])

    def configure_fit(self, server_round: int, parameters, client_manager):
        """Configure fit with current global parameters."""
        print(f"[OpenPI Server] Round {server_round}: Configuring fit")
        
        # Sample clients for this round
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        selected = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients
        )
        # 限制并发广播的客户端数量，降低复制带来的内存峰值
        if self._max_parallel_clients is not None:
            if len(selected) > self._max_parallel_clients:
                print(f"[OpenPI Server] Capping parallel clients: {len(selected)} -> {self._max_parallel_clients}")
                selected = selected[: self._max_parallel_clients]
        
        # Send current global parameters to clients
        # 优先复用框架传入的 parameters（可能已在后端共享，避免重复序列化/复制）
        if parameters is not None and getattr(parameters, "tensors", None):
            params = parameters
            print(f"[OpenPI Server] Sending global model via provided Parameters object")
        elif self._global_params:
            params = ndarrays_to_parameters(self._global_params)
            total_bytes = sum(arr.nbytes for arr in self._global_params)
            print(f"[OpenPI Server] Sending global model: {total_bytes/1024/1024:.1f} MB")
        else:
            params = ndarrays_to_parameters([])
            print(f"[OpenPI Server] No global parameters yet")
        
        # 将扁平 keys 作为字符串下发，客户端按名合并，避免全量/零填充
        keys_str = None
        if getattr(self, "_param_keys", None):
            # 使用 "/" 连接键路径，列表再 JSON 编码
            keys_str = json.dumps(["/".join(k) for k in self._param_keys])
        config = {"server_round": server_round}
        if keys_str is not None:
            config["param_keys"] = keys_str
        # 将全局运行配置中的内存优化开关传递给客户端
        if self._release_after_fit:
            config["release-after-fit"] = True
        if self._exit_after_fit:
            config["exit-after-fit"] = True
        
        return [(client, FitIns(params, config)) for client in selected]

    def aggregate_fit(self, server_round: int, results, failures):
        """Aggregate client results using streaming weighted averaging to cut memory peak."""
        print(f"[OpenPI Server] Round {server_round}: Received {len(results)} results, {len(failures)} failures")

        import ipdb; ipdb.set_trace()  # Debug
        
        if failures:
            print(f"[OpenPI Server] Failures: {failures}")
        
        if not results:
            print("[OpenPI Server] No results to aggregate")
            return None, {}

        try:
            # Streaming aggregation: keep only one client's tensors in memory at a time
            print(f"[OpenPI Server] Aggregating {len(results)} clients (streaming)")
            sum_weights = 0.0
            acc_arrays: list[np.ndarray] | None = None

            for cid, fit_res in results:
                w = float(getattr(fit_res, "num_examples", 0) or 0)
                arrays = parameters_to_ndarrays(fit_res.parameters)
                if arrays:
                    if acc_arrays is None:
                        wv = np.array(w, dtype=self._agg_dtype)
                        acc_arrays = [a.astype(self._agg_dtype, copy=False) * wv for a in arrays]
                    else:
                        for i, a in enumerate(arrays):
                            wv = np.array(w, dtype=self._agg_dtype)
                            acc_arrays[i] += a.astype(self._agg_dtype, copy=False) * wv
                    sum_weights += w

            if acc_arrays is None or sum_weights == 0:
                print("[OpenPI Server] No arrays or zero total weight; skipping update")
                aggregated_params = None
            else:
                # Finalize average; cast back to fp16 to reduce memory
                # Finalize average using所选精度，再转回 fp16 作为全局权重以降低常驻内存
                sw = np.array(sum_weights, dtype=self._agg_dtype)
                avg_arrays = [ (a / sw).astype(np.float16) for a in acc_arrays ]
                # Free previous globals early to reduce peak
                self._global_params = None
                gc.collect()
                try:
                    jax.clear_caches()
                except Exception:
                    pass
                self._global_params = avg_arrays
                aggregated_params = ndarrays_to_parameters(avg_arrays)
                print(f"[OpenPI Server] Updated global model (streaming avg)")

            # Collect metrics
            metrics = {}
            losses = []
            accuracies = []
            
            for _, fit_res in results:
                if fit_res.metrics:
                    if 'loss' in fit_res.metrics:
                        losses.append(fit_res.metrics['loss'])
                    if 'accuracy' in fit_res.metrics:
                        accuracies.append(fit_res.metrics['accuracy'])
            
            if losses:
                metrics['avg_loss'] = float(np.mean(losses))
                print(f"[OpenPI Server] Round {server_round}: Average loss = {metrics['avg_loss']:.4f}")
            
            if accuracies:
                metrics['avg_accuracy'] = float(np.mean(accuracies))
                print(f"[OpenPI Server] Round {server_round}: Average accuracy = {metrics['avg_accuracy']:.4f}")
            

            import ipdb; ipdb.set_trace()  # Debug

            return aggregated_params, metrics
            
        except Exception as e:
            print(f"[OpenPI Server] Error in aggregate_fit: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return None, {"error": str(e)}

    def configure_evaluate(self, server_round: int, parameters, client_manager):
        """Skip evaluation for VLA models."""
        print(f"[OpenPI Server] Round {server_round}: Skipping evaluation (VLA model)")
        return []

    def aggregate_evaluate(self, server_round: int, results, failures):
        """Skip evaluation aggregation for VLA models."""
        print(f"[OpenPI Server] Round {server_round}: Skipping evaluation aggregation (VLA model)")
        return None, {}


def server_fn(context: Context) -> ServerAppComponents:
    """Create and configure the Flower server components."""
    
    # Read configuration from context
    config_name = context.run_config.get("config-name", "pi0_libero_0813_fl")
    num_rounds = context.run_config.get("num-server-rounds", 10)
    min_fit_clients = context.run_config.get("min-fit-clients", 1)
    min_available_clients = context.run_config.get("min-available-clients", 1)
    fraction_fit = context.run_config.get("fraction-fit", 1.0)
    max_parallel_clients = context.run_config.get("max-parallel-clients", None)
    release_after_fit = context.run_config.get("release-after-fit", False) or context.run_config.get("release_after_fit", False)
    exit_after_fit = context.run_config.get("exit-after-fit", False) or context.run_config.get("exit_after_fit", False)
    fraction_evaluate = context.run_config.get("fraction-evaluate", 1.0)
    
    print(f"[OpenPI Server] Configuration:")
    print(f"  - Config name: {config_name}")
    print(f"  - Rounds: {num_rounds}")
    print(f"  - Min fit clients: {min_fit_clients}")
    print(f"  - Min available clients: {min_available_clients}")
    print(f"  - Fraction fit: {fraction_fit}")
    print(f"  - Evaluation: DISABLED (VLA model)")
    
    # Create strategy with pretrained weight initialization
    strategy = OpenPIFedAvg(
        config_name=config_name,  # Pass config name for weight loading
        max_parallel_clients=max_parallel_clients,
        release_after_fit=release_after_fit,
        exit_after_fit=exit_after_fit,
        min_fit_clients=min_fit_clients,
        min_available_clients=min_available_clients,
        fraction_fit=fraction_fit,
        fraction_evaluate=0.0,  # Disable evaluation for VLA
        min_evaluate_clients=0,  # Disable evaluation for VLA
        initial_parameters=None,  # Will be set by initialize_parameters
    )
    
    # Create server configuration
    config = ServerConfig(num_rounds=num_rounds)
    
    print("[OpenPI Server] Server components created successfully")
    print("[OpenPI Server] Using Flower's automatic large model transmission")
    print("[OpenPI Server] Evaluation DISABLED for VLA model - training only")
    
    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)


if __name__ == "__main__":
    print("OpenPI Flower Server - Use 'flwr run' to start this app")
