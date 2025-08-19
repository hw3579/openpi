import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "s3://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


@dataclasses.dataclass(frozen=True)
class PartialWeightLoader(WeightLoader):
    """只加载兼容的权重部分，跳过不兼容的动作相关层"""
    
    params_path: str
    # 要跳过的参数模式列表
    skip_patterns: tuple[str, ...] = ("action_in_proj", "action_out_proj", "action_expert")
    
    def load(self, params: at.Params) -> at.Params:
        """加载权重，自动跳过不兼容的层"""
        
        logger.info(f"使用部分权重加载器从 {self.params_path} 加载权重")
        logger.info(f"将跳过包含以下模式的层: {self.skip_patterns}")
        
        # 加载原始检查点
        try:
            loaded_params = _model.restore_params(
                download.maybe_download(self.params_path), 
                restore_type=np.ndarray
            )
        except Exception as e:
            logger.error(f"无法加载检查点: {e}")
            raise
        
        # 扁平化参数以便分析
        flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
        flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")
        
        # 构建兼容的参数字典
        result = {}
        skipped_keys = []
        loaded_keys = []
        
        for k, v_ref in flat_ref.items():
            # 检查是否应该跳过这个键
            should_skip = any(pattern in k for pattern in self.skip_patterns)
            
            if should_skip:
                # 跳过不兼容的层，使用参考参数的结构
                skipped_keys.append(k)
                result[k] = v_ref
                continue
            
            # 检查加载的参数中是否有这个键
            if k in flat_loaded:
                v_loaded = flat_loaded[k]
                
                # 检查v_ref是否是ShapeDtypeStruct
                if hasattr(v_ref, 'shape') and hasattr(v_ref, 'dtype') and not hasattr(v_ref, 'astype'):
                    # v_ref是ShapeDtypeStruct，检查形状和类型兼容性
                    if hasattr(v_loaded, 'shape') and v_loaded.shape == v_ref.shape:
                        # 形状匹配，转换数据类型后使用加载的权重
                        result[k] = v_loaded.astype(v_ref.dtype)
                        loaded_keys.append(k)
                    else:
                        # 形状不匹配，跳过并使用参考参数
                        logger.warning(f"跳过形状不匹配的参数: {k}, "
                                     f"预期 {v_ref.shape}, 得到 {getattr(v_loaded, 'shape', 'unknown')}")
                        skipped_keys.append(k)
                        result[k] = v_ref
                elif hasattr(v_loaded, 'shape') and hasattr(v_ref, 'shape'):
                    # 都是实际的数组
                    if v_loaded.shape == v_ref.shape:
                        # 形状匹配，加载权重并转换数据类型
                        result[k] = v_loaded.astype(v_ref.dtype)
                        loaded_keys.append(k)
                    else:
                        # 形状不匹配，跳过并使用参考参数
                        logger.warning(f"跳过形状不匹配的参数: {k}, "
                                     f"预期 {v_ref.shape}, 得到 {v_loaded.shape}")
                        skipped_keys.append(k)
                        result[k] = v_ref
                else:
                    # 没有形状信息，直接使用加载的权重
                    result[k] = v_loaded
                    loaded_keys.append(k)
            else:
                # 参数不存在于加载的权重中，使用参考参数
                skipped_keys.append(k)
                result[k] = v_ref
        
        # 记录统计信息
        logger.info(f"成功加载 {len(loaded_keys)} 个参数层")
        logger.info(f"跳过 {len(skipped_keys)} 个参数层")
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"加载的参数层: {loaded_keys}")
            logger.debug(f"跳过的参数层: {skipped_keys}")
        
        # 重构参数树并直接返回，不再使用_merge_params避免ShapeDtypeStruct问题
        result_params = flax.traverse_util.unflatten_dict(result, sep="/")
        
        # 手动处理LoRA权重
        flat_result = flax.traverse_util.flatten_dict(result_params, sep="/")
        pattern = re.compile(".*lora.*")
        for k in {k for k in flat_ref if pattern.fullmatch(k)}:
            if k not in flat_result:
                flat_result[k] = flat_ref[k]
        
        return flax.traverse_util.unflatten_dict(flat_result, sep="/")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype)

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")

