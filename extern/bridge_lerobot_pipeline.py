import tensorflow_datasets as tfds
import torch
from torch.utils.data import IterableDataset, DataLoader
import jax.numpy as jnp
import numpy as np
import logging
import math
import tensorflow as tf

logger = logging.getLogger("openpi.bridge_data")

class BridgeLeRobotIterableDataset(IterableDataset):
    """
    IterableDataset that streams BridgeV2 from TFDS and on-the-fly
    converts each step into LeRobot format, using only the main camera.
    """

    def __init__(
        self,
        dataset_name: str = "bridge_dataset",
        split: str = "train",
        data_dir: str = "/home/jiaqi/tfds_datasets",
        shuffle_episodes: bool = True,
        shuffle_buffer_size: int = 100,  # 新增：缓冲区大小
        cache_size: int = 10,  # 新增：缓存的episode数量
    ):
        """
        Args:
            dataset_name: TFDS 数据集名，比如 'bridge_v2'
            split:          'train' / 'validation' / 'test'
            data_dir:       TFDS 本地缓存路径（可选）
            shuffle_episodes: 是否在迭代前对 episode 顺序做随机打乱
            shuffle_buffer_size: 使用TF的shuffle缓冲区大小，较大的值洗牌更均匀但内存占用更高
            cache_size: 内存中保留的episode数量，用于加速重复访问
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.data_dir = data_dir
        self.shuffle_episodes = shuffle_episodes
        self.shuffle_buffer_size = shuffle_buffer_size
        self.cache_size = cache_size
        
        # 记录数据集信息
        logger.info(f"初始化Bridge数据集: {dataset_name}, split={split}, dir={data_dir}")
        logger.info(f"流式加载设置: shuffle_buffer={shuffle_buffer_size}, cache_size={cache_size}")
    
    def _create_dataset(self):
        """创建和配置TF数据集，实现高效的流式处理"""
        try:
            # 加载数据集
            ds = tfds.load(
                name=self.dataset_name,
                split=self.split,
                data_dir=self.data_dir,
                as_supervised=False,
            )
            
            # 使用TF原生shuffle，而不是加载所有数据
            if self.shuffle_episodes:
                ds = ds.shuffle(
                    buffer_size=self.shuffle_buffer_size,
                    reshuffle_each_iteration=True,
                    seed=42
                )
            
            # 预取以提高性能
            ds = ds.prefetch(tf.data.AUTOTUNE)
            
            return ds
        except Exception as e:
            logger.error(f"创建数据集失败: {e}")
            raise

    def __iter__(self):
        """返回迭代器，按需流式加载数据"""
        try:
            # 创建数据集
            ds = self._create_dataset()
            
            # 转为NumPy迭代器
            it = tfds.as_numpy(ds)
            
            # 设置LRU缓存用于保存最近使用的episodes
            from collections import OrderedDict
            episode_cache = OrderedDict()
            
            # 遍历episodes
            for i, episode in enumerate(it):
                # 缓存管理：如果缓存太大，移除最早添加的项
                if len(episode_cache) >= self.cache_size:
                    episode_cache.popitem(last=False)
                
                # 将当前episode加入缓存
                episode_id = f"ep_{i}"
                episode_cache[episode_id] = episode

                # import pdb; pdb.set_trace() 
                
                # 处理当前episode中的每个step
                # instruction = episode["language_instruction"].decode("utf-8")
                for step in episode["steps"]:
                    # 生成单个样本
                    yield {
                        "image":    torch.from_numpy(step["observation"]["image_0"]),
                        "state":    torch.from_numpy(step["observation"]["state"]),
                        "actions":  torch.from_numpy(step["action"]),
                        "task":     step["language_instruction"].decode("utf-8"),  
                        # "task":     instruction,
                    }
        except Exception as e:
            logger.error(f"迭代数据集时出错: {e}")
            raise

    def get_sample(self):
        """获取单个样本，用于检查数据格式"""
        # 创建一个单次使用的迭代器
        ds = self._create_dataset().take(1)
        it = tfds.as_numpy(ds)
        episode = next(iter(it))
        
        instruction = episode["language_instruction"].decode("utf-8")
        step = episode["steps"][0]  # 只取第一个step
        
        return {
            "image":    torch.from_numpy(step["observation"]["image_0"]),
            "state":    torch.from_numpy(step["observation"]["state"]),
            "actions":  torch.from_numpy(step["action"]),
            "task":     instruction,
        }


# 改进的DataLoader工厂函数
def make_dataloader(
    dataset_name: str = "bridge_v2",
    split: str = "train",
    data_dir: str = None,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle_episodes: bool = True,
    pin_memory: bool = True,
    shuffle_buffer_size: int = 100,
    prefetch_factor: int = 2,
):
    """
    返回一个流式DataLoader，高效处理大型数据集
    """
    ds = BridgeLeRobotIterableDataset(
        dataset_name=dataset_name,
        split=split,
        data_dir=data_dir,
        shuffle_episodes=shuffle_episodes,
        shuffle_buffer_size=shuffle_buffer_size,
    )
    
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=True if num_workers > 0 else False,
        collate_fn=lambda batch: {
            "image": torch.stack([b["image"] for b in batch]),
            "state": torch.stack([b["state"] for b in batch]),
            "actions": torch.stack([b["actions"] for b in batch]),
            "task": [b["task"] for b in batch],
        },
    )

# 优化的OpenPI适配层
class OpenPIBridgeDataset:
    """
    OpenPI兼容的数据集适配器，高效流式处理
    """
    def __init__(
        self, 
        dataset_name: str = "bridge_v2",
        split: str = "train",
        data_dir: str = None,
        shuffle_episodes: bool = True,
        shuffle_buffer_size: int = 100,
    ):
        self.raw_dataset = BridgeLeRobotIterableDataset(
            dataset_name=dataset_name,
            split=split,
            data_dir=data_dir,
            shuffle_episodes=shuffle_episodes,
            shuffle_buffer_size=shuffle_buffer_size,
        )
        self.dataset_info = {
            "name": dataset_name,
            "split": split,
            "dir": data_dir,
        }
        
    def __iter__(self):
        """流式转换数据格式"""
        for item in self.raw_dataset:
            yield self._convert_to_openpi_format(item)
    
    def _convert_to_openpi_format(self, item):
        """将Bridge样本转换为OpenPI格式"""
        return {
            "observation": {
                "images": {
                    "top": item["image"].numpy(),
                },
                "state": item["state"].numpy(),
            },
            "action": item["actions"].numpy(),
            "prompt": item["task"],
        }
    
    def get_sample(self):
        """获取单个样本"""
        raw_sample = self.raw_dataset.get_sample()
        return self._convert_to_openpi_format(raw_sample)
    
    def __len__(self):
        """估计数据集大小"""
        return 10000  # 仅为估计值


# 高效计算归一化统计的函数
def compute_normalization_stats(
    dataset_name: str = "bridge_v2",
    split: str = "train",
    data_dir: str = None,
    num_samples: int = 1000,
    batch_size: int = 32,  # 新增：批处理大小
):
    """
    高效计算数据集的归一化统计，使用批处理以减少内存使用
    """
    logger.info(f"计算数据集 {dataset_name}/{split} 的归一化统计...")
    
    # 使用流式数据集
    dataset = BridgeLeRobotIterableDataset(
        dataset_name=dataset_name,
        split=split,
        data_dir=data_dir,
        shuffle_episodes=True,
        shuffle_buffer_size=min(1000, num_samples),  # 动态调整缓冲区大小
    )
    
    # 使用在线算法计算统计数据
    # 参考: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    stats = {}
    for key in ["state", "actions"]:
        stats[key] = {
            "count": 0,
            "mean": None,
            "M2": None,  # 二阶矩，用于计算方差
        }
    
    # 收集统计数据
    samples_processed = 0
    try:
        for i, sample in enumerate(dataset):
            if samples_processed >= num_samples:
                break
                
            for key in ["state", "actions"]:
                value = sample[key].numpy()
                
                # 首次初始化
                if stats[key]["count"] == 0:
                    stats[key]["mean"] = np.zeros_like(value)
                    stats[key]["M2"] = np.zeros_like(value)
                
                # Welford在线算法
                stats[key]["count"] += 1
                delta = value - stats[key]["mean"]
                stats[key]["mean"] += delta / stats[key]["count"]
                delta2 = value - stats[key]["mean"]
                stats[key]["M2"] += delta * delta2
            
            samples_processed += 1
            
            # 定期打印进度
            if i % 100 == 0:
                logger.info(f"已处理 {samples_processed}/{num_samples} 个样本")
    
    except Exception as e:
        logger.error(f"计算统计数据时出错: {e}")
        if samples_processed == 0:
            raise
    
    # 计算最终结果
    results = {}
    for key in stats:
        if stats[key]["count"] > 0:
            mean = stats[key]["mean"]
            # 计算方差和标准差
            var = stats[key]["M2"] / stats[key]["count"]
            std = np.sqrt(np.maximum(var, 1e-6))
            
            results[key] = {
                "mean": mean,
                "std": std
            }
    
    logger.info(f"共处理了 {samples_processed} 个样本")
    logger.info(f"归一化统计计算完成")
    return results