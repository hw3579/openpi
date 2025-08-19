import h5py
import torch
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import jax.numpy as jnp
import logging
import os
import glob
from typing import Optional, Dict, Any, List
import cv2
from torch.utils.data import Dataset as TorchDataset
import jax

logger = logging.getLogger("openpi.franka_data")

class FrankaH5IterableDataset(IterableDataset):
    """
    IterableDataset that loads Franka H5 datasets and converts them to OpenPI format
    """

    def __init__(
        self,
        data_dir: str = "./franka_datasets/dataset",
        split: str = "train",
        train_ratio: float = 0.8,
        image_key: str = "color_image",  # 选择使用哪个图像
        target_image_size: tuple = (224, 224),
        shuffle_episodes: bool = True,
        use_jax: bool = True,
        action_type: str = "joint_position",  # 选择动作类型
        cache_size: int = 10,
        worker_init_seed: Optional[int] = None,  # 新增：工作进程初始化种子
    ):
        """
        Args:
            data_dir: H5文件所在目录
            split: 'train' 或 'val'
            train_ratio: 训练集比例
            image_key: 使用的图像类型 ('color_image', 'depth_image', 'zed_image', 'thermal_processed')
            target_image_size: 目标图像尺寸
            shuffle_episodes: 是否打乱episode顺序
            use_jax: 是否使用JAX数组
            action_type: 动作类型 ('joint_position', 'cartesian_position', 'target_cartesian_position')
            cache_size: 缓存的episode数量
            worker_init_seed: 工作进程初始化种子
        """
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.train_ratio = train_ratio
        self.image_key = image_key
        self.target_image_size = target_image_size
        self.shuffle_episodes = shuffle_episodes
        self.use_jax = use_jax
        self.action_type = action_type
        self.cache_size = cache_size
        self.worker_init_seed = worker_init_seed
        
        # 获取所有H5文件
        self.h5_files = self._get_h5_files()
        
        # 根据split分割文件列表
        self.episode_files = self._split_files()
        
        logger.info(f"初始化Franka H5数据集: {len(self.episode_files)} 个episodes")
        logger.info(f"数据目录: {data_dir}, split: {split}")
        logger.info(f"图像类型: {image_key}, 动作类型: {action_type}")
        logger.info(f"数组类型: {'JAX' if use_jax else 'PyTorch'}")
    
    def _get_h5_files(self) -> List[str]:
        """获取所有H5文件路径"""
        pattern = os.path.join(self.data_dir, "*.h5")
        files = glob.glob(pattern)
        
        if not files:
            raise FileNotFoundError(f"在目录 {self.data_dir} 中未找到H5文件")
        
        files.sort()  # 确保一致的顺序
        return files
    
    def _split_files(self) -> List[str]:
        """根据train_ratio分割文件"""
        total_files = len(self.h5_files)
        train_count = int(total_files * self.train_ratio)
        
        if self.split == "train":
            return self.h5_files[:train_count]
        else:  # val
            return self.h5_files[train_count:]
    
    def _convert_array(self, array):
        """根据配置转换数组类型 - 避免在worker进程中使用JAX"""
        # 如果在worker进程中，避免使用JAX以防止冲突
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and self.use_jax:
            # 在worker进程中，先返回numpy数组
            # JAX转换将在主进程的collate_fn中进行
            return array
        elif self.use_jax:
            return jnp.array(array)
        else:
            return torch.from_numpy(array)
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """调整图像尺寸"""
        if image.shape[:2] != self.target_image_size:
            # 使用OpenCV进行高质量缩放
            image = cv2.resize(
                image, 
                self.target_image_size, 
                interpolation=cv2.INTER_LINEAR
            )
        return image
    
    def _extract_task_description(self, filename: str) -> str:
        """从文件名提取任务描述"""
        # 从文件名中提取任务描述
        basename = os.path.basename(filename)
        task_name = basename.replace('.h5', '').replace('_', ' ')
        
        # 移除数字后缀（如果有的话）
        import re
        task_name = re.sub(r'_\d+$', '', task_name)
        
        return task_name
    
    def _get_state_vector(self, h5_file, step_idx: int) -> np.ndarray:
        """提取机器人状态向量"""
        # 使用关节位置作为状态
        joint_positions = h5_file['observation/robot_state/joint_positions'][step_idx]
        gripper_position = h5_file['observation/robot_state/gripper_position'][step_idx]
        
        # 组合关节位置和夹爪位置 (7+1=8维)
        state = np.concatenate([joint_positions, [gripper_position]])
        return state.astype(np.float32)
    
    def _get_action_vector(self, h5_file, step_idx: int) -> np.ndarray:
        """提取动作向量"""
        if self.action_type == "joint_position":
            joint_action = h5_file['action/joint_position'][step_idx]
            gripper_action = h5_file['action/gripper_position'][step_idx]
            # 组合关节动作和夹爪动作 (7+1=8维)
            action = np.concatenate([joint_action, [gripper_action]])
        elif self.action_type == "cartesian_position":
            cartesian_action = h5_file['action/cartesian_position'][step_idx]
            gripper_action = h5_file['action/gripper_position'][step_idx]
            # 组合笛卡尔坐标和夹爪动作 (6+1=7维)
            action = np.concatenate([cartesian_action, [gripper_action]])
        elif self.action_type == "target_cartesian_position":
            cartesian_action = h5_file['action/target_cartesian_position'][step_idx]
            gripper_action = h5_file['action/target_gripper_position'][step_idx].astype(np.float32)
            # 组合目标笛卡尔坐标和目标夹爪动作 (6+1=7维)
            action = np.concatenate([cartesian_action, [gripper_action]])
        else:
            raise ValueError(f"不支持的动作类型: {self.action_type}")
        
        return action.astype(np.float32)

    def __iter__(self):
        """返回迭代器，按需流式加载数据"""
        # 获取worker信息
        worker_info = torch.utils.data.get_worker_info()
        
        # 如果使用多进程，为每个worker分配不同的文件子集
        if worker_info is not None:
            # 多进程模式：每个worker处理文件的一个子集
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            
            # 为当前worker分配文件
            worker_files = self.episode_files[worker_id::num_workers]
            
            # 设置worker特定的随机种子
            if self.worker_init_seed is not None:
                np.random.seed(self.worker_init_seed + worker_id)
        else:
            # 单进程模式：使用所有文件
            worker_files = self.episode_files
        
        # 如果需要打乱，打乱当前worker的文件列表
        if self.shuffle_episodes:
            worker_files = worker_files.copy()
            np.random.shuffle(worker_files)
        
        # 遍历当前worker的文件
        for episode_file in worker_files:
            try:
                task_description = self._extract_task_description(episode_file)
                
                with h5py.File(episode_file, 'r') as h5_file:
                    # 获取episode长度
                    episode_length = h5_file[f'observation/{self.image_key}'].shape[0]
                    
                    # 遍历episode中的每个step
                    for step_idx in range(episode_length):
                        # 提取图像
                        image = h5_file[f'observation/{self.image_key}'][step_idx]
                        
                        # 调整图像尺寸
                        image = self._resize_image(image)
                        
                        # 归一化图像到[0,1]
                        if image.dtype == np.uint8:
                            image = image.astype(np.float32) / 255.0
                        
                        # 提取状态
                        state = self._get_state_vector(h5_file, step_idx)
                        
                        # 提取动作
                        action = self._get_action_vector(h5_file, step_idx)
                        
                        # 生成样本（在worker进程中保持为numpy数组）
                        yield {
                            "image": self._convert_array(image),
                            "state": self._convert_array(state),
                            "actions": self._convert_array(action),
                            "task": task_description,
                        }
                        
            except Exception as e:
                logger.error(f"处理文件 {episode_file} 时出错: {e}")
                continue

    def get_sample(self):
        """获取单个样本，用于检查数据格式"""
        if not self.episode_files:
            raise ValueError("没有可用的episode文件")
        
        # 使用第一个文件获取样本
        episode_file = self.episode_files[0]
        task_description = self._extract_task_description(episode_file)
        
        with h5py.File(episode_file, 'r') as h5_file:
            # 获取第一个step的数据
            image = h5_file[f'observation/{self.image_key}'][0]
            image = self._resize_image(image)
            
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            
            state = self._get_state_vector(h5_file, 0)
            action = self._get_action_vector(h5_file, 0)
            
            return {
                "image": self._convert_array(image),
                "state": self._convert_array(state),
                "actions": self._convert_array(action),
                "task": task_description,
            }


def worker_init_fn(worker_id):
    """DataLoader worker初始化函数"""
    # 设置每个worker的随机种子
    np.random.seed(torch.initial_seed() % 2**32 + worker_id)


class FrankaH5Dataset(TorchDataset):
    """
    用于OpenPI训练的Franka H5数据集，输出结构与FakeDataset严格对齐。

    返回值格式：
      {
        'base_0_rgb': jnp.ndarray,   # shape (H, W, 3)
        'wrist_0_rgb': jnp.ndarray,  # shape (H, W, 3)
        'state': jnp.ndarray,        # shape (N_state,)
        'actions': jnp.ndarray       # shape (Horizon, N_action)
      }
    """
    def __init__(self,
                 model_config: "_model.BaseModelConfig",
                 data_config: "_config.DataConfig"):
        self.model_config = model_config
        self.data_config = data_config
        assert hasattr(data_config, 'data_dir'), "data_config必须包含data_dir属性"
        self.data_dir = data_config.data_dir
        self.horizon = model_config.action_horizon

        self._episodes = self._get_episode_files()
        self._episode_lengths = self._get_episode_lengths()
        self._num_samples = sum(self._episode_lengths)
        self._index_map = self._build_index_map()

    def _get_episode_files(self):
        files = glob.glob(os.path.join(self.data_dir, "*.h5"))
        files.sort()
        return files

    def _get_episode_lengths(self):
        lengths = []
        for file in self._episodes:
            with h5py.File(file, 'r') as f:
                length = len(f['action/cartesian_position']) - (self.horizon - 1)
                lengths.append(max(0, length))
        return lengths

    def _build_index_map(self):
        idx_map = []
        for ep_idx, length in enumerate(self._episode_lengths):
            idx_map.extend([(ep_idx, i) for i in range(length)])
        return idx_map

    def _load_frame(self, ep_idx: int, frame_idx: int):
        path = self._episodes[ep_idx]
        with h5py.File(path, 'r') as f:
            # 图像保持单帧
            base = f['observation/color_image'][frame_idx, ...]
            wrist = f['observation/zed_image'][frame_idx, ...]
            base_rgb = (np.array(base).astype(np.uint8))
            wrist_rgb = (np.array(wrist).astype(np.uint8))

            # 状态依旧单帧
            joints = f['observation/robot_state/joint_positions'][frame_idx]
            grip = f['observation/robot_state/gripper_position'][frame_idx]
            state = np.concatenate([joints, [grip]]).astype(np.float32)

            # 动作序列: horizon 帧
            actions_list = []
            for i in range(self.horizon):
                cart = f['action/cartesian_position'][frame_idx + i]
                grip_a = f['action/gripper_position'][frame_idx + i]
                actions_list.append(np.concatenate([cart, [grip_a]]))
            actions = np.stack(actions_list, axis=0).astype(np.float32)

        return base_rgb, wrist_rgb, state, actions

    def __getitem__(self, index) -> dict:
        idx = index.__index__()
        ep_idx, frame_idx = self._index_map[idx]
        base_rgb, wrist_rgb, state, actions = self._load_frame(ep_idx, frame_idx)

        images = {
            'base_0_rgb': np.array(base_rgb),
            'wrist_0_rgb': np.array(wrist_rgb)
        }
        # mask 标记所有 image key 都是有效的
        # 不能是jnp 得是np 不然resize_with_pad会报错
        #return字典里面对应data 必须全部平级
        # 图像格式uint8， 不能为float32
        image_masks = {k: np.array(True) for k in images}
        return {
            'image': images,
            'image_mask': image_masks,
            'state': np.array(state),
            'actions': np.array(actions),
        }
    def __len__(self) -> int:
        return self._num_samples


def make_franka_dataloader(
    data_dir: str = "./franka_datasets/dataset",
    split: str = "train",
    batch_size: int = 16,
    num_workers: int = 0,  # 默认改为0避免多进程问题
    shuffle_episodes: bool = True,
    pin_memory: bool = False,
    prefetch_factor: int = 2,
    image_key: str = "color_image",
    action_type: str = "joint_position",
    use_jax: bool = True,
):
    """
    创建Franka数据集的DataLoader
    """
    dataset = FrankaH5IterableDataset(
        data_dir=data_dir,
        split=split,
        image_key=image_key,
        action_type=action_type,
        shuffle_episodes=shuffle_episodes,
        use_jax=use_jax,
        worker_init_seed=42,
    )
    
    # 根据数组类型选择collate函数
    if use_jax:
        def jax_collate_fn(batch):
            # 在主进程中进行JAX转换，避免worker进程中的JAX冲突
            return {
                "image": jnp.stack([jnp.array(b["image"]) for b in batch]),
                "state": jnp.stack([jnp.array(b["state"]) for b in batch]),
                "actions": jnp.stack([jnp.array(b["actions"]) for b in batch]),
                "task": [b["task"] for b in batch],
            }
        collate_fn = jax_collate_fn
    else:
        def torch_collate_fn(batch):
            return {
                "image": torch.stack([torch.from_numpy(b["image"]) if isinstance(b["image"], np.ndarray) else b["image"] for b in batch]),
                "state": torch.stack([torch.from_numpy(b["state"]) if isinstance(b["state"], np.ndarray) else b["state"] for b in batch]),
                "actions": torch.stack([torch.from_numpy(b["actions"]) if isinstance(b["actions"], np.ndarray) else b["actions"] for b in batch]),
                "task": [b["task"] for b in batch],
            }
        collate_fn = torch_collate_fn
    
    # 根据是否使用JAX决定DataLoader参数
    if use_jax:
        # JAX模式下禁用多进程以避免冲突
        actual_num_workers = 0
        logger.info("JAX模式下禁用多进程DataLoader")
    else:
        actual_num_workers = num_workers
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=actual_num_workers,
        pin_memory=pin_memory and not use_jax,
        prefetch_factor=prefetch_factor if actual_num_workers > 0 else None,
        persistent_workers=actual_num_workers > 0,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn if actual_num_workers > 0 else None,
        multiprocessing_context='spawn' if actual_num_workers > 0 else None,  # 使用spawn避免fork问题
    )


# 测试和调试函数
def inspect_franka_dataset(data_dir: str = "./franka_datasets/dataset"):
    """检查Franka数据集的基本信息"""
    dataset = FrankaH5IterableDataset(
        data_dir=data_dir,
        split="train",
        use_jax=True,
    )
    
    print(f"找到 {len(dataset.episode_files)} 个训练episode文件")
    
    # 获取样本
    sample = dataset.get_sample()
    
    print("\n样本信息:")
    print(f"图像形状: {sample['image'].shape}")
    print(f"状态形状: {sample['state'].shape}")
    print(f"动作形状: {sample['actions'].shape}")
    print(f"任务描述: {sample['task']}")
    
    # 测试迭代器
    print("\n测试迭代器 (前3个样本):")
    count = 0
    for item in dataset:
        print(f"样本 {count}: 图像{item['image'].shape}, 状态{item['state'].shape}, 动作{item['actions'].shape}")
        count += 1
        if count >= 3:
            break


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 检查数据集
    inspect_franka_dataset(data_dir="../franka_datasets/dataset")
    
    # 创建DataLoader测试
    print("\n创建DataLoader测试:")
    dataloader = make_franka_dataloader(
        data_dir="../franka_datasets/dataset",
        batch_size=2,
        num_workers=0,  # 使用单进程避免冲突
        use_jax=True,
    )
    
    # 测试一个批次
    for batch in dataloader:
        print(f"批次形状:")
        print(f"  图像: {batch['image'].shape}")
        print(f"  状态: {batch['state'].shape}")
        print(f"  动作: {batch['actions'].shape}")
        print(f"  任务: {batch['task']}")
        break