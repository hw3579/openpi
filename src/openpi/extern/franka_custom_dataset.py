import os
import glob
import re
import h5py
import numpy as np
from torch.utils.data import Dataset as TorchDataset

class DebugDict(dict):
    """
    在访问 'timestamp' 时打印当前的 ep_idx & frame_idx。
    """
    def __init__(self, *args, ep_idx=None, frame_idx=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._ep_idx = ep_idx
        self._frame_idx = frame_idx

    def __getitem__(self, key):
        if key == 'timestamp':
            print(f"[FrankaH5Dataset] timestamp accessed at episode={self._ep_idx}, frame={self._frame_idx}")
        return super().__getitem__(key)


class FrankaH5Dataset(TorchDataset):
    """
    用于 OpenPI 训练的 Franka H5 数据集，输出与 FakeDataset 严格对齐：
      {
        'base_0_rgb':    np.uint8 (H, W, 3),
        'wrist_0_rgb':   np.uint8 (H, W, 3),
        'base_0_rgb_mask': bool,
        'wrist_0_rgb_mask': bool,
        'state':         np.float32 (N_state,),
        'actions':       np.float32 (horizon, N_action),
        'timestamp':     np.float32 scalar,
        'frame_index':   np.int64 scalar,
        'episode_index': np.int64 scalar,
        'index':         np.int64 scalar,
        'task_index':    np.int64 scalar,
        'actions_is_pad': np.bool_ (horizon,),
        'task':          str
      }
    """
    def __init__(self, model_config: "_model.BaseModelConfig", data_config: "_config.DataConfig"):
        super().__init__()
        assert hasattr(data_config, 'data_dir'), "data_config 必须包含 data_dir 属性"
        self.data_dir = data_config.data_dir
        self.horizon = model_config.action_horizon

        # 收集所有 .h5 文件并排序
        self._episodes = glob.glob(os.path.join(self.data_dir, "*.h5"))
        self._episodes.sort()

        # 从文件名提取 task 名和实例 idx
        pattern = re.compile(r'(.+?)_(\d+)\.h5$')
        self._episode_tasks = []
        for p in self._episodes:
            m = pattern.search(os.path.basename(p))
            if not m:
                raise ValueError(f"无法解析文件名: {p}")
            self._episode_tasks.append(m.group(1))

        # 唯一 task → task_index
        unique = sorted(set(self._episode_tasks))
        self._task_to_idx = {t:i for i,t in enumerate(unique)}
        self._episode_task_idx = [ self._task_to_idx[t] for t in self._episode_tasks ]

        # 每个 episode 的有效帧数
        self._episode_lengths = []
        for path in self._episodes:
            with h5py.File(path, 'r') as f:
                total = len(f['action/cartesian_position'])
                valid = total - (self.horizon - 1)
                self._episode_lengths.append(max(0, valid))

        # 全局样本数 & 索引映射
        self._num_samples = sum(self._episode_lengths)
        self._index_map = [
            (ep_idx, f_idx)
            for ep_idx, L in enumerate(self._episode_lengths)
            for f_idx in range(L)
        ]

    def __len__(self):
        return self._num_samples

    def _load_frame(self, ep_idx: int, frame_idx: int):
        """
        读取第 ep_idx 个文件、第 frame_idx 帧：
        - 图像 uint8
        - 状态 float32
        - 动作序列 float32
        - 时间戳 float32
        """
        path = self._episodes[ep_idx]
        with h5py.File(path, 'r') as f:
            # 原始 uint8 图像
            base_img  = f['observation/color_image'][frame_idx][...] .astype(np.uint8)
            wrist_img = f['observation/zed_image'][frame_idx][...]  .astype(np.uint8)

            # 机器人状态
            cart_pos = f['observation/robot_state/cartesian_position'][frame_idx]    # (7,)
            grip   = f['observation/robot_state/gripper_position'][frame_idx]   # ()
            state  = np.hstack([cart_pos, [-grip], [grip]]).astype(np.float32)        # (8,)

            # horizon 步动作
            acts = []
            for i in range(self.horizon):
                cart   = f['action/cartesian_position'][frame_idx + i]         # (3,)
                grip_a = f['action/gripper_position'][frame_idx + i]          # ()
                acts.append(np.concatenate([cart, [grip_a]]))
            actions = np.stack(acts, axis=0).astype(np.float32)               # (horizon,4)

            # 时间戳
            # timestamp = np.array(f['timestamp'][frame_idx]).astype(np.float32)

            # 数据集没有时间戳，
            timestamp = np.array(0.0, dtype=np.float32)  # 使用 0.0 作为占位符

        return base_img, wrist_img, state, actions, timestamp

    def __getitem__(self, index) -> DebugDict:
        idx = int(index)
        ep_idx, frame_idx = self._index_map[idx]
        base, wrist, state, actions, timestamp = self._load_frame(ep_idx, frame_idx)

        # 平铺所有字段
        # out = {
        #     # 图像
        #     'base_0_rgb':        base,
        #     'wrist_0_rgb':       wrist,
        #     'base_0_rgb_mask':   np.bool_(True),
        #     'wrist_0_rgb_mask':  np.bool_(True),

        #     # 状态 & 动作
        #     'state':             state,
        #     'actions':           actions,
        #     'actions_is_pad':    np.zeros(self.horizon, dtype=bool),

        #     # 额外信息
        #     'timestamp':         timestamp,
        #     'frame_index':       np.int64(frame_idx),
        #     'episode_index':     np.int64(ep_idx),
        #     'index':             np.int64(idx),
        #     'task_index':        np.int64(self._episode_task_idx[ep_idx]),
        #     'task':              self._episode_tasks[ep_idx],
        # }

        out = {
            'image':          base,                               # uint8 (H,W,3)
            'wrist_image':    wrist,                              # uint8 (H,W,3)
            'state':          state,                              # float32 (N_state,)
            'actions':        actions,                            # float32 (horizon,N_action)
            'timestamp':      timestamp,                          # float32 scalar
            'frame_index':    np.int64(frame_idx),                # int64 scalar
            'episode_index':  np.int64(ep_idx),                   # int64 scalar
            'index':          np.int64(idx),                      # int64 scalar
            'task_index':     np.int64(self._episode_task_idx[ep_idx]),  # int64 scalar
            'actions_is_pad': np.zeros(self.horizon, dtype=bool), # bool (horizon,)
            'task':           self._episode_tasks[ep_idx],        # str
            'prompt':           self._episode_tasks[ep_idx],        # 不知道为什么要实现 关键在于libero数据集什么时候增加了这个键？ 但是无论怎样这个prompt和task是一样的

            # 然后 此外在config的自定义的h5dataset中的create 复写了基类的create方法中 需要手动复制libero的数据集变化实现 包括repack_transform和data_transforms
        }

        # 用 DebugDict 包一层，访问 timestamp 时会打印
        return DebugDict(out, ep_idx=ep_idx, frame_idx=frame_idx)
