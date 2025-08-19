import sys
sys.path.append('/home/jiaqi/openpi/src')

import numpy as np
from PIL import Image
import torch

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        print("无法导入 LeRobotDataset，请检查 lerobot 安装")
        LeRobotDataset = None

from openpi.policies.libero_policy import _parse_image

def check_lerobot_dataset():
    """直接检查 LeRobot 数据集中的图像数据"""
    
    print("=== 检查 LeRobot 数据集 ===")
    
    try:
        # 加载数据集
        dataset = LeRobotDataset("hw3579/libero_banana_state_7_crop")
        print(f"数据集加载成功，总样本数: {len(dataset)}")
        
        # 获取第一个样本
        sample = dataset[0]
        print(f"样本键: {list(sample.keys())}")
        
        # 检查图像数据
        if "observation.image" in sample:
            image_data = sample["observation.image"]
            print(f"\nobservation.image:")
            print(f"  类型: {type(image_data)}")
            print(f"  形状: {image_data.shape}")
            print(f"  数据类型: {image_data.dtype}")
            print(f"  值范围: [{image_data.min():.3f}, {image_data.max():.3f}]")
            
            # 转换为 numpy 并保存原始数据
            if isinstance(image_data, torch.Tensor):
                image_np = image_data.numpy()
            else:
                image_np = np.array(image_data)
                
            print(f"  numpy后形状: {image_np.shape}")
            
            # 保存原始数据（不经过 _parse_image）
            if image_np.ndim == 3 and image_np.shape[0] == 3:
                # 假设是 CHW 格式，直接转换保存
                raw_hwc = np.transpose(image_np, (1, 2, 0))
                if raw_hwc.dtype == np.float32 or raw_hwc.dtype == np.float64:
                    raw_hwc = (raw_hwc * 255).astype(np.uint8)
                Image.fromarray(raw_hwc).save("raw_dataset_image.png")
                print("  已保存原始图像为 raw_dataset_image.png")
            
            # 经过 _parse_image 处理
            parsed_image = _parse_image(image_np)
            print(f"  经过_parse_image后: shape={parsed_image.shape}, dtype={parsed_image.dtype}")
            Image.fromarray(parsed_image).save("parsed_dataset_image.png")
            print("  已保存处理后图像为 parsed_dataset_image.png")
            
            # 检查是否有条纹模式
            print("\n  条纹检查:")
            for i in range(min(5, parsed_image.shape[0])):
                row_vals = parsed_image[i, :10, 0]  # 前10个像素的第一个通道
                print(f"    第{i}行前10个像素: {row_vals}")
        
        if "observation.wrist_image" in sample:
            wrist_data = sample["observation.wrist_image"]
            print(f"\nobservation.wrist_image:")
            print(f"  形状: {wrist_data.shape}")
            print(f"  数据类型: {wrist_data.dtype}")
            print(f"  值范围: [{wrist_data.min():.3f}, {wrist_data.max():.3f}]")
            
    except Exception as e:
        print(f"加载数据集时出错: {e}")
        print(f"错误类型: {type(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_lerobot_dataset()
