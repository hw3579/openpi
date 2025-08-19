import pandas as pd
import numpy as np
from PIL import Image
import sys

def check_parquet_data():
    """检查原始 parquet 文件中的图像数据"""
    
    print("=== 检查原始 parquet 数据 ===")
    
    try:
        # 尝试找到你的 parquet 文件
        parquet_files = []
        import os
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.endswith('.parquet'):
                    parquet_files.append(os.path.join(root, file))
        
        if not parquet_files:
            print("没有找到 .parquet 文件")
            return
            
        print(f"找到 parquet 文件: {parquet_files}")
        
        # 读取第一个 parquet 文件
        parquet_file = parquet_files[0]
        df = pd.read_parquet(parquet_file)
        
        print(f"\n数据集信息:")
        print(f"  行数: {len(df)}")
        print(f"  列名: {list(df.columns)}")
        
        # 检查图像列
        for col in ['image', 'wrist_image']:
            if col in df.columns:
                print(f"\n{col} 列分析:")
                first_image = df[col].iloc[0]
                print(f"  类型: {type(first_image)}")
                
                if hasattr(first_image, '__len__'):
                    print(f"  长度: {len(first_image)}")
                    if len(first_image) == 256 * 256 * 3:
                        print(f"  ✅ 长度匹配 256*256*3 = {256*256*3}")
                        
                        # 转换为 numpy 数组检查
                        img_array = np.array(first_image)
                        print(f"  数组形状: {img_array.shape}")
                        print(f"  数据类型: {img_array.dtype}")
                        print(f"  值范围: [{img_array.min()}, {img_array.max()}]")
                        
                        # 检查前20个值，看是否有序列模式
                        print(f"  前20个值: {img_array[:20]}")
                        
                        # 尝试不同的 reshape 方式
                        print(f"\n  尝试不同 reshape 方式:")
                        
                        # 方式1: 直接 reshape 为 (256, 256, 3) - 正确方式
                        try:
                            correct_hwc = img_array.reshape(256, 256, 3)
                            Image.fromarray(correct_hwc.astype(np.uint8)).save(f"parquet_{col}_correct_hwc.png")
                            print(f"    ✅ 方式1 (256,256,3): 已保存为 parquet_{col}_correct_hwc.png")
                        except Exception as e:
                            print(f"    ❌ 方式1失败: {e}")
                        
                        # 方式2: 错误的 reshape 为 (3, 256, 256) 再转置 - 可能导致条纹
                        try:
                            wrong_chw = img_array.reshape(3, 256, 256)
                            wrong_hwc = np.transpose(wrong_chw, (1, 2, 0))
                            Image.fromarray(wrong_hwc.astype(np.uint8)).save(f"parquet_{col}_wrong_chw_transpose.png")
                            print(f"    ⚠️  方式2 (3,256,256)->transpose: 已保存为 parquet_{col}_wrong_chw_transpose.png")
                        except Exception as e:
                            print(f"    ❌ 方式2失败: {e}")
                        
                        # 检查哪种方式产生条纹
                        if len(img_array) >= 20:
                            # 检查是否有明显的序列模式（条纹特征）
                            diffs = np.diff(img_array[:50])
                            unique_diffs = np.unique(diffs)
                            print(f"    前50个值的差值模式: {len(unique_diffs)} 种不同差值")
                            if len(unique_diffs) <= 3:  # 如果差值种类很少，可能是序列数据
                                print(f"    ⚠️ 可能是序列数据: {unique_diffs}")
                    else:
                        print(f"  ❌ 长度不匹配: {len(first_image)} != {256*256*3}")
                else:
                    print(f"  ❌ 无法获取长度")
        
    except Exception as e:
        print(f"检查失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_parquet_data()
