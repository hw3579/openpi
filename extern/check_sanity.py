import jax.numpy as jnp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def check_sanity(batch):
    """
    检查 batch 内容的完整性函数
    
    Args:
        batch: tuple[Observation, Actions] 从数据加载器获取的批次数据
    """
    obs0, act0 = batch
    
    print("=" * 80)
    print("BATCH SANITY CHECK")
    print("=" * 80)
    
    # 检查观测数据
    print("\n📊 OBSERVATION DATA:")
    print(f"Type: {type(obs0)}")
    
    # 打印对象的属性
    for key in dir(obs0):
        if not key.startswith('_'):
            try:
                value = getattr(obs0, key)
                if hasattr(value, 'shape'):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                    if 'image' in key.lower():
                        print(f"    - min={jnp.min(value):.4f}, max={jnp.max(value):.4f}")
                else:
                    print(f"  {key}: {type(value)}")
            except:
                pass
    
    # 打印状态信息
    if hasattr(obs0, 'state'):
        print(f"\n🤖 STATE DATA:")
        state = obs0.state
        print(f"  Shape: {state.shape}")
        print(f"  Dtype: {state.dtype}")
        print(f"  Min: {jnp.min(state):.6f}")
        print(f"  Max: {jnp.max(state):.6f}")
        print(f"  Mean: {jnp.mean(state):.6f}")
        print(f"  Std: {jnp.std(state):.6f}")
        print(f"  First sample state: {state[0]}")
    
    # 检查动作数据
    print(f"\n🎯 ACTION DATA:")
    print(f"  Shape: {act0.shape}")
    print(f"  Dtype: {act0.dtype}")
    print(f"  Min: {jnp.min(act0):.6f}")
    print(f"  Max: {jnp.max(act0):.6f}")
    print(f"  Mean: {jnp.mean(act0):.6f}")
    print(f"  Std: {jnp.std(act0):.6f}")
    print(f"  Has NaN: {jnp.isnan(act0).any()}")
    print(f"  Has Inf: {jnp.isinf(act0).any()}")
    print(f"  First sample actions: {act0[0]}")
    
    # 保存图像
    print(f"\n🖼️  SAVING IMAGES:")
    
    # 检查是否有 images 属性，并且是字典
    if hasattr(obs0, 'images') and isinstance(obs0.images, dict):
        print(f"Found images dict with keys: {list(obs0.images.keys())}")
        image_keys = list(obs0.images.keys())[:2]  # 只取前两个
        
        saved_count = 0
        for i, key in enumerate(image_keys):
            try:
                img_data = obs0.images[key]
                
                # 取第一个批次的图像
                img = img_data[0]
                
                # ── 处理通道顺序 ─────────────────────────────────────
                if img.ndim == 3 and img.shape[0] == 3:
                    # (3,H,W) -> (H,W,3)
                    img = jnp.transpose(img, (1, 2, 0))
                elif img.ndim == 4 and img.shape[-1] == 3:
                    # 形如 (P, H, W, 3) —— 去掉 patch / time 维
                    P, H, W, C = img.shape
                    img = jnp.reshape(img, (P * H, W, C))        # 拼回完整高度
                
                # 转换为numpy并调整数据范围
                img_np = np.array(img)
                
                # 如果是浮点数且在[-1,1]或[0,1]范围内，转换为[0,255]
                if img_np.dtype in [np.float32, np.float64]:
                    if img_np.min() >= -1.0 and img_np.max() <= 1.0:
                        # 从[-1,1]或[0,1]转换到[0,255]
                        img_np = ((img_np + 1.0) / 2.0 * 255).astype(np.uint8)
                    else:
                        # 如果不在标准范围，进行归一化
                        img_np = ((img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255).astype(np.uint8)
                
                # 确保是uint8类型
                img_np = img_np.astype(np.uint8)
                
                # 保存图像
                filename = f"{i+1}.png"
                if len(img_np.shape) == 3:
                    # RGB图像
                    img_pil = Image.fromarray(img_np)
                elif len(img_np.shape) == 2:
                    # 灰度图像
                    img_pil = Image.fromarray(img_np, mode='L')
                else:
                    print(f"    ❌ Unsupported image shape for {key}: {img_np.shape}")
                    continue
                    
                img_pil.save(filename)
                print(f"    ✅ Saved {key} as {filename} (shape: {img_np.shape})")
                
                # =========================
                # 深度分析条纹问题
                # =========================
                if key == "base_0_rgb" and i == 0:
                    print(f"\n🔍 深度分析条纹问题 ({key}):")
                    print(f"    原始数据形状: {img_data[0].shape}")
                    print(f"    原始数据类型: {img_data[0].dtype}")
                    print(f"    原始值范围: [{jnp.min(img_data[0]):.4f}, {jnp.max(img_data[0]):.4f}]")
                    
                    # 检查条纹模式
                    first_img = img_np
                    print(f"    处理后形状: {first_img.shape}")
                    
                    # 检查前几行的像素模式
                    print(f"    前5行的R通道前10个像素:")
                    for r in range(min(5, first_img.shape[0])):
                        row_pixels = first_img[r, :10, 0]
                        print(f"      第{r}行: {row_pixels}")
                    
                    # 检查是否有序列模式（条纹的典型特征）
                    flat_r = first_img[:, :, 0].flatten()
                    print(f"    扁平化后前20个R通道值: {flat_r[:20]}")
                    
                    # 检查相邻像素差值（序列数据会有固定差值）
                    diffs = np.diff(flat_r[:50])
                    unique_diffs = np.unique(diffs)
                    print(f"    前50个像素的相邻差值唯一值: {unique_diffs[:10]}...")
                    
                    # 检查周期性（如果是CHW->HWC错误变换会有周期性）
                    if len(flat_r) >= 672:  # 3*224
                        stride_pattern = flat_r[::224][:10]  # 每隔224个像素
                        print(f"    每隔224个像素的值: {stride_pattern}")
                    
                    # 分析是否符合错误reshape的模式
                    total_pixels = first_img.shape[0] * first_img.shape[1] * first_img.shape[2]
                    if total_pixels == 224 * 224 * 3:
                        # 模拟错误的reshape过程
                        test_seq = np.arange(total_pixels, dtype=np.uint8)
                        wrong_chw = test_seq.reshape(3, 224, 224)
                        wrong_hwc = np.transpose(wrong_chw, (1, 2, 0))
                        
                        # 比较前100个像素
                        match_score = np.sum(flat_r[:100] == wrong_hwc.flatten()[:100])
                        print(f"    与错误reshape模式匹配度: {match_score}/100")
                        
                        if match_score > 80:  # 如果80%以上匹配
                            print(f"    ⚠️ 检测到可能的reshape错误！")
                            print(f"    问题：数据可能被错误地从扁平序列reshape为(C,H,W)再转为(H,W,C)")
                
                saved_count += 1
                
            except Exception as e:
                print(f"    ❌ Failed to save {key}: {e}")
    else:
        print("    ❌ No images attribute found or it's not a dict")
        saved_count = 0
    
    if saved_count == 0:
        print("    ⚠️  No images found or saved")
    
    print("\n" + "=" * 80)
    print("SANITY CHECK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    # 示例用法
    print("This is a utility module for checking batch sanity.")
    print("Import and use: from extern.check_sanity import check_sanity")
