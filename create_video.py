#!/usr/bin/env python3
"""
Generate a video from saved logs: displays images, text, state and actions data on white background.
Logs directory structure:
  logs_dir/
    YYYYMMDD_HHMMSS/
      image.png
      wrist_image.png
      prompt.txt
      state.json
      actions.json (or actions_predicted.json + actions_ground_truth.json)
      result.json
      action_comparison.png (if ground truth exists)

Usage:
  python create_video.py --logs_dir ../logs --output video.mp4 --fps 2
"""
import argparse
import cv2
import os
import json
import numpy as np
from pathlib import Path

def load_json_array(file_path):
    """Load numpy array from JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        # Handle different JSON structures
        if isinstance(data, dict):
            # Find the array data in the dict
            for key, value in data.items():
                if isinstance(value, list):
                    return np.array(value)
        elif isinstance(data, list):
            return np.array(data)
        return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def create_joint_comparison_chart(predicted_actions, gt_actions, width=500, height=400):
    """Create a horizontal bar chart comparing predicted and ground truth actions for each joint with percentage differences."""
    if predicted_actions is None or gt_actions is None:
        # Create empty chart with error message
        chart = np.ones((height, width, 3), dtype=np.uint8) * 255
        cv2.putText(chart, "No action data", (width//2-60, height//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        return chart
    
    # Create white background
    chart = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Chart parameters - 增大边距以容纳百分比信息
    margin = 50
    chart_width = width - 2 * margin - 80  # 预留右侧空间显示百分比
    chart_height = height - 2 * margin - 40  # 预留底部空间
    
    # Joint parameters
    num_joints = min(len(predicted_actions), len(gt_actions), 7)  # Max 7 joints
    joint_height = chart_height // num_joints
    bar_height = max(joint_height // 2, 20)  # 增大条形高度，最小20像素
    
    # Find max absolute value for scaling
    all_values = np.concatenate([predicted_actions[:num_joints], gt_actions[:num_joints]])
    max_abs_val = max(abs(np.min(all_values)), abs(np.max(all_values)), 0.001)  # Avoid division by zero
    
    # Draw chart
    for i in range(num_joints):
        y_center = margin + i * joint_height + joint_height // 2
        
        # Draw joint label
        cv2.putText(chart, f"J{i}", (5, y_center + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Draw zero line
        zero_x = margin + chart_width // 2
        cv2.line(chart, (zero_x, y_center - bar_height//2), 
                (zero_x, y_center + bar_height//2), (128, 128, 128), 2)  # 加粗零线
        
        # Get values for this joint
        pred_val = predicted_actions[i] if i < len(predicted_actions) else 0
        gt_val = gt_actions[i] if i < len(gt_actions) else 0
        
        # Calculate percentage difference
        if abs(gt_val) > 1e-6:  # 避免除零
            percentage_diff = abs(pred_val - gt_val) / abs(gt_val) * 100
        else:
            # 如果ground truth接近零，使用绝对差异
            percentage_diff = abs(pred_val - gt_val) * 100
        
        # Draw predicted action bar (blue) - 增大条形
        if i < len(predicted_actions):
            bar_width = int((abs(pred_val) / max_abs_val) * (chart_width // 2))
            if pred_val >= 0:
                bar_start_x = zero_x
                bar_end_x = zero_x + bar_width
            else:
                bar_start_x = zero_x - bar_width
                bar_end_x = zero_x
            
            cv2.rectangle(chart, 
                         (bar_start_x, y_center - bar_height//2 - 3), 
                         (bar_end_x, y_center - 3), 
                         (255, 0, 0), -1)  # Blue for predicted
            
            # 添加预测值标签
            cv2.putText(chart, f"{pred_val:.3f}", (bar_end_x + 5 if pred_val >= 0 else bar_start_x - 60, y_center - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)
        
        # Draw ground truth bar (green) - 增大条形
        if i < len(gt_actions):
            bar_width = int((abs(gt_val) / max_abs_val) * (chart_width // 2))
            if gt_val >= 0:
                bar_start_x = zero_x
                bar_end_x = zero_x + bar_width
            else:
                bar_start_x = zero_x - bar_width
                bar_end_x = zero_x
            
            cv2.rectangle(chart, 
                         (bar_start_x, y_center + 3), 
                         (bar_end_x, y_center + bar_height//2 + 3), 
                         (0, 255, 0), -1)  # Green for ground truth
            
            # 添加真实值标签
            cv2.putText(chart, f"{gt_val:.3f}", (bar_end_x + 5 if gt_val >= 0 else bar_start_x - 60, y_center + 18), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 128, 0), 1)
        
        # 在右侧显示百分比差异
        diff_color = (0, 0, 255) if percentage_diff > 50 else (255, 165, 0) if percentage_diff > 20 else (0, 150, 0)
        cv2.putText(chart, f"{percentage_diff:.1f}%", (width - 70, y_center + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, diff_color, 1)
    
    # Add title and legend
    cv2.putText(chart, "Joint Actions Comparison with % Diff", (width//2-120, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Add legend - 调整位置
    legend_y = height - 35
    cv2.rectangle(chart, (margin, legend_y), (margin + 15, legend_y + 10), (255, 0, 0), -1)
    cv2.putText(chart, "Predicted", (margin + 20, legend_y + 8), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    cv2.rectangle(chart, (margin + 100, legend_y), (margin + 115, legend_y + 10), (0, 255, 0), -1)
    cv2.putText(chart, "Ground Truth", (margin + 120, legend_y + 8), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # 添加百分比说明
    cv2.putText(chart, "% Diff", (width - 70, legend_y + 8), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    return chart

def format_array_text(arr, label):
    """Format numpy array into readable text lines with individual values."""
    if arr is None:
        return [f"{label}: None"]
    
    arr = np.asarray(arr)
    if arr.ndim == 1:
        # Format 1D array values showing each dimension
        lines = [f"{label}:"]
        for i, val in enumerate(arr):
            lines.append(f"  [{i}]: {val:.6f}")
        return lines
    else:
        return [f"{label}: shape {arr.shape}"]

def format_array_text_grouped(arr, label):
    """Format numpy array into grouped text lines (0-6, 7-13, etc.)."""
    if arr is None:
        return [f"{label}: None"]
    
    arr = np.asarray(arr)
    if arr.ndim == 1:
        lines = [f"{label}:"]  # 只保留总标题
        print(f"Processing {label} with {len(arr)} elements")  # 调试信息
        # Group by 7 elements (0-6, 7-13, etc.)
        for group_start in range(0, len(arr), 7):
            # 直接显示元素，不显示组标题
            for i in range(group_start, min(group_start + 7, len(arr))):
                lines.append(f"  [{i}]: {arr[i]:.6f}")
            
            # Add separator line between groups (except for the last group)
            if group_start + 7 < len(arr):
                lines.append("  " + "-" * 30)
        
        print(f"Generated {len(lines)} lines for {label}")  # 调试信息
        return lines
    else:
        return [f"{label}: shape {arr.shape}"]

def create_text_image(text_lines, width, height):
    """Create an image with text on white background."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5  # 增大字体
    color = (0, 0, 0)  # Black text
    thickness = 1
    line_height = 22  # 增加行高
    
    y = 30  # 增加起始位置
    for line in text_lines:
        # 自动换行处理
        words = line.split(' ')
        current_line = ""
        
        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            # 测试文本宽度
            (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
            
            if text_width < width - 30:  # 留出边距
                current_line = test_line
            else:
                # 当前行已满，输出并开始新行
                if current_line and y + line_height < height:
                    cv2.putText(img, current_line, (15, y), font, font_scale, color, thickness, cv2.LINE_AA)
                    y += line_height
                current_line = word
        
        # 输出最后一行
        if current_line and y + line_height < height:
            cv2.putText(img, current_line, (15, y), font, font_scale, color, thickness, cv2.LINE_AA)
            y += line_height
    
    return img

def detect_video_mode(logs_dir: Path, runs):
    """
    智能检测视频模式：
    - RT-Only模式：logs目录下没有replay子文件夹
    - Replay-Only模式：logs目录下有replay子文件夹但没有rt子文件夹
    - Both模式：logs目录下既有replay子文件夹又有rt子文件夹
    - Sim模式：logs目录名为sim或检测到仿真数据，套用Both模式显示
    """
    has_ground_truth = False
    has_rt_images = False
    has_regular_images = False
    
    # 检查logs目录下的子文件夹结构来判断模式
    has_replay_folder = (logs_dir / "replay").exists()
    has_rt_folder = (logs_dir / "rt").exists()
    has_sim_folder = (logs_dir / "sim").exists()
    
    # 通过目录名快速判断（优先检查）
    dir_name = logs_dir.name.lower()
    if (dir_name == "sim" or has_sim_folder) and has_replay_folder:
        # Sim模式：既有replay又有sim文件夹，套用Both模式的显示逻辑
        return {
            "mode_name": "Sim Mode (Simulation vs Dataset, using Both display)",
            "is_replay_mode": True,  # 套用Both模式，所以设为True
            "has_rt_images": True   # 套用Both模式，显示sim数据作为"RT"对比
        }
    elif (dir_name == "sim" or has_sim_folder) and not has_replay_folder:
        # 纯Sim模式：只有sim文件夹
        return {
            "mode_name": "Pure Sim Mode (Simulation Only)",
            "is_replay_mode": False,
            "has_rt_images": False
        }
    elif not has_replay_folder and has_rt_folder:
        # 只有rt文件夹，没有replay文件夹 -> RT-Only模式
        return {
            "mode_name": "RT-Only Mode (Real-time Only)",
            "is_replay_mode": False,
            "has_rt_images": True
        }
    elif has_replay_folder and not has_rt_folder and not has_sim_folder:
        # 只有replay文件夹，没有rt文件夹 -> Replay-Only模式
        return {
            "mode_name": "Replay-Only Mode (Dataset Comparison)",
            "is_replay_mode": True,
            "has_rt_images": False
        }
    elif has_replay_folder and has_rt_folder:
        # 既有replay又有rt文件夹 -> Both模式
        return {
            "mode_name": "Both Mode (Dataset + Real-time Comparison)",
            "is_replay_mode": True,
            "has_rt_images": True
        }
    
    # 通过目录名快速判断（保留原有逻辑作为后备）
    dir_name = logs_dir.name.lower()
    if dir_name == "rt":
        return {
            "mode_name": "RT-Only Mode (Real-time Images Only)",
            "is_replay_mode": False,
            "has_rt_images": True
        }
    elif dir_name == "replay":
        return {
            "mode_name": "Replay-Only Mode (Dataset Comparison)",
            "is_replay_mode": True,
            "has_rt_images": False
        }
    elif dir_name == "both":
        return {
            "mode_name": "Both Mode (Dataset + Real-time Comparison)",
            "is_replay_mode": True,
            "has_rt_images": True
        }
    
    # 如果目录名不明确，则分析文件内容
    for run in runs[:5]:  # 只检查前5个运行以提高效率
        # 检查是否有ground truth对比文件
        if (run / "actions_predicted.json").exists() and (run / "actions_ground_truth.json").exists():
            has_ground_truth = True
        
        # 检查是否有实时图像
        if (run / "rt_image.png").exists() or (run / "rt_wrist_image.png").exists():
            has_rt_images = True
            
        # 检查是否有常规图像
        if (run / "image.png").exists():
            has_regular_images = True
    
    # 根据文件存在情况判断模式
    if has_ground_truth and has_rt_images:
        mode_name = "Both Mode (Dataset + Real-time Comparison)"
        is_replay_mode = True
    elif has_ground_truth and not has_rt_images:
        mode_name = "Replay-Only Mode (Dataset Comparison)"
        is_replay_mode = True
    elif not has_ground_truth and has_rt_images:
        mode_name = "RT-Only Mode (Real-time Images Only)"
        is_replay_mode = False
    elif not has_ground_truth and not has_rt_images and has_regular_images:
        mode_name = "Real-time Test Mode"
        is_replay_mode = False
    else:
        mode_name = "Unknown Mode"
        is_replay_mode = False
    
    return {
        "mode_name": mode_name,
        "is_replay_mode": is_replay_mode,
        "has_rt_images": has_rt_images
    }

def resize_image(img, target_height, max_width=None):
    """Resize image maintaining aspect ratio with optional max width limit."""
    h, w = img.shape[:2]
    aspect_ratio = w / h
    new_width = int(target_height * aspect_ratio)
    
    # If max_width is specified and new_width exceeds it, scale down
    if max_width and new_width > max_width:
        new_width = max_width
        target_height = int(max_width / aspect_ratio)
    
    return cv2.resize(img, (new_width, target_height))

def main(logs_dir: Path, output: Path, fps: int):
    # 首先检查是否需要重定向到子目录
    has_replay_folder = (logs_dir / "replay").exists()
    has_rt_folder = (logs_dir / "rt").exists()
    has_sim_folder = (logs_dir / "sim").exists()
    
    # 智能重定向：根据子文件夹结构
    if has_sim_folder and has_replay_folder and not has_rt_folder:
        print(f"🔄 重定向到Sim模式目录，套用Both模式展示: {logs_dir / 'replay'} + {logs_dir / 'sim'}")
        # Sim模式：套用Both模式的展示逻辑，将sim文件夹作为RT数据源
        actual_logs_dir = logs_dir / "replay"  # 主要使用replay数据
        rt_logs_dir = logs_dir / "sim"  # sim数据用于对比（相当于RT数据）
    elif has_sim_folder and not has_replay_folder and not has_rt_folder:
        print(f"🔄 重定向到Pure Sim模式目录: {logs_dir / 'sim'}")
        # 纯Sim模式：只有sim文件夹的情况
        actual_logs_dir = logs_dir / "sim"
        rt_logs_dir = None
    elif not has_replay_folder and has_rt_folder and not has_sim_folder:
        print(f"🔄 重定向到RT模式目录: {logs_dir / 'rt'}")
        actual_logs_dir = logs_dir / "rt"
        rt_logs_dir = None  # RT-Only模式不需要对比数据
    elif has_replay_folder and not has_rt_folder and not has_sim_folder:
        print(f"🔄 重定向到Replay模式目录: {logs_dir / 'replay'}")
        actual_logs_dir = logs_dir / "replay"
        rt_logs_dir = None  # Replay-Only模式不需要RT数据
    elif has_replay_folder and has_rt_folder:
        print(f"🔄 检测到Both模式，合并replay和rt数据")
        # Both模式：需要合并两个目录的数据
        actual_logs_dir = logs_dir / "replay"  # 主要使用replay数据
        rt_logs_dir = logs_dir / "rt"  # RT数据用于对比
    else:
        actual_logs_dir = logs_dir
        rt_logs_dir = None  # 默认情况下不需要RT对比数据
    
    # Collect and sort run directories
    runs = [d for d in actual_logs_dir.iterdir() if d.is_dir()]
    if not runs:
        print(f"No run subdirectories found in {actual_logs_dir}")
        return
    
    # In both mode, we need to merge RT comparison data
    if rt_logs_dir and rt_logs_dir.exists():
        rt_runs = {d.name: d for d in rt_logs_dir.iterdir() if d.is_dir()}
        print(f"📊 找到 {len(rt_runs)} 个RT对比数据目录")
    else:
        rt_runs = {}

    # Sort runs by timestamp, handling both old and new formats
    def get_timestamp_sort_key(run_dir):
        name = run_dir.name
        try:
            # Handle new format: YYYYMMDD_HHMMSS_mmm
            if name.count('_') == 2:
                date_part, time_part, ms_part = name.split('_')
                return f"{date_part}_{time_part}_{ms_part.zfill(3)}"
            # Handle old format: YYYYMMDD_HHMMSS
            elif name.count('_') == 1:
                return f"{name}_000"  # Add 000 milliseconds for old format
            else:
                return name
        except:
            return name
    
    runs.sort(key=get_timestamp_sort_key)
    
    # 智能检测模式：根据目录结构和文件内容（使用原始logs_dir进行检测）
    mode_info = detect_video_mode(logs_dir, runs)
    is_replay_mode = mode_info["is_replay_mode"]
    mode_name = mode_info["mode_name"]
    has_rt_images_global = mode_info["has_rt_images"]
    
    print(f"🎯 检测到模式: {mode_name}")
    if has_rt_images_global:
        print("📷 包含实时图像数据")
    
    # Define layout parameters for 1080p video
    video_width = 1920   # 1080p宽度
    video_height = 1080  # 1080p高度
    
    img_height = 300     # 保持图像合适大小
    spacing = 40         # 增大间距
    
    # 定义区域分布 - 优化文字区域
    # 左侧：图像区域 (主图像 + 手腕图像 + 比较图表)
    # 右侧：文字区域分为多列
    left_area_width = 950     # 减小左侧图像区域宽度，为文字区域让出更多空间
    right_area_width = 930    # 增加右侧文字区域宽度
    
    text_column_width = 300   # 减小每列文字宽度，可以放更多列
    
    if is_replay_mode:
        print(f"Creating video in REPLAY MODE ({mode_name}) - 1080p")
    else:
        print(f"Creating video in REAL-TIME MODE ({mode_name}) - 1080p")
    
    # Setup video writer - 使用更通用的H.264编码器
    # 尝试多种编码器，优先使用最兼容的
    fourcc_options = [
        cv2.VideoWriter_fourcc(*'H264'),  # H.264 (最通用)
        cv2.VideoWriter_fourcc(*'XVID'),  # XVID (广泛支持)
        cv2.VideoWriter_fourcc(*'MJPG'),  # Motion JPEG (兼容性好)
        cv2.VideoWriter_fourcc(*'mp4v')   # MP4V (原始选项作为后备)
    ]
    
    video = None
    for fourcc in fourcc_options:
        try:
            video = cv2.VideoWriter(str(output), fourcc, fps, (video_width, video_height))
            if video.isOpened():
                print(f"✅ 使用编码器: {fourcc}")
                break
            else:
                video.release()
                video = None
        except:
            if video:
                video.release()
                video = None
                
    if video is None:
        print("❌ 无法初始化视频编码器，使用默认选项")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(str(output), fourcc, fps, (video_width, video_height))

    for frame_idx, run in enumerate(runs, 1):  # 从1开始计数帧数
        img_path = run / "image.png"
        wrist_img_path = run / "wrist_image.png"
        prompt_path = run / "prompt.txt"
        state_path = run / "state.json"
        actions_path = run / "actions.json"
        
        # RT图像路径 - 初始设为None
        rt_img_path = None
        rt_wrist_img_path = None
        
        # In both mode, find corresponding RT data by index (按顺序配对)
        if rt_runs:
            # 将rt_runs转换为按名称排序的列表
            sorted_rt_runs = sorted(rt_runs.items())
            # 按当前run的索引找对应的RT数据
            current_run_index = runs.index(run)
            if current_run_index < len(sorted_rt_runs):
                rt_run_name, rt_run = sorted_rt_runs[current_run_index]
                # RT数据现在也使用统一的文件名：image.png和wrist_image.png
                potential_rt_img = rt_run / "image.png"
                potential_rt_wrist_img = rt_run / "wrist_image.png"
                
                if potential_rt_img.exists():
                    rt_img_path = potential_rt_img
                if potential_rt_wrist_img.exists():
                    rt_wrist_img_path = potential_rt_wrist_img
                
        actions_pred_path = run / "actions_predicted.json"
        actions_gt_path = run / "actions_ground_truth.json"
        comparison_plot_path = run / "action_comparison.png"
        
        # 检查必需文件 - 根据模式调整
        if mode_name.startswith("RT-Only"):
            # RT-Only模式：现在RT数据保存为image.png和wrist_image.png（键名统一后）
            if not img_path.exists() or not prompt_path.exists():
                print(f"Skipping {run}, missing required RT files (image.png or prompt.txt)")
                continue
            # RT-Only模式下不需要额外的RT对比图像
            rt_img_path = None
            rt_wrist_img_path = None
        else:
            # 其他模式：需要image.png和prompt.txt
            if not img_path.exists() or not prompt_path.exists():
                print(f"Skipping {run}, missing required files (image.png or prompt.txt)")
                continue
            
        # Create white background frame (1080p)
        frame = np.ones((video_height, video_width, 3), dtype=np.uint8) * 255
        
        # 左侧图像区域布局 - 重新设计为上下布局
        # 上半部分：主相机和手腕图片并列
        # 中间部分：实时图像并列（如果存在）
        # 下半部分：对比图表
        
        images_start_x = spacing
        images_start_y = spacing
        
        # 计算图像区域尺寸
        available_img_width = left_area_width - spacing * 2
        
        # 检查是否有实时图像
        has_rt_images = rt_img_path is not None or rt_wrist_img_path is not None
        
        # 根据是否有实时图像调整布局
        if has_rt_images:
            # 三层布局：RT图像在上、Replay图像在中、对比图表在下
            img_area_height = (video_height - spacing * 4) // 3  # 分为三部分
        else:
            # 两层布局：Replay图像、对比图表
            img_area_height = (video_height - spacing * 3) // 2  # 分为上下两部分
        
        # 第一层：RT图像（如果存在）- 放在最上面
        if has_rt_images:
            rt_images_y = images_start_y
            current_x_offset = images_start_x
            
            # Load and resize realtime main image
            if rt_img_path is not None:
                rt_main_img = cv2.imread(str(rt_img_path))
                if rt_main_img is not None:
                    max_img_width = available_img_width // 2 - spacing
                    rt_main_img = resize_image(rt_main_img, img_height, max_width=max_img_width)
                    h, w = rt_main_img.shape[:2]
                    frame[rt_images_y:rt_images_y+h, current_x_offset:current_x_offset+w] = rt_main_img
                    
                    # 添加RT/Sim标签
                    if mode_name.startswith("Sim"):
                        cv2.putText(frame, "Sim Main", (current_x_offset, rt_images_y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)  # 橙色代表Sim
                    else:
                        cv2.putText(frame, "RT Main", (current_x_offset, rt_images_y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    current_x_offset += w + spacing
            
            # Load and resize realtime wrist image
            if rt_wrist_img_path is not None:
                rt_wrist_img = cv2.imread(str(rt_wrist_img_path))
                if rt_wrist_img is not None:
                    max_img_width = available_img_width // 2 - spacing
                    rt_wrist_img = resize_image(rt_wrist_img, img_height, max_width=max_img_width)
                    h, w = rt_wrist_img.shape[:2]
                    frame[rt_images_y:rt_images_y+h, current_x_offset:current_x_offset+w] = rt_wrist_img
                    
                    # 添加RT/Sim标签
                    if mode_name.startswith("Sim"):
                        cv2.putText(frame, "Sim Wrist", (current_x_offset, rt_images_y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)  # 橙色代表Sim
                    else:
                        cv2.putText(frame, "RT Wrist", (current_x_offset, rt_images_y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 第二层：主图像（主相机和手腕图片并列）
        if has_rt_images:
            replay_images_y = images_start_y + img_height + spacing
        else:
            replay_images_y = images_start_y
            
        current_x_offset = images_start_x
        
        # Load and resize main image
        main_img = cv2.imread(str(img_path))
        if main_img is not None:
            # 为并列显示调整图像大小
            max_img_width = available_img_width // 2 - spacing
            main_img = resize_image(main_img, img_height, max_width=max_img_width)
            h, w = main_img.shape[:2]
            frame[replay_images_y:replay_images_y+h, current_x_offset:current_x_offset+w] = main_img
            
            # 根据模式添加合适的标签
            if mode_name.startswith("RT-Only"):
                cv2.putText(frame, "RT Main", (current_x_offset, replay_images_y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif is_replay_mode:
                cv2.putText(frame, "Replay Main", (current_x_offset, replay_images_y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            current_x_offset += w + spacing
        
        # Load and resize wrist image (右侧并列)
        if wrist_img_path.exists():
            wrist_img = cv2.imread(str(wrist_img_path))
            if wrist_img is not None:
                max_img_width = available_img_width // 2 - spacing
                wrist_img = resize_image(wrist_img, img_height, max_width=max_img_width)
                h, w = wrist_img.shape[:2]
                frame[replay_images_y:replay_images_y+h, current_x_offset:current_x_offset+w] = wrist_img
                
                # 根据模式添加合适的标签
                if mode_name.startswith("RT-Only"):
                    cv2.putText(frame, "RT Wrist", (current_x_offset, replay_images_y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif is_replay_mode:
                    cv2.putText(frame, "Replay Wrist", (current_x_offset, replay_images_y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 最后一层：对比图表
        if is_replay_mode and comparison_plot_path.exists():
            comparison_img = cv2.imread(str(comparison_plot_path))
            if comparison_img is not None:
                # 对比图表放在最下面
                if has_rt_images:
                    # RT图像 + Replay图像 + 对比图表
                    bottom_y = images_start_y + (img_height + spacing) * 2
                else:
                    # Replay图像 + 对比图表
                    bottom_y = replay_images_y + img_height + spacing
                    
                max_comparison_width = available_img_width  # 使用完整宽度
                max_comparison_height = img_area_height + 100  # 增加高度
                
                # 首先按高度缩放，保持宽高比
                comparison_img = resize_image(comparison_img, max_comparison_height, max_width=max_comparison_width)
                h, w = comparison_img.shape[:2]
                
                # 居中放置对比图表
                center_x = images_start_x + (available_img_width - w) // 2
                
                # 确保图表不会超出视频边界
                if bottom_y + h > video_height:
                    h = video_height - bottom_y - spacing
                    comparison_img = comparison_img[:h, :, :]
                if center_x + w > video_width:
                    w = video_width - center_x - spacing
                    comparison_img = comparison_img[:, :w, :]
                    
                if h > 0 and w > 0:  # 确保有效尺寸
                    frame[bottom_y:bottom_y+h, center_x:center_x+w] = comparison_img
        
        # 准备文字内容 - 分为两列
        # 左列：基本信息和状态
        left_text_lines = []
        # 右列：动作数据
        right_text_lines = []
        
        # Determine mode based on available files
        has_comparison = actions_pred_path.exists() and actions_gt_path.exists()
        display_mode = mode_name  # 使用智能检测的模式名称
        
        # Add frame number, mode and timestamp to left column
        left_text_lines.extend([
            f"# Frame {frame_idx}",
            ""
        ])
        
        timestamp_str = run.name  # YYYYMMDD_HHMMSS_mmm format (with milliseconds)
        try:
            # Handle both old format (YYYYMMDD_HHMMSS) and new format (YYYYMMDD_HHMMSS_mmm)
            if '_' in timestamp_str and len(timestamp_str.split('_')) >= 2:
                parts = timestamp_str.split('_')
                date_part = parts[0]  # YYYYMMDD
                time_part = parts[1]  # HHMMSS
                ms_part = parts[2] if len(parts) > 2 else None  # mmm (milliseconds)
                
                formatted_date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
                formatted_time = f"{time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}"
                
                if ms_part:
                    formatted_time += f".{ms_part}"
                
                left_text_lines.extend([
                    f"Mode: {display_mode}",
                    f"Time: {formatted_date}",
                    f"      {formatted_time}",
                    ""
                ])
            else:
                # Fallback for old format
                left_text_lines.extend([
                    f"Mode: {display_mode}",
                    f"Time: {timestamp_str}",
                    ""
                ])
        except:
            left_text_lines.extend([
                f"Mode: {display_mode}",
                f"Time: {timestamp_str}",
                ""
            ])
        
        # Add prompt to left column
        if prompt_path.exists():
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
            left_text_lines.extend([
                "Task Prompt:",
                prompt,  # 不再强制截断，让自动换行处理
                ""
            ])
        
        # Add state information to left column - Both模式和Sim模式下显示两个robot state
        if state_path.exists():
            state = load_json_array(state_path)
            if (mode_name.startswith("Both") or mode_name.startswith("Sim")) and rt_runs:
                # Both模式和Sim模式：先显示RT/Sim robot state (在上)，再显示Replay robot state (在下)
                
                # 1. 获取对应的RT/Sim robot state
                current_run_index = runs.index(run)
                sorted_rt_runs = sorted(rt_runs.items())
                if current_run_index < len(sorted_rt_runs):
                    rt_run_name, rt_run = sorted_rt_runs[current_run_index]
                    rt_state_path = rt_run / "state.json"
                    if rt_state_path.exists():
                        rt_state = load_json_array(rt_state_path)
                        if mode_name.startswith("Sim"):
                            left_text_lines.extend(format_array_text(rt_state, "Robot State (Sim)"))
                        else:
                            left_text_lines.extend(format_array_text(rt_state, "Robot State (RT)"))
                        left_text_lines.append("")
                    else:
                        if mode_name.startswith("Sim"):
                            left_text_lines.extend([
                                "Robot State (Sim): Not available",
                                ""
                            ])
                        else:
                            left_text_lines.extend([
                                "Robot State (RT): Not available",
                                ""
                            ])
                
                # 2. 显示Replay robot state
                left_text_lines.extend(format_array_text(state, "Robot State (Replay)"))
                left_text_lines.append("")
                
            else:
                # 其他模式：显示单个robot state
                state_label = "Robot State"
                if mode_name.startswith("RT-Only"):
                    state_label = "Robot State (RT)"
                elif mode_name.startswith("Sim"):
                    state_label = "Robot State (Simulation)"
                left_text_lines.extend(format_array_text(state, state_label))
                left_text_lines.append("")
        else:
            left_text_lines.extend([
                "Robot State: Not available",
                ""
            ])
        
        # Add realtime image status to left column
        if has_rt_images:  # 使用当前run的RT图像检查结果而不是全局变量
            left_text_lines.extend([
                "Realtime Images:",
                f"Main: {'True' if rt_img_path and rt_img_path.exists() else 'False'}",
                f"Wrist: {'True' if rt_wrist_img_path and rt_wrist_img_path.exists() else 'False'}",
                ""
            ])
        
        # Add actions information to right column
        predicted_actions = None
        gt_actions = None
        
        if has_comparison:
            right_text_lines.extend([
                "=== REPLAY MODE ===",
                "Model vs Dataset:",
                ""
            ])
            
            # Load predicted actions
            predicted_actions = load_json_array(actions_pred_path)
            if predicted_actions is not None and len(predicted_actions) == 70:
                predicted_actions = predicted_actions[:7]  # 只取第一个时间步的7个动作
            if predicted_actions is not None:
                right_text_lines.extend(format_array_text_grouped(predicted_actions, "Model Predictions"))
                right_text_lines.append("")
            
            # Load ground truth actions
            gt_actions = load_json_array(actions_gt_path)
            if gt_actions is not None:
                right_text_lines.extend(format_array_text_grouped(gt_actions, "Ground Truth"))
                right_text_lines.append("")
            
            # Add differences
            if predicted_actions is not None and gt_actions is not None:
                diff = predicted_actions - gt_actions
                right_text_lines.extend(format_array_text_grouped(diff, "Difference"))
                right_text_lines.append("")
            
            # Add comparison metrics
            metrics_path = run / "comparison_metrics.json"
            if metrics_path.exists():
                try:
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                    right_text_lines.extend([
                        "Performance Metrics:",
                        f"MSE: {metrics.get('mse', 'N/A'):.8f}",
                        f"MAE: {metrics.get('mae', 'N/A'):.8f}",
                        f"RMSE: {metrics.get('rmse', 'N/A'):.8f}",
                        f"Corr: {metrics.get('correlation', 'N/A'):.6f}",
                        f"MaxDiff: {metrics.get('max_difference', 'N/A'):.8f}",
                        ""
                    ])
                except Exception as e:
                    right_text_lines.extend([f"Metrics Error: {e}", ""])
        
        elif actions_path.exists():
            right_text_lines.extend([
                "=== REAL-TIME TEST ===",
                "Live Predictions:",
                ""
            ])
            # Load single actions file
            actions = load_json_array(actions_path)
            if actions is not None:
                right_text_lines.extend(format_array_text_grouped(actions, "Model Output"))
        else:
            right_text_lines.extend([
                "=== NO ACTIONS ===",
                "No action data found",
                ""
            ])
        
        # 创建并放置文字图像 - 智能分列处理，向左移动文字区域
        right_area_x_start = left_area_width - spacing  # 向左移动，减少间距
        
        # 左列文字 (基本信息)
        left_text_img = create_text_image(left_text_lines, text_column_width, video_height - spacing * 2)
        text_y_start = spacing
        frame[text_y_start:text_y_start + left_text_img.shape[0], 
              right_area_x_start:right_area_x_start + text_column_width] = left_text_img
        
        # 在both模式下，为关节对比图预留空间
        chart_height = 400  # 增大图表高度
        chart_width = 500   # 增大图表宽度
        chart_reserved_space = chart_height + spacing if has_comparison else 0
        
        # 智能分割右侧文字内容 - 根据实际可显示行数计算（减去图表空间）
        line_height = 22
        available_height = video_height - spacing * 2 - 30 - chart_reserved_space  # 减去图表空间
        max_lines_per_column = int(available_height // line_height) - 2  # 留出一些缓冲
        
        print(f"Available height: {available_height}, Max lines per column: {max_lines_per_column}")
        print(f"Total right text lines: {len(right_text_lines)}")
        
        # 计算需要多少列
        num_columns_needed = (len(right_text_lines) + max_lines_per_column - 1) // max_lines_per_column if right_text_lines else 0
        print(f"Need {num_columns_needed} columns for right text")
        
        # 分割文字到多列
        text_columns = []
        for i in range(num_columns_needed):
            start_idx = i * max_lines_per_column
            end_idx = min((i + 1) * max_lines_per_column, len(right_text_lines))
            text_columns.append(right_text_lines[start_idx:end_idx])
            print(f"Column {i+1}: lines {start_idx} to {end_idx-1} ({len(text_columns[i])} lines)")
        
        # 渲染各列 - 更紧凑的布局
        current_x = right_area_x_start + text_column_width + spacing // 4  # 减少列间距
        available_width_for_columns = video_width - current_x - spacing // 2  # 减少右边距
        
        for i, column_text in enumerate(text_columns):
            if not column_text:
                continue
                
            # 计算列宽 - 更均匀的分配
            if i == len(text_columns) - 1:  # 最后一列使用剩余空间
                column_width = min(text_column_width, video_width - current_x - spacing // 2)
            else:
                column_width = text_column_width
            
            if column_width > 100:  # 确保有足够空间显示文字
                text_img = create_text_image(column_text, column_width, video_height - spacing * 2)
                end_x = min(current_x + column_width, video_width - spacing // 2)
                
                if current_x < video_width - spacing // 2:
                    frame[text_y_start:text_y_start + text_img.shape[0], 
                          current_x:end_x] = text_img[:, :end_x-current_x]
                    print(f"Rendered column {i+1} at x={current_x}, width={end_x-current_x}")
                    
                current_x += column_width + spacing // 4  # 移动到下一列
        
        # 在both模式下添加关节对比图
        if has_comparison and predicted_actions is not None and gt_actions is not None:
            # 创建关节对比图
            joint_chart = create_joint_comparison_chart(predicted_actions, gt_actions, chart_width, chart_height)
            
            # 将图表放置在右侧区域底部
            chart_x = right_area_x_start + text_column_width + spacing
            chart_y = video_height - chart_height - spacing
            
            # 确保图表不超出边界
            end_x = min(chart_x + chart_width, video_width - spacing)
            end_y = min(chart_y + chart_height, video_height - spacing)
            actual_width = end_x - chart_x
            actual_height = end_y - chart_y
            
            if actual_width > 0 and actual_height > 0:
                frame[chart_y:end_y, chart_x:end_x] = joint_chart[:actual_height, :actual_width]
            else:
                print(f"Skipping column {i+1} - insufficient width ({column_width})")
                break
        
        video.write(frame)

    video.release()
    print(f"Video saved to {output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create video from logs')
    parser.add_argument('--logs_dir', type=str, default='../logs', help='Parent logs directory')
    parser.add_argument('--output', type=str, default='../logs/output.mp4', help='Output video file path')
    parser.add_argument('--fps', type=int, default=2, help='Frames per second')
    args = parser.parse_args()
    main(Path(args.logs_dir), Path(args.output), args.fps)
