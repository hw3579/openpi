# 假设 img, wrist_img, obs_state, instruction 已经准备好

import numpy as np

def convert_example_to_policy_input(example, action_dim=7, model_type="PI0-FAST"):
    # 1. pad state
    def pad_to_dim(arr, dim):
        arr = np.asarray(arr)
        if arr.shape[-1] >= dim:
            return arr
        pad_width = [(0, 0)] * arr.ndim
        pad_width[-1] = (0, dim - arr.shape[-1])
        return np.pad(arr, pad_width, mode='constant')

    # 2. parse image (如需转 uint8，可加 image_tools.convert_to_uint8)
    def _parse_image(img):
        return img

    mask_padding = model_type == "PI0"

    state = pad_to_dim(example["observation/state"], action_dim)
    base_image = _parse_image(example["observation/image"])
    wrist_image = _parse_image(example["observation/wrist_image"])

    inputs = {
        "state": state,
        "image": {
            "base_0_rgb": base_image,
            "left_wrist_0_rgb": wrist_image,
            "right_wrist_0_rgb": np.zeros_like(base_image),
        },
        "image_mask": {
            "base_0_rgb": True,
            "left_wrist_0_rgb": True,
            "right_wrist_0_rgb": False if mask_padding else True,
        },
    }
    if "prompt" in example:
        inputs["prompt"] = example["prompt"]
    if "actions" in example:
        inputs["actions"] = pad_to_dim(example["actions"], action_dim)
    return inputs

def save_log_realtime(example: dict, result: dict, logs_dir: str = "../logs/rt") -> str:
    """
    Save realtime inference data to a timestamped folder under logs_dir.
    This function only handles data recording, no computation or plotting.
    
    Args:
        example: Input example data with observation/image, observation/wrist_image, observation/state, prompt
        result: Model prediction result with actions
        logs_dir: Directory to save logs (default: ../logs/rt for realtime)
    
    Returns:
        Path to the created log directory
    """
    import os, time, json
    import numpy as np
    import cv2
    from pathlib import Path
    
    # Ensure logs_dir exists
    base_dir = Path(logs_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a timestamped subdirectory with milliseconds precision
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    milliseconds = int((time.time() % 1) * 1000)  # Get milliseconds
    timestamp_with_ms = f"{timestamp}_{milliseconds:03d}"
    run_dir = base_dir / timestamp_with_ms
    run_dir.mkdir(exist_ok=True)
    
    # Save images
    img = example.get("observation/image")
    wrist = example.get("observation/wrist_image")
    rt_img = example.get("observation/rt_image")  # 实时图像
    rt_wrist = example.get("observation/rt_wrist_image")  # 实时手腕图像
    
    if img is not None:
        img_converted = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(run_dir / "image.png"), img_converted)
    if wrist is not None:
        wrist_converted = cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(run_dir / "wrist_image.png"), wrist_converted)
    
    # 保存实时图像（如果存在）
    if rt_img is not None:
        rt_img_converted = cv2.cvtColor(rt_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(run_dir / "rt_image.png"), rt_img_converted)
    if rt_wrist is not None:
        rt_wrist_converted = cv2.cvtColor(rt_wrist, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(run_dir / "rt_wrist_image.png"), rt_wrist_converted)
    
    # Save state as JSON
    state = example.get("observation/state")
    if state is not None and len(state) == 8:
        state = np.hstack([state[:6], state[7]])
    if state is not None:
        state_data = {"state": np.asarray(state).tolist()}
        with open(run_dir / "state.json", "w") as f:
            json.dump(state_data, f, indent=2)
    
    # Save predicted actions as JSON
    actions = result.get("actions")
    if actions is not None:
        actions_data = {"actions": np.asarray(actions).tolist()}
        with open(run_dir / "actions.json", "w") as f:
            json.dump(actions_data, f, indent=2)
    
    # Save prompt and metadata
    prompt = example.get("prompt", "")
    with open(run_dir / "prompt.txt", "w") as f:
        f.write(str(prompt))
    
    # Save basic result JSON (realtime mode marker)
    result_data = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in result.items()}
    result_data["mode"] = "realtime"
    with open(run_dir / "result.json", "w") as f:
        json.dump(result_data, f, indent=2)
    
    return str(run_dir)

def save_log_replay(example: dict, result: dict, ground_truth_actions: np.ndarray = None, 
                   logs_dir: str = "../logs/replay", comparison_name: str = "replay") -> str:
    """
    Save replay data with ground truth comparison to a timestamped folder under logs_dir.
    This function only handles data recording, no computation or plotting.
    
    Args:
        example: Input example data
        result: Model prediction result
        ground_truth_actions: Ground truth actions for comparison (optional)
        logs_dir: Directory to save logs (default: ../logs/replay for replay mode)
        comparison_name: Name for the comparison (e.g., "replay", "human", "ground_truth")
    
    Returns:
        Path to the created log directory
    """
    import os, time, json
    import numpy as np
    import cv2
    from pathlib import Path
    
    # Ensure logs_dir exists
    base_dir = Path(logs_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a timestamped subdirectory with milliseconds precision
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    milliseconds = int((time.time() % 1) * 1000)  # Get milliseconds
    timestamp_with_ms = f"{timestamp}_{milliseconds:03d}"
    run_dir = base_dir / timestamp_with_ms
    run_dir.mkdir(exist_ok=True)
    
    # Save images
    img = example.get("observation/image")
    wrist = example.get("observation/wrist_image")
    rt_img = example.get("observation/rt_image")  # 实时图像
    rt_wrist = example.get("observation/rt_wrist_image")  # 实时手腕图像
    
    if img is not None:
        img_converted = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(run_dir / "image.png"), img_converted)
    if wrist is not None:
        wrist_converted = cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(run_dir / "wrist_image.png"), wrist_converted)
    
    # 保存实时图像（如果存在）
    if rt_img is not None:
        rt_img_converted = cv2.cvtColor(rt_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(run_dir / "rt_image.png"), rt_img_converted)
    if rt_wrist is not None:
        rt_wrist_converted = cv2.cvtColor(rt_wrist, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(run_dir / "rt_wrist_image.png"), rt_wrist_converted)
    
    # Save state as JSON
    state = example.get("observation/state")
    if state is not None and len(state) == 8:
        state = np.hstack([state[:6], state[7]])
    if state is not None:
        state_data = {"state": np.asarray(state).tolist()}
        with open(run_dir / "state.json", "w") as f:
            json.dump(state_data, f, indent=2)
    
    # Save predicted actions as JSON
    predicted_actions = result.get("actions")
    if predicted_actions is not None:
        predicted_actions = np.asarray(predicted_actions)
        actions_data = {"actions_predicted": predicted_actions.tolist()}
        with open(run_dir / "actions_predicted.json", "w") as f:
            json.dump(actions_data, f, indent=2)
    
    # Save ground truth actions as JSON if provided (no computation/plotting here)
    if ground_truth_actions is not None:
        ground_truth_actions = np.asarray(ground_truth_actions)
        gt_data = {"actions_ground_truth": ground_truth_actions.tolist()}
        with open(run_dir / "actions_ground_truth.json", "w") as f:
            json.dump(gt_data, f, indent=2)
    
    # Save prompt and metadata
    prompt = example.get("prompt", "")
    with open(run_dir / "prompt.txt", "w") as f:
        f.write(str(prompt))
    
    # Save result JSON with replay mode marker
    result_data = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in result.items()}
    result_data["mode"] = "replay"
    result_data["has_ground_truth"] = ground_truth_actions is not None
    result_data["comparison_name"] = comparison_name
    if ground_truth_actions is not None:
        result_data["ground_truth_actions"] = ground_truth_actions.tolist()
    
    with open(run_dir / "result.json", "w") as f:
        json.dump(result_data, f, indent=2)
    
    return str(run_dir)

def _create_action_comparison_plot(predicted_actions, ground_truth_actions, save_path, comparison_name="ground_truth"):
    """Create a comparison plot between predicted and ground truth actions."""
    import matplotlib.pyplot as plt
    
    # Ensure both arrays have the same shape for comparison
    min_len = min(len(predicted_actions), len(ground_truth_actions))
    pred = predicted_actions[:min_len] if len(predicted_actions.shape) == 1 else predicted_actions[0][:min_len]
    gt = ground_truth_actions[:min_len] if len(ground_truth_actions.shape) == 1 else ground_truth_actions[0][:min_len]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Bar comparison
    x_labels = [f"Action_{i}" for i in range(len(pred))]
    x_pos = np.arange(len(x_labels))
    
    width = 0.35
    ax1.bar(x_pos - width/2, pred, width, label='Predicted', alpha=0.8, color='blue')
    ax1.bar(x_pos + width/2, gt, width, label=f'{comparison_name.title()}', alpha=0.8, color='red')
    
    ax1.set_xlabel('Action Dimensions')
    ax1.set_ylabel('Action Values')
    ax1.set_title('Action Comparison: Predicted vs Ground Truth')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Difference plot
    diff = pred - gt
    ax2.bar(x_pos, diff, color='green', alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Action Dimensions')
    ax2.set_ylabel('Difference (Predicted - Ground Truth)')
    ax2.set_title('Action Differences')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels, rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def _calculate_action_metrics(predicted_actions, ground_truth_actions):
    """Calculate comparison metrics between predicted and ground truth actions."""
    import numpy as np
    
    # Ensure both arrays have the same shape for comparison
    min_len = min(len(predicted_actions), len(ground_truth_actions))
    pred = predicted_actions[:min_len] if len(predicted_actions.shape) == 1 else predicted_actions[0][:min_len]
    gt = ground_truth_actions[:min_len] if len(ground_truth_actions.shape) == 1 else ground_truth_actions[0][:min_len]
    
    # Calculate various metrics
    mse = np.mean((pred - gt) ** 2)
    mae = np.mean(np.abs(pred - gt))
    rmse = np.sqrt(mse)
    
    # Per-dimension metrics
    per_dim_mse = ((pred - gt) ** 2).tolist()
    per_dim_mae = np.abs(pred - gt).tolist()
    
    # Correlation coefficient
    correlation = np.corrcoef(pred, gt)[0, 1] if len(pred) > 1 else 0.0
    
    # Max difference
    max_diff = np.max(np.abs(pred - gt))
    
    metrics = {
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
        "correlation": float(correlation),
        "max_difference": float(max_diff),
        "per_dimension_mse": per_dim_mse,
        "per_dimension_mae": per_dim_mae,
        "predicted_actions": pred.tolist(),
        "ground_truth_actions": gt.tolist(),
        "action_differences": (pred - gt).tolist()
    }
    
    return metrics

def create_video_from_logs(logs_dir: str = "../logs", output_path: str = "../output.mp4", fps: int = 2, mode: str = "auto"):
    """
    Create a video from saved logs using the create_video.py script.
    This function now handles all computation, plotting, and comparison functionality.
    
    Args:
        logs_dir: Directory containing the log subdirectories (e.g., "../logs/rt" or "../logs/replay")
        output_path: Path where the output video will be saved
        fps: Frames per second for the video
        mode: Video mode - "auto" (detect automatically), "realtime" (realtime test), or "replay" (with ground truth)
    """
    import subprocess
    import sys
    from pathlib import Path
    
    # Get the directory of the current script
    current_dir = Path(__file__).parent
    create_video_script = current_dir / "create_video.py"
    
    if not create_video_script.exists():
        print(f"Error: create_video.py not found at {create_video_script}")
        return False
    
    # Auto-detect mode based on log contents and directory structure if mode is "auto"
    if mode == "auto":
        logs_path = Path(logs_dir)
        if logs_path.exists():
            # 智能检测：优先通过目录名判断
            dir_name = logs_path.name.lower()
            if dir_name == "rt":
                mode = "realtime"
                print(f"Auto-detected mode: {mode} (RT-Only)")
            elif dir_name == "replay":
                mode = "replay"
                print(f"Auto-detected mode: {mode} (Replay-Only)")
            elif dir_name == "both":
                mode = "replay"  # Both模式也使用replay处理逻辑
                print(f"Auto-detected mode: {mode} (Both Mode)")
            else:
                # 如果目录名不明确，则检查文件内容
                has_comparison_files = False
                has_rt_images = False
                for subdir in logs_path.iterdir():
                    if subdir.is_dir():
                        if (subdir / "actions_predicted.json").exists() and (subdir / "actions_ground_truth.json").exists():
                            has_comparison_files = True
                        if (subdir / "rt_image.png").exists() or (subdir / "rt_wrist_image.png").exists():
                            has_rt_images = True
                        if has_comparison_files:  # 找到一个就够了
                            break
                            
                if has_comparison_files:
                    mode = "replay"
                    mode_desc = "Both Mode" if has_rt_images else "Replay-Only"
                    print(f"Auto-detected mode: {mode} ({mode_desc})")
                else:
                    mode = "realtime"
                    mode_desc = "RT-Only" if has_rt_images else "Real-time Test"
                    print(f"Auto-detected mode: {mode} ({mode_desc})")
    else:
        print(f"Manual mode specified: {mode}")
    
    # Before creating video, process logs for computation and plotting if needed
    if mode == "replay":
        _process_replay_logs_for_video(logs_dir)
    
    try:
        # Run the create_video.py script
        cmd = [
            sys.executable, str(create_video_script),
            "--logs_dir", logs_dir,
            "--output", output_path,
            "--fps", str(fps)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Video created successfully: {output_path}")
            print(f"Mode: {mode.upper()}")
            return True
        else:
            print(f"Error creating video: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Error running create_video.py: {e}")
        return False

def _process_replay_logs_for_video(logs_dir: str):
    """
    Process replay logs to generate comparison plots and metrics before video creation.
    This function handles all computation and plotting that was previously done in save_log_replay.
    """
    from pathlib import Path
    import json
    import numpy as np
    
    logs_path = Path(logs_dir)
    if not logs_path.exists():
        print(f"Warning: Logs directory {logs_dir} does not exist")
        return
    
    print(f"Processing replay logs for comparison analysis...")
    processed_count = 0
    
    for run_dir in logs_path.iterdir():
        if not run_dir.is_dir():
            continue
        
        # Check if this run has both predicted and ground truth actions
        pred_file = run_dir / "actions_predicted.json"
        gt_file = run_dir / "actions_ground_truth.json"
        comparison_plot = run_dir / "action_comparison.png"
        metrics_file = run_dir / "comparison_metrics.json"
        
        if pred_file.exists() and gt_file.exists():
            # Skip if already processed
            if comparison_plot.exists() and metrics_file.exists():
                continue
                
            try:
                # Load predicted actions
                with open(pred_file, 'r') as f:
                    pred_data = json.load(f)
                predicted_actions = np.array(pred_data.get("actions_predicted", []))
                
                # Load ground truth actions
                with open(gt_file, 'r') as f:
                    gt_data = json.load(f)
                ground_truth_actions = np.array(gt_data.get("actions_ground_truth", []))
                
                # Get comparison name from result.json if available
                result_file = run_dir / "result.json"
                comparison_name = "ground_truth"
                if result_file.exists():
                    with open(result_file, 'r') as f:
                        result_data = json.load(f)
                        comparison_name = result_data.get("comparison_name", "ground_truth")
                
                # Create action comparison plot
                _create_action_comparison_plot(
                    predicted_actions, ground_truth_actions, 
                    str(comparison_plot), comparison_name
                )
                
                # Calculate and save comparison metrics
                metrics = _calculate_action_metrics(predicted_actions, ground_truth_actions)
                with open(metrics_file, "w") as f:
                    json.dump(metrics, f, indent=2)
                
                processed_count += 1
                print(f"  Processed comparison for {run_dir.name}")
                
            except Exception as e:
                print(f"  Error processing {run_dir.name}: {e}")
    
    print(f"Processed {processed_count} replay logs for comparison analysis")

# 向后兼容性包装函数
def save_log(example: dict, result: dict, logs_dir: str = "../logs/rt") -> str:
    """
    向后兼容的save_log函数，现在默认使用realtime模式
    """
    return save_log_realtime(example, result, logs_dir)

def save_log_with_comparison(example: dict, result: dict, ground_truth_actions: np.ndarray = None, 
                           logs_dir: str = "../logs/replay", comparison_name: str = "replay") -> str:
    """
    向后兼容的save_log_with_comparison函数，现在使用replay模式
    """
    return save_log_replay(example, result, ground_truth_actions, logs_dir, comparison_name)

# 用法示例 - 重构后的新版本
# 
# 1. 实时测试模式（realtime，没有ground truth）：
# log_dir = save_log_realtime(example, result)  # 保存到 ../logs/rt/
# create_video_from_logs(logs_dir="../logs/rt", output_path="../rt_video.mp4")
#
# 2. 回放模式（replay，有ground truth对比）：
# log_dir = save_log_replay(example, result, ground_truth_actions)  # 保存到 ../logs/replay/
# create_video_from_logs(logs_dir="../logs/replay", output_path="../replay_video.mp4")
#
# 3. 自动检测模式：
# create_video_from_logs(logs_dir="../logs/rt", output_path="../video.mp4", mode="auto")  # 自动检测模式
#
# 4. 向后兼容（旧代码仍然可以使用）：
# log_dir = save_log(example, result)  # 现在默认保存到 ../logs/rt
# log_dir = save_log_with_comparison(example, result, ground_truth_actions)  # 现在默认保存到 ../logs/replay
#
# policy_input = convert_example_to_policy_input(example)

