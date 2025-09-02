#!/usr/bin/env python3
"""
Parquet replay server that         print(f"💾 数据保存: 只保存parquet数据到../logs/replay/目录")
            
    def setup_routes(self):erves robot data to franka_test_droid client.
This server loads parquet data and provides HTTP endpoints for franka_test_droid to request data.

Usage:
  python replay_parquet.py --parquet_file episode_000001.parquet --port 9001
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from flask import Flask, Response, jsonify, request
from tools import save_log_replay
import threading
import time
import re


class ParquetReplayServer:
    def __init__(self, parquet_file, port=9001, zero_state=False, mode="replay-only", rt_server="http://192.168.191.210:9001"):
        """
        Initialize the parquet replay server.
        
        Args:
            parquet_file: Path to the parquet file
            port: Server port to listen on
            zero_state: 是否使用零状态
            mode: 运行模式 ("replay-only", "both") - 注意：不再支持"rt-only"
            rt_server: 实时图像服务器地址（保留参数但不使用）
        """
        self.parquet_file = Path(parquet_file)
        self.port = port
        self.data = None
        self.current_index = 0
        self.pending_action = None  # 存储待保存的推理动作
        self.app = Flask(__name__)
        self.zero_state = zero_state  # 是否使用零状态
        self.mode = mode  # 运行模式
        self.rt_server = rt_server  # 实时图像服务器地址（保留但不使用）
        
        # 根据模式设置功能标志
        self.use_parquet = mode in ["replay-only", "both"]
        
        if mode == "rt-only":
            print("⚠️  replay_parquet.py不再支持rt-only模式，请直接使用franka_test_droid.py")
            return
        
        if zero_state:
            print("🔧 Using zero state for robot state")
        
        print(f"🎯 运行模式: {mode.upper()}")
        print(f"📊 将使用parquet数据: {parquet_file}")
        print(f"� 数据保存: 只保存parquet数据到../logs/replay/目录")
            
    def get_realtime_images(self):
        """从实时服务器获取图像"""
        if not self.use_rt_images:
            return None, None
            
        try:
            import requests
            
            # 获取主图像
            resp = requests.get(f"{self.rt_server}/image_raw", timeout=2)
            if resp.status_code == 200:
                headers = resp.headers
                height = int(headers["X-Height"])
                width = int(headers["X-Width"])
                channels = int(headers["X-Channels"])
                dtype = np.dtype(headers["X-Dtype"])
                rt_img = np.frombuffer(resp.content, dtype=dtype).reshape((height, width, channels))
            else:
                rt_img = None
                
            # 获取手腕图像
            resp = requests.get(f"{self.rt_server}/image_wrist", timeout=2)
            if resp.status_code == 200:
                headers = resp.headers
                height = int(headers["X-Height"])
                width = int(headers["X-Width"])
                channels = int(headers["X-Channels"])
                dtype = np.dtype(headers["X-Dtype"])
                rt_wrist_img = np.frombuffer(resp.content, dtype=dtype).reshape((height, width, channels))
            else:
                rt_wrist_img = None
                
            return rt_img, rt_wrist_img
            
        except Exception as e:
            print(f"Warning: Failed to get realtime images: {e}")
            return None, None
        
    def setup_routes(self):
        """Setup Flask routes for serving robot data."""
        
        @self.app.route('/image_raw', methods=['GET'])
        def get_image_raw():
            """Serve main camera image."""            
            if not self.use_parquet:
                return jsonify({"error": "Parquet data not enabled in current mode"}), 404
                
            if self.data is None or len(self.data) == 0:
                return jsonify({"error": "No data loaded"}), 404
            
            if self.current_index >= len(self.data):
                print(f"\n✅ 所有 {len(self.data)} 帧已处理完成！")
                print("🛑 自动关闭服务器...")
                # 启动关闭线程
                def shutdown_server():
                    time.sleep(1)  # 给点时间完成当前响应
                    import os
                    os._exit(0)
                
                shutdown_thread = threading.Thread(target=shutdown_server, daemon=True)
                shutdown_thread.start()
                return jsonify({"error": "All frames completed", "shutdown": True}), 404
                
            row = self.data.iloc[self.current_index]
            if 'image' not in row or row['image'] is None:
                return jsonify({"error": "No image data"}), 404
                
            # Process image data
            image_data = self.process_image_data(row['image'])
            
            print(f"📸 {self.mode.upper()}: 发送第 {self.current_index + 1}/{len(self.data)} 帧图像")
            
            # Create response with headers
            response = Response(
                image_data.tobytes(),
                mimetype='application/octet-stream'
            )
            response.headers['X-Height'] = str(image_data.shape[0])
            response.headers['X-Width'] = str(image_data.shape[1])
            response.headers['X-Channels'] = str(image_data.shape[2])
            response.headers['X-Dtype'] = str(image_data.dtype)
            
            return response
        
        @self.app.route('/image_wrist', methods=['GET'])
        def get_image_wrist():
            """Serve wrist camera image."""
            if not self.use_parquet:
                return jsonify({"error": "Parquet data not enabled in current mode"}), 404
                
            if self.data is None or len(self.data) == 0:
                return jsonify({"error": "No data loaded"}), 404
            
            if self.current_index >= len(self.data):
                self.current_index = 0  # Loop back to start
                
            row = self.data.iloc[self.current_index]
            if 'wrist_image' not in row or row['wrist_image'] is None:
                return jsonify({"error": "No wrist image data"}), 404
                
            # Process image data
            image_data = self.process_image_data(row['wrist_image'])
            
            print(f"🤏 {self.mode.upper()}: 发送第 {self.current_index + 1}/{len(self.data)} 帧腕部图像")
            
            # Create response with headers
            response = Response(
                image_data.tobytes(),
                mimetype='application/octet-stream'
            )
            response.headers['X-Height'] = str(image_data.shape[0])
            response.headers['X-Width'] = str(image_data.shape[1])
            response.headers['X-Channels'] = str(image_data.shape[2])
            response.headers['X-Dtype'] = str(image_data.dtype)
            
            return response
        
        @self.app.route('/robot_state_ee', methods=['GET'])
        def get_robot_state():
            """Serve robot state."""
            if not self.use_parquet:
                return jsonify({"error": "Parquet data not enabled in current mode"}), 404
                
            if self.data is None or len(self.data) == 0:
                return jsonify({"error": "No data loaded"}), 404
            
            if self.current_index >= len(self.data):
                self.current_index = 0  # Loop back to start
                
            row = self.data.iloc[self.current_index]
            if 'state' not in row or row['state'] is None:
                return jsonify({"error": "No state data"}), 404
                
            # Get state data
            try:
                if self.zero_state:
                    state_data = np.zeros(7)
                    print(f"🤖 {self.mode.upper()}: 返回零状态 (第 {self.current_index + 1}/{len(self.data)} 帧)")
                else:
                    state_data = np.array(row['state'][:7])  # 取前7个状态值
                    print(f"🤖 {self.mode.upper()}: 返回状态 (第 {self.current_index + 1}/{len(self.data)} 帧): {state_data}")
                    
                return jsonify({"obs_state": state_data.tolist()})
                
            except Exception as e:
                print(f"❌ 处理状态数据出错: {e}")
                return jsonify({"error": f"Failed to process state data: {str(e)}"}), 500
                return jsonify({"error": "No state data"}), 404
                
            state_data = np.array(row['state'])
            state_data = np.hstack((state_data[:6], [state_data[7]]))
            
            if self.zero_state:
                state_data = np.zeros(7)
                
            return jsonify({"obs_state": state_data.tolist()})
        
        @self.app.route('/instruction', methods=['GET'])
        def get_instruction():
            """Serve instruction."""
            return jsonify({
                "instruction": "pick up the banana and put it on the plate"
            })
        
        @self.app.route('/action', methods=['POST'])
        def receive_action():
            """接收模型推理的动作结果并保存当前帧"""
            try:
                request_data = request.get_json()
                actions = request_data.get('actions', [])
                print(f"🎯 接收到推理动作: {[f'{x:.3f}' for x in actions] if isinstance(actions, list) else actions}")
                
                # 立即保存当前帧数据
                self.save_current_frame_with_action(actions)
                
                return jsonify({"status": "success", "message": "Action received and saved"})
            except Exception as e:
                print(f"❌ 接收动作失败: {e}")
                return jsonify({"error": "Invalid action data"}), 400
        
    def load_data(self):
        """Load parquet data and analyze structure."""
        if not self.use_parquet:
            print("🎯 RT-Only模式：跳过parquet数据加载")
            return
            
        print(f"正在读取文件: {self.parquet_file}")
        self.data = pd.read_parquet(self.parquet_file)
        
        print(f"数据形状: {self.data.shape}")
        print(f"列名: {list(self.data.columns)}")
        
        # Analyze image data
        if 'image' in self.data.columns:
            sample_image = self.data['image'].iloc[0]
            print(f"\n主视角图像:")
            print(f"  数据类型: {type(sample_image).__name__}")
            if hasattr(sample_image, 'dtype'):
                print(f"  数组类型: {sample_image.dtype}")
            if hasattr(sample_image, 'shape'):
                print(f"  形状: {sample_image.shape}")
            if hasattr(sample_image, '__len__'):
                print(f"  数据长度: {len(sample_image)}")
                
        if 'wrist_image' in self.data.columns:
            sample_wrist = self.data['wrist_image'].iloc[0]
            print(f"\n手腕视角图像:")
            print(f"  数据类型: {type(sample_wrist).__name__}")
            if hasattr(sample_wrist, 'dtype'):
                print(f"  数组类型: {sample_wrist.dtype}")
            if hasattr(sample_wrist, 'shape'):
                print(f"  形状: {sample_wrist.shape}")
            if hasattr(sample_wrist, '__len__'):
                print(f"  数据长度: {len(sample_wrist)}")
        
        print(f"\n数据加载完成，共 {len(self.data)} 条记录")

        if self.zero_state:
            # 如果使用零状态，确保所有状态数据都是零向量
            self.data['state'] = self.data['state'].apply(lambda x: np.zeros(8, dtype=np.float32) if isinstance(x, list) else np.zeros(8, dtype=np.float32))
            print("已将所有状态数据设置为零向量")
        
    def process_image_data(self, image_data):
        """Process image data to ensure correct format."""
        # Convert to numpy array if not already
        if not isinstance(image_data, np.ndarray):
            image_data = np.array(image_data)
        
        # Handle flattened image data (196608,) -> (256, 256, 3)
        if image_data.ndim == 1 and image_data.size == 196608:  # 256 * 256 * 3
            image_data = image_data.reshape(256, 256, 3)
        elif image_data.size == 196608 and image_data.shape != (256, 256, 3):
            image_data = image_data.reshape(256, 256, 3)
        
        # Convert from int64 to uint8 format
        if image_data.dtype == np.int64:
            # Clip values to valid range and convert to uint8
            image_data = np.clip(image_data, 0, 255).astype(np.uint8)
        elif image_data.dtype != np.uint8:
            # Handle other data types
            image_data = np.clip(image_data, 0, 255).astype(np.uint8)
        
        # Ensure the shape is correct
        if image_data.shape != (256, 256, 3):
            print(f"Warning: Unexpected image shape {image_data.shape}, attempting to reshape...")
            if image_data.size == 196608:
                image_data = image_data.reshape(256, 256, 3)
            else:
                raise ValueError(f"Cannot process image with size {image_data.size}")
        
        return image_data
    
    def save_current_frame_with_action(self, predicted_actions):
        """收到action后保存当前帧数据并前进（简化版本）"""
        
        # 只处理parquet数据
        if not self.use_parquet or self.data is None or len(self.data) == 0:
            print("⚠️  无parquet数据可保存")
            return
        
        if self.current_index >= len(self.data):
            print(f"\n✅ 所有 {len(self.data)} 帧已处理完成！")
            return
        
        row = self.data.iloc[self.current_index]

        # 添加 prompt 列
        prompt = "pick_up_the_banana_and_put_it_on_the_plate"
        row['prompt'] = re.sub(r'_', ' ', prompt)
        
        # 准备保存的数据 - 只保存parquet中的数据
        example = {
            "observation/image": self.process_image_data(row['image']),
            "observation/wrist_image": self.process_image_data(row.get('wrist_image', [])),
            "observation/state": np.array(row.get('state', [])),
            "prompt": row.get('prompt', '')
        }
        
        # 使用收到的推理动作
        result = {"actions": predicted_actions}
        
        # parquet中的动作作为ground truth
        ground_truth_actions = row.get('actions', None)
        if ground_truth_actions is not None:
            ground_truth_actions = np.array(ground_truth_actions)
        
        # 保存到replay目录
        saved_path = save_log_replay(
            example, result, ground_truth_actions, 
            logs_dir="../logs/replay", comparison_name="model_vs_ground_truth"
        )
        print(f"💾 已保存第 {self.current_index + 1}/{len(self.data)} 帧replay数据: {saved_path}")
        
        # 前进到下一帧
        self.current_index += 1
        print(f"➡️  前进到第 {self.current_index + 1}/{len(self.data)} 帧")
    
    def auto_save_current_frame(self):
        """自动保存当前帧数据（已弃用，现在等action回传后再保存）"""
        pass  # 不再使用此方法
    
    def start_server(self):
        """Start the Flask server."""
        # 设置路由
        self.setup_routes()
        
        print(f"🚀 Starting parquet replay server on port {self.port}")
        print(f"📊 Serving {len(self.data) if self.data is not None else 0} frames")
        print(f"🔗 Available endpoints:")
        print(f"   GET /image_raw - Main camera image")
        print(f"   GET /image_wrist - Wrist camera image") 
        print(f"   GET /robot_state_ee - Robot state")
        print(f"   GET /instruction - Task instruction")
        print(f"   POST /action - Receive inference actions (triggers save & advance)")
        print(f"💾 保存机制: 收到 /action 请求时保存对比数据并前进到下一帧")
        print(f"🛑 自动退出: 处理完所有 {len(self.data) if self.data is not None else 0} 帧后自动关闭服务器")
        
        self.app.run(host='0.0.0.0', port=self.port, debug=False)

def parse_args():
    parser = argparse.ArgumentParser(description='Parquet replay server (只支持Both模式)')
    parser.add_argument('--parquet_file', type=str, default='episode_000001.parquet',
                        help='Path to the parquet file')
    parser.add_argument('--port', type=int, default=19001,
                        help='Server port to listen on')
    parser.add_argument('--zero_state', action='store_true',
                        help='Whether to use zero state for robot state (default: False)')
    
    # 模式选择参数（保留向后兼容性）
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--rt-only', action='store_true',
                           help='(已弃用) RT操作请使用franka_test_droid.py')
    mode_group.add_argument('--replay-only', action='store_true', default=True,
                           help='Replay模式：只使用parquet数据 (默认)')
    mode_group.add_argument('--both', action='store_true',
                           help='Both模式：使用parquet数据进行重放对比')
    
    # 向后兼容的参数
    parser.add_argument('--enable-rt-images', action='store_true',
                        help='(已弃用) 使用 --both 替代')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 确定运行模式 - 只支持replay-only和both模式
    if args.rt_only:
        print("❌ RT-Only模式已不再支持，请使用franka_test_droid.py进行RT操作")
        print("💡 建议: 使用 --both 模式进行对比测试")
        return
    elif args.replay_only:
        mode = "replay-only"
    elif args.both:
        mode = "both"
    elif getattr(args, 'enable_rt_images', False):  # 向后兼容
        mode = "both"
        print("⚠️  --enable-rt-images 已弃用，使用 --both 模式")
    else:
        mode = "replay-only"  # 默认模式
    
    # 检查parquet文件是否存在
    if not Path(args.parquet_file).exists():
        print(f"❌ 错误: 文件 {args.parquet_file} 不存在")
        return
    
    print(f"🚀 启动 {mode.upper()} 模式的Parquet重放服务器")
    print(f"📁 数据文件: {args.parquet_file}")
    print(f"🌐 端口: {args.port}")
    
    # 启动HTTP服务
    replay_server = ParquetReplayServer(
        parquet_file=args.parquet_file,
        port=args.port,
        zero_state=args.zero_state,
        mode=mode
    )
    
    # 加载数据
    replay_server.load_data()
    
    try:
        replay_server.start_server()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")

if __name__ == "__main__":
    main()
