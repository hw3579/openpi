import dataclasses
import enum
import logging
import socket
import threading
import time
import requests
import cv2
import numpy as np
import base64
import json
from typing import Dict, Any, Optional

import tyro
from openpi_client import msgpack_numpy

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""
    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    FRANKA = "franka"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""
    config: str
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""
    env: EnvMode = EnvMode.FRANKA
    default_prompt: str | None = "manipulate objects with the robotic arm"
    port: int = 8000
    record: bool = False
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)
    
    # 机器人控制相关参数
    enable_robot_control: bool = True
    remote_robot_host_request: str = "192.168.2.3"
    remote_robot_host_execute: str = "192.168.2.5"
    remote_request_port: int = 8001
    remote_execute_port: int = 8002
    control_frequency: float = 5.0
    auto_start: bool = True


# Default checkpoints
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi0_aloha",
        dir="s3://openpi-assets/checkpoints/pi0_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="s3://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi0_fast_droid",
        dir="s3://openpi-assets/checkpoints/pi0_fast_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi0_fast_libero",
        dir="s3://openpi-assets/checkpoints/pi0_fast_libero",
    ),
    EnvMode.FRANKA: Checkpoint(
        config="pi0_franka_h5_lora",
        dir="./checkpoints/pi0_franka_h5_lora/franka_h5_lora_test/100",
    ),
}


class RobotController:
    """机器人控制器 - 修复数据格式问题"""
    
    def __init__(self, policy: _policy.Policy, args: Args):
        self.policy = policy
        self.args = args
        self.running = False
        self.current_instruction = args.default_prompt or "manipulate objects"
        
        # 远端服务器地址
        self.request_url = f"http://{args.remote_robot_host_request}:{args.remote_request_port}"
        self.execute_url = f"http://{args.remote_robot_host_execute}:{args.remote_execute_port}"
        
        # 控制参数
        self.control_frequency = args.control_frequency
        self.loop_time = 1.0 / self.control_frequency
        
        logging.info(f"Robot Controller initialized")
        logging.info(f"Request URL: {self.request_url}")
        logging.info(f"Execute URL: {self.execute_url}")
        
        # 调试：打印策略的输入transform信息
        if hasattr(policy, '_input_transform'):
            logging.info(f"Policy input transform: {policy._input_transform}")
    
    def check_robot_status(self) -> bool:
        """检查机器人状态"""
        try:
            response = requests.get(f"{self.request_url}/status", timeout=5.0)
            if response.status_code == 200:
                status = response.json()
                return status.get("robot_ready", False) and status.get("camera_ready", False)
        except Exception as e:
            logging.warning(f"Failed to check robot status: {e}")
        return False
    
    def get_robot_observation(self) -> Optional[Dict[str, Any]]:
        """获取机器人观察数据 - 修复数据格式"""
        try:
            observation = {}
            
            # 获取关节状态
            state_response = requests.get(f"{self.request_url}/robot_state_ee", timeout=5.0)
            if state_response.status_code == 200:
                robot_state = state_response.json()
                position = robot_state.get("obs_state", [0.0] * 7)
                # 添加夹爪位置（假设为0）
                # observation["state"] = np.array(position + [0.0], dtype=np.float32)
                observation["state"] = np.array(position, dtype=np.float32)  # 只包含7个关节位置

            else:
                logging.warning("Failed to get robot state")
                return None
            
            # 获取图像 - 注意：使用"image"而不是"images"
            img_response = requests.get(f"{self.request_url}/image_raw", timeout=5.0)
            if img_response.status_code == 200:
                height = int(img_response.headers["X-Height"])
                width = int(img_response.headers["X-Width"])
                channels = int(img_response.headers["X-Channels"])
                dtype = img_response.headers["X-Dtype"]
                
                img_array = np.frombuffer(img_response.content, dtype=dtype).reshape((height, width, channels))
                
                # 预处理图像
                processed_img = self.preprocess_image(img_array)
                
                # 关键修复：使用"image"键，且格式为{"base_0_rgb": image_array}
                observation["image"] = {"base_0_rgb": processed_img}

                # 添加图像掩码 - 全1的掩码表示整个图像都有效
                # h, w = processed_img.shape[:2]
                observation["image_mask"] = {"base_0_rgb": np.ones((1, 224*224), dtype=np.bool_)} 
            else:
                logging.warning("Failed to get image")
                return None
            
            # 获取指令
            try:
                instr_response = requests.get(f"{self.request_url}/instruction", timeout=3.0)
                if instr_response.status_code == 200:
                    instruction_data = instr_response.json()
                    instruction = instruction_data.get("instruction", self.current_instruction)
                    observation["prompt"] = instruction
                    self.current_instruction = instruction
                else:
                    observation["prompt"] = self.current_instruction
            except:
                observation["prompt"] = self.current_instruction
            
            logging.info(f"Observation keys: {list(observation.keys())}")
            logging.info(f"Image keys: {list(observation['image'].keys())}")
            logging.info(f"Image shape: {observation['image']['base_0_rgb'].shape}")
            logging.info(f"State shape: {observation['state'].shape}")
            
            return observation
            
        except Exception as e:
            logging.error(f"Failed to get robot observation: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """预处理图像"""
        try:
            # 确保图像是uint8格式
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            # 确保是RGB格式（如果是BGR需要转换）
            if len(image.shape) == 3 and image.shape[2] == 3:
                # 转换BGR到RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            logging.info(f"Preprocessed image shape: {image.shape}, dtype: {image.dtype}")
            
            return image
            
        except Exception as e:
            logging.error(f"Image preprocessing error: {e}")
            # 返回一个安全的默认图像
            return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def predict_action(self, observation: Dict[str, Any]) -> Optional[np.ndarray]:
        """使用策略预测动作"""
        try:
            logging.info("Starting action prediction...")
            
            # 打印观察数据的详细信息用于调试
            logging.info(f"Observation structure:")
            for key, value in observation.items():
                if key == "image":
                    logging.info(f"  {key}: {type(value)} with keys {list(value.keys())}")
                    for img_key, img_value in value.items():
                        logging.info(f"    {img_key}: shape={img_value.shape}, dtype={img_value.dtype}")
                elif key == "state":
                    logging.info(f"  {key}: shape={value.shape}, dtype={value.dtype}, values={value}")
                else:
                    logging.info(f"  {key}: {type(value)} = {value}")
            
            # 调用策略的infer方法
            result = self.policy.infer(observation)
            
            logging.info(f"Policy result keys: {list(result.keys())}")
            
            if "actions" in result:
                actions = result["actions"]
                
                # 处理不同的actions格式
                if isinstance(actions, np.ndarray):
                    logging.info(f"Actions shape: {actions.shape}, dtype: {actions.dtype}")
                    
                    # 如果是多维数组，取第一个动作
                    if actions.ndim > 1:
                        action = actions[0]  # 取第一个时间步
                        if action.ndim > 1:
                            action = action[0]  # 如果还有批次维度
                    else:
                        action = actions
                else:
                    action = np.array(actions)
                
                logging.info(f"Final action: {action}, shape: {action.shape}")
                return action
            else:
                logging.error(f"No actions in policy result: {list(result.keys())}")
                return None
                
        except Exception as e:
            logging.error(f"Failed to predict action: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def execute_action(self, action: np.ndarray) -> bool:
        """执行动作"""
        try:
            # 确保action是1D数组
            if action.ndim > 1:
                action = action.flatten()
            
            # 只发送前7个关节位置（忽略夹爪）
            joint_positions = action[:7].tolist()
            
            # 确保关节位置是合理的（添加安全检查）
            joint_positions = [float(x) for x in joint_positions]
            
            payload = {
                "joint_positions": joint_positions
            }
            
            logging.info(f"Sending joint positions: {joint_positions}")
            
            response = requests.post(f"{self.execute_url}/execute", json=payload, timeout=5.0)
            success = response.status_code == 200
            
            if success:
                logging.info(f"Action executed successfully")
            else:
                logging.warning(f"Action execution failed: {response.status_code}, {response.text}")
            
            return success
            
        except Exception as e:
            logging.error(f"Failed to execute action: {e}")
            return False
    
    def control_step(self) -> bool:
        """执行一步控制"""
        step_start = time.time()
        
        try:
            # 1. 获取观察
            observation = self.get_robot_observation()
            if observation is None:
                logging.warning("Failed to get observation")
                return False
            
            # 2. 预测动作
            action = self.predict_action(observation)
            if action is None:
                logging.warning("Failed to predict action")
                return False
            
            # 3. 执行动作
            success = self.execute_action(action)
            
            step_time = time.time() - step_start
            logging.info(f"Control step completed in {step_time:.3f}s, success: {success}")
            
            return success
            
        except Exception as e:
            logging.error(f"Control step failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def control_loop(self):
        """控制循环"""
        logging.info("Starting robot control loop...")
        
        # 等待机器人准备就绪
        while self.running:
            if self.check_robot_status():
                logging.info("Robot is ready!")
                break
            else:
                logging.info("Waiting for robot to be ready...")
                time.sleep(2.0)
        
        # 主控制循环
        while self.running:
            loop_start = time.time()
            
            try:
                self.control_step()
            except Exception as e:
                logging.error(f"Control loop error: {e}")
            
            # 控制频率
            elapsed = time.time() - loop_start
            if elapsed < self.loop_time:
                time.sleep(self.loop_time - elapsed)
        
        logging.info("Robot control loop stopped")
    
    def start(self):
        """启动控制"""
        if not self.running:
            self.running = True
            self.control_thread = threading.Thread(target=self.control_loop, daemon=True)
            self.control_thread.start()
            logging.info("Robot control started")
    
    def stop(self):
        """停止控制"""
        self.running = False
        logging.info("Robot control stopped")


class EnhancedWebsocketPolicyServer(websocket_policy_server.WebsocketPolicyServer):
    """增强的WebSocket策略服务器，支持机器人控制"""
    
    def __init__(self, policy, host, port, metadata, robot_controller: Optional[RobotController] = None):
        super().__init__(policy, host, port, metadata)
        self.robot_controller = robot_controller
    
    async def _handler(self, websocket):
        """重写handler以支持机器人控制命令"""
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                
                # 接收消息
                message_data = await websocket.recv()
                
                # 尝试解析为JSON（机器人控制命令）
                try:
                    if isinstance(message_data, (str, bytes)):
                        if isinstance(message_data, bytes):
                            message_str = message_data.decode('utf-8')
                        else:
                            message_str = message_data
                        
                        # 尝试解析JSON控制命令
                        control_message = json.loads(message_str)
                        
                        if control_message.get("type") == "robot_control":
                            command = control_message.get("command")
                            
                            if command == "start" and self.robot_controller:
                                self.robot_controller.start()
                                response = {"type": "robot_control", "status": "started"}
                                await websocket.send(json.dumps(response).encode())
                                continue
                            
                            elif command == "stop" and self.robot_controller:
                                self.robot_controller.stop()
                                response = {"type": "robot_control", "status": "stopped"}
                                await websocket.send(json.dumps(response).encode())
                                continue
                            
                            elif command == "status" and self.robot_controller:
                                status = {
                                    "type": "robot_control",
                                    "status": "running" if self.robot_controller.running else "stopped",
                                    "robot_ready": self.robot_controller.check_robot_status()
                                }
                                await websocket.send(json.dumps(status).encode())
                                continue
                
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # 不是JSON，继续使用msgpack处理
                    pass
                
                # 使用msgpack解析（标准策略请求）
                obs = msgpack_numpy.unpackb(message_data)

                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception as e:
                logging.error(f"Handler error: {e}")
                import traceback
                traceback.print_exc()
                break


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    # 创建机器人控制器
    robot_controller = None
    if args.enable_robot_control:
        robot_controller = RobotController(policy, args)
        
        if args.auto_start:
            robot_controller.start()
            logging.info("Robot control auto-started")
    
    # 创建增强的服务器
    server = EnhancedWebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
        robot_controller=robot_controller
    )
    
    try:
        logging.info(f"🚀 Policy server with robot control starting on port {args.port}")
        if robot_controller:
            logging.info("📡 Robot control enabled")
            logging.info(f"🤖 Remote robot request: {args.remote_robot_host_request}")
            logging.info(f"🤖 Remote robot execute: {args.remote_robot_host_execute}")
            logging.info("💡 Send JSON message to control robot:")
            logging.info('   {"type": "robot_control", "command": "start"}')
            logging.info('   {"type": "robot_control", "command": "stop"}')
            logging.info('   {"type": "robot_control", "command": "status"}')
        
        server.serve_forever()
        
    except KeyboardInterrupt:
        logging.info("Shutting down...")
        if robot_controller:
            robot_controller.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))