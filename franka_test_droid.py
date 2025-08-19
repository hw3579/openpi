import requests
import numpy as np
import cv2
import argparse
from openpi_client import websocket_client_policy, image_tools
import threading

MAX_STEPS = 120
step = 0

def get_image_from_server(endpoint, server_url):
    url = f"{server_url}/{endpoint}"
    resp = requests.get(url)
    if resp.status_code != 200:
        try:
            error_msg = resp.json().get('error', f'HTTP {resp.status_code}')
        except:
            error_msg = f'HTTP {resp.status_code}: {resp.text[:100]}'
        raise RuntimeError(f"Failed to get image: {error_msg}")
    headers = resp.headers
    height = int(headers["X-Height"])
    width = int(headers["X-Width"])
    channels = int(headers["X-Channels"])
    dtype = np.dtype(headers["X-Dtype"])
    image = np.frombuffer(resp.content, dtype=dtype).reshape((height, width, channels))
    return image

def get_instruction(server_url):
    url = f"{server_url}/instruction"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()["instruction"]

def get_robot_state(server_url):
    url = f"{server_url}/robot_state_ee"
    resp = requests.get(url)
    if resp.status_code != 200:
        raise RuntimeError("Failed to get robot state")
    return resp.json()

def send_action_to_server(action, server_url):
    """将action发送到服务器"""
    try:
        url = f"{server_url}/action"
        payload = {"actions":action.tolist()}
        resp = requests.post(url, json=payload)
        if resp.status_code == 200:
            print(f"Action sent successfully: {payload}")
            return resp.json()
        else:
            print(f"Failed to send action: {resp.status_code}, {resp.text}")
            return None
    except Exception as e:
        print(f"Error sending action: {e}")
        return None

def get_realtime_data_from_rt_server(rt_server_url="http://192.168.191.210:9001"):
    """从RT服务器（210）获取实时图像和robot state用于对比显示"""
    try:
        rt_img = get_image_from_server("image_raw", rt_server_url)
        rt_wrist_img = get_image_from_server("image_wrist", rt_server_url)
        rt_robot_state = get_robot_state(rt_server_url)
        return rt_img, rt_wrist_img, rt_robot_state
    except Exception as e:
        print(f"Warning: Failed to get realtime data from RT server (210): {e}")
        return None, None, None

def parse_args():
    parser = argparse.ArgumentParser(description='Franka robot control client')
    parser.add_argument('--replay-server', type=str, default='http://192.168.191.143:9001',
                        help='Replay server URL (143 - for replay mode)')
    parser.add_argument('--rt-server', type=str, default='http://192.168.191.210:9001',
                        help='RT server URL (210 - for realtime mode)')
    parser.add_argument('--policy-host', type=str, default='localhost',
                        help='Policy websocket host (default: localhost)')
    parser.add_argument('--policy-port', type=int, default=8000,
                        help='Policy websocket port (default: 8000)')
    parser.add_argument('--mode', type=str, choices=['rt', 'both', 'sim'], default='rt',
                        help='Operation mode: rt (realtime only), both (replay data input + realtime comparison), sim (simulation only, no RT server needed)')
    return parser.parse_args()

def main(replay_server, rt_server, policy_host, policy_port, mode='rt'):
    print(f"Starting robot control client in {mode} mode")
    print(f"Replay server (143): {replay_server}")
    print(f"RT server (210): {rt_server}")
    
    while True:
        if mode == 'both':
            # Both模式：从replay服务器获取数据用于喂模型 (143)
            print(f"Fetching replay data from server for model input: {replay_server}...")
            replay_img = get_image_from_server("image_raw", replay_server)
            replay_wrist_img = get_image_from_server("image_wrist", replay_server)
            replay_instruction = get_instruction(replay_server)
            replay_robot_state = get_robot_state(replay_server)
            
            print(f"Replay robot state: {replay_robot_state}")
            
            # Both模式：同时获取实时数据用于对比记录 (210)
            print(f"Fetching realtime data for comparison: {rt_server}...")
            rt_img, rt_wrist_img, rt_robot_state = get_realtime_data_from_rt_server(rt_server)
            
            if rt_img is not None:
                print(f"RT robot state: {rt_robot_state}")
            else:
                print("Warning: Failed to get RT data for comparison")
            
        elif mode == 'rt':
            # RT模式：从RT服务器获取实时数据 (210)
            print(f"Fetching realtime data from RT server: {rt_server}...")
            rt_img, rt_wrist_img, rt_robot_state = get_realtime_data_from_rt_server(rt_server)
            
            if rt_img is not None:
                print(f"RT robot state: {rt_robot_state}")
            else:
                print("Failed to get RT data, skipping this iteration")
                continue
        elif mode == 'sim':
            # Sim模式：只从replay服务器获取数据进行仿真评估 (143)
            print(f"Fetching simulation data from replay server: {replay_server}...")
            replay_img = get_image_from_server("image_raw", replay_server)
            replay_wrist_img = get_image_from_server("image_wrist", replay_server)
            replay_instruction = get_instruction(replay_server)
            replay_robot_state = get_robot_state(replay_server)
            
            print(f"Simulation robot state: {replay_robot_state}")
            
            # Sim模式下创建零值RT数据（不连接真实机器人）
            rt_img = np.zeros_like(replay_img)  # 创建与replay图像相同尺寸的零图像
            rt_wrist_img = np.zeros_like(replay_wrist_img)  # 创建与replay手腕图像相同尺寸的零图像
            rt_robot_state = {
                "obs_state": np.zeros(7, dtype=np.float32).tolist()  # 7维零状态
            }
            print("Sim mode: Using zero RT data (no real robot connection)")
        
        
        # 根据模式选择喂给模型的数据源
        if mode == 'rt':
            # RT模式：使用实时数据
            img, wrist_img = rt_img, rt_wrist_img
            instruction = get_instruction(rt_server)
            robot_state = rt_robot_state
        elif mode in ['both', 'sim']:
            # Both模式和Sim模式：使用replay数据喂模型（更准确的ground truth）
            img, wrist_img = replay_img, replay_wrist_img
            instruction = replay_instruction
            robot_state = replay_robot_state

        obs_state = np.array(robot_state["obs_state"], dtype=np.float32)
        ############ assertion
        assert img.shape == (256, 256, 3), f"Expected image shape (256, 256, 3), but got {img.shape}"
        assert wrist_img.shape == (256, 256, 3), f"Expected wrist image shape (256, 256, 3), but got {wrist_img.shape}"
        assert obs_state.shape == (7,), f"Expected observation state shape (7,), but got {obs_state.shape}"
        assert instruction is not None, "Instruction cannot be None"
        ############ assertion



        # 处理图像数据
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # wrist_img = cv2.cvtColor(wrist_img, cv2.COLOR_BGR2RGB)


        #######################################################
        '''
        src/openpi/training/config.py line292 定义了libero_policy的输入格式
        class LeRobotLiberoDataConfig(DataConfigFactory):

        src/openpi/policies/libero_policy.py line 50 定义了LiberoInputs类
        class LiberoInputs(transforms.DataTransformFn):
        '''
        ###################################################
        from matplotlib import pyplot as plt
        from PIL import Image
        
        # img = img[100:720,200:1000]
        
        # 调整图像尺寸
        # img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
        # wrist_img = cv2.resize(wrist_img, (256, 256), interpolation=cv2.INTER_LINEAR)

        Image.fromarray(img).save("1.png")
        Image.fromarray(wrist_img).save("2.png")

        example = {
            "observation/image": img,
            "observation/wrist_image": wrist_img,
            "observation/state": obs_state,
            "prompt": instruction
        }

        example_rt = {
            "observation/image": rt_img,
            "observation/wrist_image": rt_wrist_img,
            "observation/state": np.array(rt_robot_state["obs_state"], dtype=np.float32),
            "prompt": instruction
        }

        # 分工保存：
        # - RT模式：droid保存RT数据到 ../logs/rt/
        # - Both模式：droid保存对比数据，replay_parquet.py保存replay数据到 ../logs/replay/

        
        from tools import convert_example_to_policy_input, save_log_realtime
        # policy_input = convert_example_to_policy_input(example)

        print("Running inference...")
        print(f"Instruction: {instruction}")
        #### 
        #注意这里调用的policy和example不一样！！！那个是pretrained 这个是web的
        ###
        
        client = websocket_client_policy.WebsocketClientPolicy(host=policy_host, port=policy_port)
        result = client.infer(example)

        result["actions"] = result["actions"][0]
        # result["actions"] = result["actions"].flatten()

        # 保存日志数据：分工明确
        if mode == 'rt':
            # RT模式：droid负责保存RT数据到rt目录
            save_log_realtime(example_rt, result)
        elif mode == 'both':
            # Both模式：droid保存包含对比数据的记录到rt目录
            save_log_realtime(example_rt, result)
        elif mode == 'sim':
            # Sim模式：保存模拟数据到sim目录（包含零值RT数据）
            save_log_realtime(example_rt, result, logs_dir="../logs/sim")
        # Both模式：replay数据保存由replay_parquet.py服务器负责（通过接收action触发）

        print("Action:", result["actions"])

        # 根据模式发送动作到相应服务器
        if mode == 'rt':
            # RT模式：发送到RT服务器 (210)
            rt_response = send_action_to_server(result["actions"], rt_server)
            if rt_response: 
                print("RT server response:", rt_response)
        elif mode == 'both':
            # Both模式：发送到replay服务器 (143) 触发保存
            replay_response = send_action_to_server(result["actions"], replay_server)
            if replay_response:
                print("Replay server response:", replay_response)
            
            # Both模式：也可以选择发送到RT服务器执行动作
            rt_response = send_action_to_server(result["actions"], rt_server)
            # if rt_response: 
            #     print("RT server response:", rt_response)
        elif mode == 'sim':
            # Sim模式：只发送到replay服务器进行仿真评估（不发送到RT服务器）
            replay_response = send_action_to_server(result["actions"], replay_server)
            if replay_response:
                print("Simulation server response:", replay_response)
            print("Sim mode: No real robot action execution")

        global step
        step += 1
        if step >= MAX_STEPS:
            print("Reached maximum steps, exiting...")
            break
            


if __name__ == "__main__":
    args = parse_args()
    main(args.replay_server, args.rt_server, args.policy_host, args.policy_port, args.mode)

    # 测试命令示例：
    # RT模式: python franka_test_droid.py --mode rt
    # Both模式 (对比分析): python franka_test_droid.py --mode both
    # Sim模式 (仿真评估，无需机器人): python franka_test_droid.py --mode sim
    # 自定义服务器: python franka_test_droid.py --mode both --rt-server http://192.168.191.210:9001 --replay-server http://192.168.191.143:9001 