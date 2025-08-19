import requests
import numpy as np
import cv2
import argparse
from openpi_client import websocket_client_policy, image_tools

MAX_STEPS = 100
step = 0

def get_image_from_server(endpoint, server_url):
    url = f"{server_url}/{endpoint}"
    resp = requests.get(url)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to get image: {resp.json().get('error')}")
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
            # print(f"Action sent successfully: {payload}")
            return resp.json()
        else:
            print(f"Failed to send action: {resp.status_code}, {resp.text}")
            return None
    except Exception as e:
        print(f"Error sending action: {e}")
        return None

def parse_args():
    parser = argparse.ArgumentParser(description='Franka robot control client')
    parser.add_argument('--input-server', type=str, default='http://192.168.2.3:8001',
                        help='Input server URL (default: http://localhost:8001)')
    parser.add_argument('--output-server', type=str, default='http://192.168.2.3:8002',
                        help='Output server URL (default: http://localhost:8002)')
    parser.add_argument('--policy-host', type=str, default='localhost',
                        help='Policy websocket host (default: localhost)')
    parser.add_argument('--policy-port', type=int, default=8000,
                        help='Policy websocket port (default: 8000)')
    return parser.parse_args()

def main(input_server, output_server, policy_host, policy_port):

        while True:
            print(f"Fetching data from robot server: {input_server}...")
            img = get_image_from_server("image_raw", input_server)
            wrist_img = get_image_from_server("image_wrist", input_server)
            instruction = get_instruction(input_server)
            robot_state = get_robot_state(input_server)

            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # wrist_img = cv2.cvtColor(wrist_img, cv2.COLOR_BGR2RGB)
            obs_state = np.array(robot_state["obs_state"], dtype=np.float32)
            print(f"Robot state: {robot_state}")

            # obs_state[5] = obs_state[5] + np.pi/4
            obs_state = np.hstack((obs_state[:5], obs_state[6:]))
            # example = {
            #     "observation/image": image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224)),
            #     "observation/wrist_image": image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, 224, 224)),
            #     "observation/state": obs_state,
            #     "prompt": instruction
            # }


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

            img = img[100:720,200:1000]


            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
            wrist_img = cv2.resize(wrist_img, (256, 256), interpolation=cv2.INTER_LINEAR)

            Image.fromarray(img).save("1.png")
            Image.fromarray(wrist_img).save("2.png")

            example = {
                "observation/image": img,
                "observation/wrist_image": wrist_img,
                "observation/state": obs_state,
                "prompt": instruction
            }

            from tools import convert_example_to_policy_input
            policy_input = convert_example_to_policy_input(example)

            print("Running inference...")
            print(f"Instruction: {instruction}")
            #### 
            #注意这里调用的policy和example不一样！！！那个是pretrained 这个是web的

            ###
            client = websocket_client_policy.WebsocketClientPolicy(host=policy_host, port=policy_port)

            result = client.infer(example)

            result["actions"] = result["actions"][0]

            print("Action:", result["actions"])

            # 如需发送动作到输出服务器，取消注释下面两行
            response = send_action_to_server(result["actions"], output_server)
            # if response: print("Server response:", response)

            
            global step
            step += 1
            if step >= MAX_STEPS:
                print("Reached maximum steps, exiting...")
                break


if __name__ == "__main__":
    args = parse_args()
    main(args.input_server, args.output_server, args.policy_host, args.policy_port)


    # 测试命令 