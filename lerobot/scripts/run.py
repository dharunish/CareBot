from lerobot.common.policies.act.modeling_act_try import ACTPolicy
import time
from lerobot.scripts.control_robot import busy_wait
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus 
from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera
import torch 
import os
import platform

record_time_s = 30
fps = 30

states = []
actions = []

leader_port = "/dev/tty.usbmodem58760434871"
follower_port = "/dev/tty.usbmodem585A0077011"

leader_arm = FeetechMotorsBus(
    port=leader_port,
    motors={
        # name: (index, model)
        "shoulder_pan": (1, "sts3215"),
        "shoulder_lift": (2, "sts3215"),
        "elbow_flex": (3, "sts3215"),
        "wrist_flex": (4, "sts3215"),
        "wrist_roll": (5, "sts3215"),
        "gripper": (6, "sts3215"),
    },
)

follower_arm = FeetechMotorsBus(
    port=follower_port,
    motors={
        # name: (index, model)
        "shoulder_pan": (1, "sts3215"),
        "shoulder_lift": (2, "sts3215"),
        "elbow_flex": (3, "sts3215"),
        "wrist_flex": (4, "sts3215"),
        "wrist_roll": (5, "sts3215"),
        "gripper": (6, "sts3215"),
    },
)

robot = ManipulatorRobot(
    robot_type="moss",
    leader_arms={"main": leader_arm},
    follower_arms={"main": follower_arm},
    calibration_dir=".cache/calibration/moss",
    cameras={
        "laptop": OpenCVCamera(0, fps=30, width=640, height=480),
        "phone": OpenCVCamera(1, fps=30, width=640, height=480),
    },
)


def say(text, blocking=False):
    # Check if mac, linux, or windows.
    if platform.system() == "Darwin":
        cmd = f'say "{text}"'
    elif platform.system() == "Linux":
        cmd = f'spd-say "{text}"'
    elif platform.system() == "Windows":
        cmd = (
            'PowerShell -Command "Add-Type -AssemblyName System.Speech; '
            f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')\""
        )

    if not blocking and platform.system() in ["Darwin", "Linux"]:
        # TODO(rcadene): Make it work for Windows
        # Use the ampersand to run command in the background
        cmd += " &"

    os.system(cmd)


inference_time_s = 120
fps = 30
device = "mps"  # TODO: On Mac, use "mps" or "cpu"

ckpt_path = "/Users/dharunish/Desktop/Robotics/trained_models/feeding_task_all_8ah_32model/pretrained_model"
# ckpt_path = "/Users/helper2424/Documents/lerobot/outputs/koch_move_obj_static_cameras/model.safetensors"
policy = ACTPolicy.from_pretrained(ckpt_path, force_download=True)
policy.to(device)

say("I am going to collect objects")
for _ in range(inference_time_s * fps):
    start_time = time.perf_counter()

    # Read the follower state and access the frames from the cameras
    observation = robot.capture_observation()

    # Convert to pytorch format: channel first and float32 in [0,1]
    # with batch dimension
    for name in observation:
        if "image" in name:
            observation[name] = observation[name].type(torch.float32) / 255
            observation[name] = observation[name].permute(2, 0, 1).contiguous()
        observation[name] = observation[name].unsqueeze(0)
        observation[name] = observation[name].to(device)

    # Compute the next action with the policy
    # based on the current observation
    action = policy.select_action(observation)
    # Remove batch dimension
    action = action.squeeze(0)
    # Move to cpu, if not already the case
    action = action.to("cpu")
    # Order the robot to move
    robot.send_action(action)

    dt_s = time.perf_counter() - start_time
    busy_wait(1 / fps - dt_s)

say("I have finished collecting objects")
