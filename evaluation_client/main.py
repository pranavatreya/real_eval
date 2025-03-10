import argparse
import datetime
import io
import os
import requests
import time

from moviepy.editor import ImageSequenceClip
from PIL import Image
from tqdm import tqdm
import numpy as np

from eval_config import EvalConfig, load_config
from websocket_client_policy import WebsocketClientPolicy
import image_tools

try:
    from droid.robot_env import RobotEnv
except ModuleNotFoundError:
    # r2d2 is the old name for the package
    from r2d2.robot_env import RobotEnv

import faulthandler
faulthandler.enable()

# ========== HELPER FUNCTIONS ==========

def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> Image.Image:
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image  # No need to resize if already correct size.

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, (height - resized_height) // 2)
    pad_width = max(0, (width - resized_width) // 2)
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return zero_image

def resize_with_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """Replicates tf.image.resize_with_pad for multiple images using PIL."""
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape
    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([
        _resize_with_pad_pil(Image.fromarray(im), height, width, method=method)
        for im in images
    ])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])

def extract_observation(obs_dict):
    image_observations = obs_dict["image"]
    left_image, right_image, wrist_image = None, None, None
    for key in image_observations.keys():
        # According to
        # https://github.com/droid-dataset/droid/blob/main/droid/camera_utils/camera_readers/zed_camera.py#L142
        if setting.cameras["left"] in key and "left" in key:
            left_image = image_observations[key]
        elif setting.cameras["right"] in key and "left" in key:
            right_image = image_observations[key]
        elif setting.cameras["wrist"] in key and "left" in key:
            wrist_image = image_observations[key]
    # Drop alpha dimension
    left_image = left_image[..., :3]
    right_image = right_image[..., :3]
    wrist_image = wrist_image[..., :3]
    # Convert to RGB
    left_image = np.concatenate([left_image[...,2:], left_image[...,1:2], left_image[..., :1]], axis=-1)
    right_image = np.concatenate([right_image[...,2:], right_image[...,1:2], right_image[..., :1]], axis=-1)
    wrist_image = np.concatenate([wrist_image[...,2:], wrist_image[...,1:2], wrist_image[..., :1]], axis=-1)
    # Resize
    left_image = np.array(Image.fromarray(left_image).resize((512, 288), resample=Image.LANCZOS))
    right_image = np.array(Image.fromarray(right_image).resize((512, 288), resample=Image.LANCZOS))
    wrist_image = np.array(Image.fromarray(wrist_image).resize((512, 288), resample=Image.LANCZOS))

    robot_state = obs_dict["robot_state"]
    cartesian_position = np.array(robot_state["cartesian_position"])
    joint_position = np.array(robot_state["joint_positions"])
    gripper_position = np.array([robot_state["gripper_position"]])

    return {
        "left_image": left_image,
        "right_image": right_image,
        "wrist_image": wrist_image,
        "cartesian_position": cartesian_position,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
    }

# ========== MAIN EVALUATION CODE ==========

def main():
    """
    1) Query the central orchestrating server for a session and two policy IPs.
    2) Evaluate them in an A/B fashion for each user prompt.
    3) Upload the video/metadata to the central server at the end of each rollout.
    4) Continue until user is done with all of his/her evals.
    5) Terminate the session on the central server associated with this eval.
    """
    base_image: str = setting.third_person_camera
    logging_server_ip: str = setting.logging_server_ip

    r = requests.get(f"http://{logging_server_ip}/get_policies_to_compare")
    if not r.ok:
        print("Failed to get policies to compare from logging server!")
        print(r.status_code, r.text)
        return

    session_info = r.json()
    session_id = session_info["session_id"]
    policyA_ip = session_info["policyA"]  # e.g. "10.103.116.247:8000"
    policyB_ip = session_info["policyB"]

    # Initialize the environment
    env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
    desired_dt = 1 / 15

    while True:
        lang_command = input("Enter language command (or 'quit'): ")
        if lang_command.strip().lower() in ["quit", "exit"]:
            break

        # We do two evaluations, one for A, one for B
        for policy_label, policy_ip in [("A", policyA_ip), ("B", policyB_ip)]:
            # Initialize policy client
            ip, port = policy_ip.split(":")
            policy_client = WebsocketClientPolicy(ip, port)

            print(f"Starting eval of policy {policy_label}...")
            time.sleep(2)

            # Setup for the rollout
            max_timesteps = 600
            open_loop_horizon = 8
            actions_from_chunk_completed = 0
            pred_action_chunk = None

            # We'll store frames in memory and upload a video at the end
            frames = []

            print(f"\n=== Evaluating policy {policy_label} ({policy_ip}) ===")
            bar = tqdm(range(max_timesteps))
            for t_step in bar:
                start_time = time.time()
                try:
                    # Get the current observation
                    curr_obs = extract_observation(env.get_observation())
                    # Store the right_image (or whichever we prefer) for the video
                    frames.append(curr_obs["right_image"])

                    # If it's time to request a new action chunk
                    if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= open_loop_horizon:
                        actions_from_chunk_completed = 0
                        request_data = {
                            f"observation/exterior_image_1_left": image_tools.resize_with_pad(curr_obs[base_image], 224, 224),
                            #"observation/exterior_image_1_left_mask": np.array(True),
                            "observation/wrist_image_left": image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
                            #"observation/wrist_image_left_mask": np.array(True),
                            "observation/joint_position": curr_obs["joint_position"],
                            "observation/gripper_position": curr_obs["gripper_position"],
                            #"raw_text": lang_command,
                            "prompt": lang_command,
                        }
                        #pred_action_chunk = _make_request(policy_ip, "/infer", request_data)
                        pred_action_chunk = policy_client.infer(request_data)["actions"]
                        pred_action_chunk = pred_action_chunk[:8]
                        assert pred_action_chunk.shape == (8, 8)

                    # Select current action to execute
                    action = pred_action_chunk[actions_from_chunk_completed]
                    actions_from_chunk_completed += 1
                    action = np.array(action.tolist()) # to remove read-only restriction

                    # binarize gripper
                    if action[-1] > 0.5:
                        action[-1] = 1.0
                    else:
                        action[-1] = 0.0
                    # Clip
                    action = np.clip(action, -1, 1)

                    env.step(action)

                    # Sleep to achieve desired_dt
                    elapsed_dt = time.time() - start_time
                    time.sleep(max(0, desired_dt - elapsed_dt))
                except KeyboardInterrupt:
                    break

            # Now that the rollout is done, let’s ask for success measure:
            success_val = None
            while success_val is None:
                user_inp = input(
                    f"Did policy {policy_label} succeed? (y=100%, n=0%, or numeric 0..100) "
                )
                if user_inp.lower() in ["y", "yes"]:
                    success_val = 1.0
                elif user_inp.lower() in ["n", "no"]:
                    success_val = 0.0
                else:
                    try:
                        numeric = float(user_inp)
                        if numeric < 0 or numeric > 100:
                            raise ValueError
                        success_val = numeric / 100.0
                    except ValueError:
                        print("Please enter 'y', 'n', or a number in [0..100].")

            # Turn the frames into an mp4 in memory (no local saving)
            timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            video_buf = io.BytesIO()
            clip = ImageSequenceClip(list(frames), fps=10)
            # Write to BytesIO buffer
            clip.write_videofile("/tmp/temp_eval.mp4", codec="libx264", audio=False)
            # Now read that file into memory:
            with open("/tmp/temp_eval.mp4", "rb") as f:
                video_bytes = f.read()

            # Clean up the temp file to avoid leaving large disk usage
            os.remove("/tmp/temp_eval.mp4")

            # Send the data to the logging server
            upload_url = f"http://{logging_server_ip}/upload_eval_data"
            files = {
                "video": ("rollout.mp4", video_bytes, "video/mp4")
            }
            data = {
                "session_id": session_id,
                "policy_name": f"policy_{policy_label}",
                "command": lang_command,
                "success": str(success_val),
                "duration": str(t_step),
                "timestamp": timestamp,
            }
            resp = requests.post(upload_url, files=files, data=data)
            if not resp.ok:
                print("Error uploading data to logging server:", resp.text)

            if policy_label == "A":
                # Reset the env and wait for user
                env.reset()
                input("Resetting the arm to begin evaluation of policy B. Please also reset the environment. Press enter when ready: ")

        # After we finish both policy A and policy B, ask user if we continue
        cont = input("Do you want to evaluate another task? (y/n) ")
        if cont.strip().lower() != "y":
            break
        env.reset()

    # When fully done, tell the central server to terminate the session
    term_url = f"http://{logging_server_ip}/terminate_session"
    requests.post(term_url, data={"session_id": session_id})
    print("Evaluation session terminated. Exiting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load evaluation config YAML file.")
    parser.add_argument("config_path", type=str, help="Path to the evaluation config YAML file")
    args = parser.parse_args()

    setting: EvalConfig = load_config(args.config_path)
    main()
