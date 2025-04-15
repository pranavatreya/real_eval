import argparse
import datetime
import io
import os
import sys
import time
import select

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip
import matplotlib.pyplot as plt

try:
    from droid.robot_env import RobotEnv
except ModuleNotFoundError:
    from r2d2.robot_env import RobotEnv

import faulthandler
faulthandler.enable()

from eval_config import EvalConfig, load_config
from websocket_client_policy import WebsocketClientPolicy
import image_tools

def flush_stdin_buffer():
    while sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        # read and discard
        sys.stdin.readline()

def extract_observation(obs_dict, setting):
    """Extract left/right/wrist images if available, plus robot state."""
    def is_camera_image(camera_name: str, observation_key: str) -> bool:
        """Returns True if the observation is for the specified camera."""
        if camera_name not in setting.cameras:
            return False
        camera_id: str = str(setting.cameras[camera_name])
        return (camera_id in observation_key) and ("left" in observation_key)

    image_observations = obs_dict["image"]
    left_image, right_image, wrist_image = None, None, None

    for key in image_observations.keys():
        if is_camera_image("left", key):
            left_image = image_observations[key]
        elif is_camera_image("right", key):
            right_image = image_observations[key]
        elif is_camera_image("wrist", key):
            wrist_image = image_observations[key]

    def process_image(img):
        if img is None:
            return None
        # Drop alpha channel if present
        img = img[..., :3]
        # Convert from BGR to RGB if needed
        img = np.concatenate([img[..., 2:], img[..., 1:2], img[..., :1]], axis=-1)
        # Resize to 288x512
        img = np.array(Image.fromarray(img).resize((512, 288), resample=Image.LANCZOS))
        return img

    left_image = process_image(left_image)
    right_image = process_image(right_image)
    wrist_image = process_image(wrist_image)

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


def run_evaluation(setting, evaluator_name, institution):
    """
    Main evaluation logic
    """
    env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
    desired_dt = 1/15
    base_image = setting.third_person_camera  # e.g. "left_image" or "right_image"

    # Display the current camera feeds so user can confirm
    zeroth_obs = extract_observation(env.get_observation(), setting)
    left_img, right_img = zeroth_obs["left_image"], zeroth_obs["right_image"]
    if left_img is None:
        left_img = np.zeros((288, 512, 3), dtype=np.uint8)
    if right_img is None:
        right_img = np.zeros((288, 512, 3), dtype=np.uint8)
    left_plus_right = np.concatenate([left_img, right_img], axis=1)
    plt.imshow(left_plus_right)
    plt.title("Current left/right camera views.\nCheck if these are pointed correctly at the robot/scene")
    plt.show(block=False)

    user_response = input("Are these third-person cameras pointed correctly (y/n)? If no, we will exit, and please take the time to adjust the cameras.  ").strip().lower()
    if user_response != "y":
        print("Exiting to allow for camera readjustments.")
        plt.close()
        exit()

    plt.close()
    should_switch = input(
        f"The default third-person camera is {base_image}. Type 'switch' to switch vantage, or press ENTER to keep: "
    ).strip().lower()
    if should_switch == "switch":
        if base_image == "left_image":
            base_image = "right_image"
        else:
            base_image = "left_image"

    logging_server_ip = setting.logging_server_ip
    get_url = f"http://{logging_server_ip}/get_policies_to_compare"
    # We pass evaluator_name and institution as 'eval_location'
    params = {
        "eval_location": institution,
        "evaluator_name": evaluator_name,
        "robot_name": "DROID"
    }
    r = requests.get(get_url, params=params)
    if not r.ok:
        print("Failed to get policies from central server!")
        print(r.status_code, r.text)
        exit()

    session_info = r.json()
    session_id = session_info["session_id"]
    policies = session_info["policies"]

    print("\nSession started successfully.")
    print(f"Session ID: {session_id}")
    print("Beginning evaluation of policies A, B, C, and D...")

    # We'll store the user’s single language command that we want each policy to follow
    lang_command = input("Enter the language command to be given to all policies: ")

    # Save the current robot observation so that we can display it later
    reset_obs = extract_observation(env.get_observation(), setting)
    left_img, right_img = reset_obs["left_image"], reset_obs["right_image"]
    if left_img is None:
        left_img = np.zeros((288, 512, 3), dtype=np.uint8)
    if right_img is None:
        right_img = np.zeros((288, 512, 3), dtype=np.uint8)
    ref_reset_state = np.concatenate([left_img, right_img], axis=1)

    preference_ab = None
    comparative_feedback = None
    max_timesteps = 400

    for i, pol_dict in enumerate(policies):
        label = pol_dict["label"]  # A/B/C/D
        policy_name = pol_dict["policy_name"]
        ip = pol_dict["ip"]
        port = pol_dict["port"]

        print(f"\n=== Evaluating policy {label} ===")
        print("IMPORTANT: You can type 't' (then ENTER) at any time to terminate this episode early.\n")

        policy_client = WebsocketClientPolicy(ip, port)

        frames_left = []
        frames_right = []
        frames_wrist = []
        episode_data = []

        open_loop_horizon = 8
        actions_from_chunk_completed = 0
        pred_action_chunk = None

        bar = tqdm(range(max_timesteps))
        for t_step in bar:
            start_time = time.time()

            # Obtain observation
            obs = env.get_observation()
            curr_obs = extract_observation(obs, setting)

            # Collect frames
            if curr_obs["left_image"] is not None:
                frames_left.append(curr_obs["left_image"])
            if curr_obs["right_image"] is not None:
                frames_right.append(curr_obs["right_image"])
            if curr_obs["wrist_image"] is not None:
                frames_wrist.append(curr_obs["wrist_image"])

            # If we need a new chunk of actions from the policy server
            if (pred_action_chunk is None) or (actions_from_chunk_completed >= open_loop_horizon):
                actions_from_chunk_completed = 0
                request_data = {
                    "observation/exterior_image_1_left": image_tools.resize_with_pad(
                        curr_obs.get(base_image, None), 224, 224
                    ) if curr_obs.get(base_image, None) is not None else None,
                    "observation/wrist_image_left": image_tools.resize_with_pad(
                        curr_obs["wrist_image"], 224, 224
                    ) if curr_obs["wrist_image"] is not None else None,
                    "observation/joint_position": curr_obs["joint_position"],
                    "observation/gripper_position": curr_obs["gripper_position"],
                    "prompt": lang_command,
                }
                result = policy_client.infer(request_data)
                pred_action_chunk = result["actions"][:8]

            action = np.array(pred_action_chunk[actions_from_chunk_completed], dtype=np.float32)
            actions_from_chunk_completed += 1

            # Binarize gripper
            if action[-1] > 0.5:
                action[-1] = 1.0
            else:
                action[-1] = 0.0

            # Clip
            action = np.clip(action, -1, 1)

            # Step environment
            env.step(action)

            step_data = {
                "cartesian_position": curr_obs["cartesian_position"].tolist(),
                "joint_position": curr_obs["joint_position"].tolist(),
                "gripper_position": curr_obs["gripper_position"].tolist(),
                "action": action.tolist(),
            }
            episode_data.append(step_data)

            # Check if user typed 't' to terminate
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                line = sys.stdin.readline().strip().lower()
                if line == 't':
                    print("User typed 't'. Terminating the rollout early.")
                    break

            elapsed_dt = time.time() - start_time
            time.sleep(max(0, desired_dt - elapsed_dt))

        flush_stdin_buffer()

        # Query user for partial success
        partial_succ = 0.0
        while True:
            user_inp = input(f"Please rate partial success of policy {label} in [0..100]: ")
            try:
                val = float(user_inp)
                if 0 <= val <= 100:
                    partial_succ = val / 100.0
                    break
            except:
                pass
            print("Invalid input. Must be a number from 0..100.")

        # Automatically infer binary success from partial success
        bin_succ = 1 if partial_succ == 1.0 else 0

        # After policy B is done, ask for preference (A/B/tie) and long-form feedback
        if label == "B":
            flush_stdin_buffer()
            while True:
                pref = input("Which policy did you prefer, A, B, or 'tie'? ").strip().lower()
                if pref in ["a", "b", "tie"]:
                    preference_ab = pref.upper()  # store as "A", "B", or "TIE"
                    break
                print("Please enter 'A', 'B', or 'tie' exactly.")
            flush_stdin_buffer()
            comparative_feedback = input(
                "Now please provide long-form textual feedback comparing policy A vs. policy B:\n"
            )
            flush_stdin_buffer()

        print()
        # Construct the left/right/wrist videos if frames exist
        timestamp_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        def make_video_file(frames_list, suffix):
            if not frames_list:
                return None
            out_path = f"/tmp/temp_{suffix}.mp4"
            clip = ImageSequenceClip(frames_list, fps=10)
            clip.write_videofile(out_path, codec="libx264", audio=False)
            with open(out_path, "rb") as f:
                vid_bytes = f.read()
            os.remove(out_path)
            return (f"video_{suffix}", (f"{suffix}.mp4", vid_bytes, "video/mp4"))

        files = {}
        left_video_file = make_video_file(frames_left, "left")
        if left_video_file:
            files[left_video_file[0]] = left_video_file[1]
        right_video_file = make_video_file(frames_right, "right")
        if right_video_file:
            files[right_video_file[0]] = right_video_file[1]
        wrist_video_file = make_video_file(frames_wrist, "wrist")
        if wrist_video_file:
            files[wrist_video_file[0]] = wrist_video_file[1]

        # Save the episode data as .npz
        npz_out_path = "/tmp/temp_episode_data.npz"
        np.savez_compressed(npz_out_path, data=episode_data)
        with open(npz_out_path, "rb") as f:
            npz_bytes = f.read()
        os.remove(npz_out_path)
        files["npz_file"] = ("episode_data.npz", npz_bytes, "application/octet-stream")

        data = {
            "session_id": session_id,
            "policy_name": policy_name,
            "command": lang_command,
            "binary_success": str(bin_succ),
            "partial_success": str(partial_succ),
            "duration": str(t_step),  # how many steps we did
            "policy_ip": str(ip),
            "policy_port": str(port),
            "third_person_camera_type": base_image,
            "third_person_camera_id": str(setting.cameras.get(base_image, "")),
            "policy_letter": label,
            "timestamp": timestamp_str,
        }

        print(f"Uploading episode data for policy {label}...")
        upload_url = f"http://{logging_server_ip}/upload_eval_data"
        resp = requests.post(upload_url, files=files, data=data)
        if not resp.ok:
            print("Error uploading data to logging server:", resp.text)
            break
        else:
            print("Data upload succeeded.")

        # Ensure env.reset() after each policy.
        env.reset()
        if i < len(policies) - 1:
            fig, ax = plt.subplots()
            ax.set_title("Reminder: reset scene to the original starting condition.\nFor reference, this is what your starting state looked like:")
            ax.imshow(ref_reset_state)
            plt.show(block=False)

            user_resp = input(
                "\nPlease reset the robot and environment to match the original starting conditions.\n"
                "Press ENTER when done: "
            )
            plt.close(fig)

    valid_yesno = input("Did everything go well? Mark this session as valid? (y/n): ").strip().lower()
    final_notes = ""
    if valid_yesno == "y":
        final_notes += "VALID_SESSION:\n"

    if preference_ab:
        final_notes += f"PREFERENCE={preference_ab}\n"
    if comparative_feedback:
        final_notes += f"LONGFORM_FEEDBACK={comparative_feedback}\n"

    term_url = f"http://{setting.logging_server_ip}/terminate_session"
    term_data = {
        "session_id": session_id,
        "evaluation_notes": final_notes
    }
    requests.post(term_url, data=term_data)

    print(f"\nEvaluation session {session_id} terminated.")
    print("Completed the multi-policy eval. Thank you!\n\n(If the script doesn't auto-terminate, please hit control-C.)")
    exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load evaluation config YAML file.")
    parser.add_argument("config_path", type=str, help="Path to the evaluation config YAML file")
    args = parser.parse_args()

    setting: EvalConfig = load_config(args.config_path)

    # Check if evaluator_name/institution exist in the YAML. If so, give the user a choice to keep or change.
    default_evaluator_name = getattr(setting, "evaluator_name", None)
    default_institution = getattr(setting, "institution", None)

    if default_evaluator_name and default_institution:
        print(f"We see from your YAML that evaluator_name={default_evaluator_name}, institution={default_institution}.")
        resp = input("Press ENTER to keep them, or type 'change' to override: ").strip().lower()
        if resp == 'change':
            evaluator_name = input("Hi Evaluator! Please enter your full name: ").strip()
            institution = input("At which institution are you performing this eval (e.g., Berkeley, UPenn)? ").strip()
        else:
            evaluator_name = default_evaluator_name
            institution = default_institution
    else:
        evaluator_name = input("Hi Evaluator! Please enter your full name: ").strip()
        institution = input("At which institution are you performing this eval (e.g., Berkeley, UPenn)? ").strip()

    run_evaluation(setting, evaluator_name, institution)
