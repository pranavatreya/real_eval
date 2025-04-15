import argparse
import datetime
import io
import os
import time

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip
import matplotlib.pyplot as plt

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


def extract_observation(obs_dict, setting):
    """Extract left/right/wrist images if available, plus robot state."""
    def is_camera_image(camera_name: str, observation_key: str) -> bool:
        """Returns True if the observation is for the specified camera. False, otherwise."""
        if camera_name not in setting.cameras:
            # For some setups, certain cameras may not be present (e.g., the right camera at Stanford)
            return False

        camera_id: str = str(setting.cameras[camera_name])
        return camera_id in observation_key and "left" in observation_key

    image_observations = obs_dict["image"]
    left_image, right_image, wrist_image = None, None, None

    # The user’s config file may define which cameras correspond to 'left','right','wrist'
    # e.g. setting.cameras["left"] might be "24259877", etc.
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


def main():
    """
    1) Ask the user for their name and institution (for logging purposes)
    2) Show the current feeds from the left/right cameras, and ask user if this looks ok (script will exit out if not)
    3) Ask the user if they want to use the current default third-person camera (options are left/right), or change it
    4) Query the central server for a session
    5) For each policy provided, do a rollout:
       - Collect frames from left, right, wrist cameras.
       - Collect robot states & actions each timestep -> store in .npz
       - At the end, ask user for success, partial success, feedback, etc.
       - Upload everything to the central server via /upload_eval_data.
    5) Call /terminate_session to mark session complete.
    6) Notify the evaluator that if they want to run another A/B eval, they should simply re-run this script
    """

    # Initialize the environment
    env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
    desired_dt = 1 / 15

    evaluator_name = input("Hi Evaluator! Please enter your full name: ")
    evaluator_name = evaluator_name.strip()

    institution = input("And at which institution are you performing this eval (e.g., Berkeley, UPenn): ")
    institution = institution.strip()

    # Get the current third-person camera observations and ask evaluator if they look good
    zeroth_obs = extract_observation(env.get_observation(), setting)
    left_img, right_img = zeroth_obs["left_image"], zeroth_obs["right_image"]
    assert not (left_img is None and right_img is None)
    if left_img is None:
        left_img = np.zeros((288, 512, 3), dtype=np.uint8)
    if right_img is None:
        right_img = np.zeros((288, 512, 3), dtype=np.uint8)
    left_plus_right = np.concatenate([left_img, right_img], axis=1)
    plt.imshow(left_plus_right)
    plt.title("Current left/right camera views. See terminal for instructions.")
    plt.show(block=False)

    user_response = input("Are the third person cameras correctly pointed at the relevant parts of the scene (y/n)? If not, this script will terminate, and please adjust the camera angles. ")
    user_response = user_response.strip().lower()
    if user_response != "y":
        print("Exiting...")
        exit()

    # The config file is loaded in __main__
    base_image: str = setting.third_person_camera  # e.g. "left", "right"
    logging_server_ip: str = setting.logging_server_ip

    should_switch = input(f"The default third-person camera is specified as the {base_image} camera. Press ENTER to keep this default, or to eval with the other third-person camera, type in 'switch': ")
    should_switch = should_switch.strip().lower()
    if should_switch == "switch":
        base_image = "left_image" if base_image == "right_image" else "right_image"
        print(f"Switch the third-person camera viewpoint to the {base_image}")

    plt.close()

    # 1) Get policies from server
    #   Possibly we read query args like "?eval_type=single-policy&eval_location=Stanford" if needed.
    #   For brevity, just do a GET with no args:
    get_url = f"http://{logging_server_ip}/get_policies_to_compare"
    r = requests.get(get_url)
    if not r.ok:
        print("Failed to get policies to compare from central server!")
        print(r.status_code, r.text)
        return

    session_info = r.json()
    session_id = session_info["session_id"]
    evaluation_type = session_info.get("evaluation_type", "A/B")
    print(f"Session ID: {session_id} / Type: {evaluation_type}")

    # We will be given an arbitrary number of policies, the first two of which we evaluate A/B style
    # The remaining policies will also be evaluated A/B style, with but they are used to get g.t., results
    policies_to_evaluate = []
    curr_letter = "A"
    while True:
        policy_data = session_info.get("policy" + curr_letter, None)
        if policy_data is None:
            break
        policies_to_evaluate.append((curr_letter, policy_data))
        print(f"Policy {curr_letter} => {policy_data}")
        curr_letter = chr(ord(curr_letter) + 1)

    lang_command = input("\nEnter language command: ")        
    comparative_feedback = None

    for policy_label, policy_data in policies_to_evaluate:
        policy_name = policy_data["policy_name"]  # unique name in DB
        ip = policy_data["ip"]
        port = policy_data["port"]

        print(f"\n=== Evaluating policy {policy_label} => {policy_name} ({ip}:{port}) ===")

        # Create the policy client
        policy_client = WebsocketClientPolicy(ip, port)
        #policy_client = WebsocketClientPolicy("128.32.175.199", port)

        # Prepare for rollout
        max_timesteps = 600
        open_loop_horizon = 8
        actions_from_chunk_completed = 0
        pred_action_chunk = None

        # We'll store frames from all 3 cameras if they exist
        frames_left = []
        frames_right = []
        frames_wrist = []

        # We also store a list of dicts for robot state + the action we took at each step
        episode_data = []

        bar = tqdm(range(max_timesteps))
        for t_step in bar:
            start_time = time.time()
            try:
                obs = env.get_observation()
                curr_obs = extract_observation(obs, setting)
                # Collect frames
                if curr_obs["left_image"] is not None:
                    frames_left.append(curr_obs["left_image"])
                if curr_obs["right_image"] is not None:
                    frames_right.append(curr_obs["right_image"])
                if curr_obs["wrist_image"] is not None:
                    frames_wrist.append(curr_obs["wrist_image"])

                # If time to request a new chunk of actions
                if (pred_action_chunk is None) or (actions_from_chunk_completed >= open_loop_horizon):
                    actions_from_chunk_completed = 0
                    request_data = {
                        "observation/exterior_image_1_left": image_tools.resize_with_pad(
                            curr_obs[base_image], 224, 224
                        ),
                        "observation/wrist_image_left": image_tools.resize_with_pad(
                            curr_obs["wrist_image"], 224, 224
                        ),
                        "observation/joint_position": curr_obs["joint_position"],
                        "observation/gripper_position": curr_obs["gripper_position"],
                        "prompt": lang_command,
                    }
                    # Infer
                    result = policy_client.infer(request_data)
                    pred_action_chunk = result["actions"]
                    # We only use the first 8 actions from the chunk
                    pred_action_chunk = pred_action_chunk[:8]

                # Select current action
                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1

                # Convert to a writable numpy array
                action = np.array(action, dtype=np.float32)

                # Binarize gripper
                if action[-1] > 0.5:
                    action[-1] = 1.0
                else:
                    action[-1] = 0.0
                # Clip
                action = np.clip(action, -1, 1)

                # Step environment
                env.step(action)

                # Record proprio + action for this step
                step_data = {
                    "cartesian_position": curr_obs["cartesian_position"].tolist(),
                    "joint_position": curr_obs["joint_position"].tolist(),
                    "gripper_position": curr_obs["gripper_position"].tolist(),
                    "action": action.tolist(),
                }
                episode_data.append(step_data)

                # Sleep to try to maintain ~15Hz
                elapsed_dt = time.time() - start_time
                time.sleep(max(0, desired_dt - elapsed_dt))
            except KeyboardInterrupt:
                print("User interrupted the rollout.")
                break

        # Query user for binary success
        binary_success = 0
        yesno = input(f"\nDid policy {policy_label} fully succeed? (y/n): ")
        if yesno.strip().lower() in ["y", "yes"]:
            binary_success = 1

        # Query user for partial success (0..100)
        partial_success = 0.0
        while True:
            user_inp = input(f"Please rate partial success of policy {policy_label} in [0..100]: ")
            try:
                val = float(user_inp)
                if 0 <= val <= 100:
                    partial_success = val / 100.0
                    break
            except:
                pass
            print("Invalid input. Must be a number 0..100.")

        # Query user for textual feedback
        #feedback = input("Any textual feedback on how the policy performed? ")
        # Rather than per-episode feedback, we decided to ask for comparative feedback
        if policy_label == "B":
            comparative_feedback = input("Policies A and B have been rolled out. Now, please provide long-form textual feedback on how policy A did compared to policy B. Feedback to include here includes qualitative behavior of the two policies, anything the policies did particularly well or bad (e.g., instruction following, dexterity, generalization capabilites): ")
            print("Thanks for the feedback!")

        # Construct the left/right/wrist videos if frames exist
        timestamp_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        def make_video_file(frames_list, suffix):
            """Helper: turn frames_list -> .mp4 in memory, return (file_field_name, file_tupl)."""
            if not frames_list:
                return None  # No frames => no video
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
            # left_video_file is a tuple of (field_name, (filename, bytes, content-type))
            files[left_video_file[0]] = left_video_file[1]

        right_video_file = make_video_file(frames_right, "right")
        if right_video_file:
            files[right_video_file[0]] = right_video_file[1]

        wrist_video_file = make_video_file(frames_wrist, "wrist")
        if wrist_video_file:
            files[wrist_video_file[0]] = wrist_video_file[1]

        # Save the episode data as .npz
        npz_out_path = "/tmp/temp_episode_data.npz"
        np.savez_compressed(npz_out_path, data=episode_data)  # store as an array of dicts
        with open(npz_out_path, "rb") as f:
            npz_bytes = f.read()
        os.remove(npz_out_path)

        files["npz_file"] = ("episode_data.npz", npz_bytes, "application/octet-stream")

        # Prepare form data
        data = {
            "session_id": session_id,
            "policy_name": policy_name,
            "command": lang_command,
            "binary_success": str(binary_success),
            "partial_success": str(partial_success),
            "duration": str(t_step),  # how many steps
            "policy_ip": str(ip),
            "policy_port": str(port),
            "third_person_camera_type": base_image,
            "third_person_camera_id": str(setting.cameras.get(base_image, "")),
            "feedback": None, # we're omitting per-episode feedback
            "timestamp": timestamp_str,
        }

        print(f"\nUploading episode data for policy {policy_label} => {policy_name} ...")
        upload_url = f"http://{logging_server_ip}/upload_eval_data"
        resp = requests.post(upload_url, files=files, data=data)
        if not resp.ok:
            print("Error uploading data to logging server:", resp.text)
        else:
            print("Data upload succeeded.")

        env.reset()
        input("\nResetting the arm to evaluate the next policy. Press also reset the scene, and hit enter when ready: ")

    # 4) Terminate the session
    term_url = f"http://{logging_server_ip}/terminate_session"
    requests.post(term_url, data={"session_id": session_id, "evaluation_notes": comparative_feedback})
    print(f"\nEvaluation session {session_id} terminated.\n")

    print("Completed an A/B Eval. To run another eval, please re-run this script!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load evaluation config YAML file.")
    parser.add_argument("config_path", type=str, help="Path to the evaluation config YAML file")
    args = parser.parse_args()

    # Load the config, which presumably sets .logging_server_ip, .third_person_camera, .cameras, etc.
    setting: EvalConfig = load_config(args.config_path)
    main()
