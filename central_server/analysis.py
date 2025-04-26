from dataclasses import asdict, dataclass, field
import argparse
import json
import os
import pdb
import textwrap
import yaml

from flask import Flask, abort, render_template, send_from_directory
from tqdm import tqdm
import cv2
import fsspec

from database.schema import PolicyModel, SessionModel
from database.connection import initialize_database_connection
from llm.openai_client import OpenAIClient
from logger import logger

"""
Analyzes the performance of robot manipulation policies based on evaluation sessions.
It downloads camera recordings, processes them, and generates reports using LLMs.
Finally, it serves the analysis results through a web interface.

Usage:
    python analysis.py

    python analysis.py --skip-refresh
    --skip-refresh: Skips recomputing the analysis and use the cached version of the analysis JSON on disk.
"""


BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
OUTPUT_DIR = os.path.abspath(os.path.join(ROOT_DIR, "output"))
ANALYSIS_JSON_PATH = os.path.join(OUTPUT_DIR, "policy_analysis.json")

app = Flask(
    __name__,
    template_folder="frontend/templates",
    static_folder="frontend/static",
    static_url_path="/static",
)


@dataclass
class Session:
    id: str
    """Unique ID of the session."""

    location: str
    """Location of the evaluation."""

    evaluator: str
    """Name of the evaluator."""

    prompt: str
    """The text prompt the evaluator gave in this session."""


@dataclass
class Cameras:
    left_gcs_path: str | None = None
    left_local_path: str | None = None
    left_first_frame_local_path: str | None = None

    right_gcs_path: str | None = None
    right_local_path: str | None = None
    right_first_frame_local_path: str | None = None

    wrist_gcs_path: str | None = None
    wrist_local_path: str | None = None
    wrist_first_frame_local_path: str | None = None

    @staticmethod
    def extract_first_frame(video_path: str) -> str | None:
        """
        Extracts and saves the first frame of a video. Returns the image path.
        Skips extraction if the image already exists.
        """
        if not os.path.exists(video_path):
            logger.warning(f"Video file not found: {video_path}")
            return None

        base, ext = os.path.splitext(video_path)
        image_path = f"{base}_first_frame.jpg"

        if os.path.exists(image_path):
            return image_path

        cap = cv2.VideoCapture(video_path)
        success, frame = cap.read()
        cap.release()

        assert success, f"Failed to read the first frame from {video_path}"
        cv2.imwrite(image_path, frame)
        return image_path

    def download(self, gcs_bucket: str) -> None:
        """
        Download camera videos from GCS and extract first frame image.
        """
        if self.left_gcs_path:
            self.left_local_path = os.path.join(output_path, self.left_gcs_path)
            download_from_gcs(
                f"gs://{gcs_bucket}/{self.left_gcs_path}", self.left_local_path
            )
            self.left_first_frame_local_path = self.extract_first_frame(
                self.left_local_path
            )

        if self.right_gcs_path:
            self.right_local_path = os.path.join(output_path, self.right_gcs_path)
            download_from_gcs(
                f"gs://{gcs_bucket}/{self.right_gcs_path}", self.right_local_path
            )
            self.right_first_frame_local_path = self.extract_first_frame(
                self.right_local_path
            )

        if self.wrist_gcs_path:
            self.wrist_local_path = os.path.join(output_path, self.wrist_gcs_path)
            download_from_gcs(
                f"gs://{gcs_bucket}/{self.wrist_gcs_path}", self.wrist_local_path
            )
            self.wrist_first_frame_local_path = self.extract_first_frame(
                self.wrist_local_path
            )

    def sample_video(self) -> str:
        """
        Get a video from the camera prioritizing the third-person cameras. Return the local path.
        """
        sample_video_path: str | None = None
        if self.left_local_path:
            sample_video_path = self.left_local_path
        elif self.right_local_path:
            sample_video_path = self.right_local_path
        elif self.wrist_local_path:
            sample_video_path = self.wrist_local_path

        assert sample_video_path, "No video available for this episode."
        return sample_video_path


@dataclass
class HeadToHead:
    perspective_policy: str
    """In the perspective of this policy."""

    was_policy_a: bool
    """Whether the policy was policy A in the head-to-head evaluation."""

    won: bool
    """Whether the policy won the head-to-head evaluation."""

    tied: bool
    """Whether the head-to-head evaluation was a tie."""

    ab_notes: str
    """Notes from the head-to-head evaluation."""

    @property
    def full_notes(self) -> str:
        """
        Generate the full notes of the head-to-head evaluation in perspective of `perspective_policy` policy`.
        """
        return textwrap.dedent(
            f"""
        Policy A or B: {self.perspective_policy} was Policy {"A" if self.was_policy_a else "B"}
        Result: {self.perspective_policy} {"tied" if self.tied else "won" if self.won else "lost"}
        Evaluation notes: {self.ab_notes}
        """
        )


@dataclass
class Episode:
    session: Session
    """The session this episode belongs to."""

    partial_success_score: float
    """The partial success score of the episode."""

    duration: float
    """The duration of the episode in seconds."""

    cameras: Cameras
    """Camera data for the episode."""

    head_to_head: HeadToHead | None = None
    """Head-to-head evaluation data for the episode (if exists)."""

    annotations: str | None = None
    """Annotations from a VLM using the first frames of the cameras and the task at hand."""

    def annotate(self) -> None:
        """
        Annotate the episode using first frames from available cameras.
        """
        if not self.head_to_head:
            # Don't annotate the beginning of the episode if there is no head-to-head evaluation
            return

        image_paths = [
            self.cameras.left_first_frame_local_path,
            self.cameras.right_first_frame_local_path,
            self.cameras.wrist_first_frame_local_path,
        ]
        image_paths = [p for p in image_paths if p and os.path.exists(p)]

        if not image_paths:
            logger.warning(
                f"No valid first-frame images found for session {self.session.id}"
            )
            return

        prompt = (
            f"You're analyzing the first-frame images of videos recorded during a robot manipulation evaluation, where the robot is commanded to carry out a task.\n\n"
            f"Task description: {self.session.prompt.strip()}\n\n"
            f"One of the images is a top-down view of the scene taken from the wrist camera of the robot arm, and the other one or two images are third-person views of the scene taken from the left and right cameras.\n\n"
            f"Please describe the following aspects based on the provided images:\n\n"
            f"Camera angle: Describe the provided first frames and whether they provide a clear view of the objects and environment necessary for executing the task.\n"
            f"Lighting: Evaluate whether the lighting is sufficient. Note if any shadows, glares, or dim areas make the task harder to observe or complete.\n"
            f"Clarity of task: Based on the task description, is what the robot is expected to do clear? Mention any ambiguity. Note spelling/grammar mistakes/lowercase vs capitalized letters.\n"
            f"Scene: Describe the overall scene setup. Are there distractors, unnecessary clutter, or many objects that may interfere with completing the task? Describe the individual objects (e.g., orientation, hidden, etc.) and if they cause difficulty in carrying out the task.\n"
            f"Difficulty: Estimate how difficult the task appears, considering the setup, task clarity, overall scene, object descriptions and placements, and visibility. Explain why it seems easy or hard (e.g., the handle on the drawer is tiny; the task for this scene requires the ability to execute very precise/dextrous manipulation, etc.).\n\n"
            f"Output your response in the following format only:\n\n"
            f"Camera angle: <your analysis>\n"
            f"Lighting: <your analysis>\n"
            f"Clarity of task: <your analysis>\n"
            f"Scene: <your analysis>\n"
            f"Difficulty: <your analysis>\n\n"
            f"Explain your answers thoroughly while strictly following the format above. Do not incorporate markdown or any other formatting."
        )

        try:
            # We use GPT-4.5 since it is a frontier model for image understanding
            # https://crfm.stanford.edu/helm/vhelm/v2.1.2/#/leaderboard/visual_perception
            response, cached = openai_client.run_image_inference(
                model="gpt-4.5-preview-2025-02-27",
                image_paths=image_paths,
                text=prompt,
                temperature=0,
                max_tokens=2048,
            )
            self.annotations = response["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"Annotation failed for session {self.session.id}: {e}")

    @property
    def report(self) -> str:
        """
        Generate a report that summarizes the episode with the head-to-head evaluation notes.
        """
        assert (
            self.head_to_head
        ), "Episode has no head-to-head evaluation, so nothing to report."
        return textwrap.dedent(
            f"""Session ID: {self.session.id}
Task: {self.session.prompt}

Scene Setup and Task Analysis
{self.annotations}

Head-to-Head Comparison
{self.head_to_head.full_notes.strip()}"""
        )


@dataclass
class Policy:
    name: str
    """Unique name of the policy."""

    episodes: list[Episode] = field(default_factory=list)
    """All the episodes across all sessions for this policy."""

    def add_episode(self, episode: Episode):
        """
        Add an episode to the policy.
        """
        self.episodes.append(episode)

    def get_all_head_to_head_episodes(self) -> list[Episode]:
        """
        Get all episodes that have head-to-head evaluations.
        """
        return [episode for episode in self.episodes if episode.head_to_head]


@dataclass
class PolicyPerformanceAnalysis:
    policy_name: str
    """Name of the policy."""

    number_of_head_to_head_evaluations: int
    """Number of head-to-head evaluations for this policy."""

    full_report: str
    """Full report of the head-to-head evaluations for this policy."""

    summary: str
    """Summary of the full report."""

    episode_reports: list[str]
    """List of episode reports for this policy."""

    session_id_to_video_path: dict[str, str]
    """Mapping of session IDs to sample video path for this policy."""

    session_id_to_prompt: dict[str, str]
    """Mapping of session IDs to prompt for this policy."""


def get_all_valid_policies() -> dict[str, Policy]:
    """
    Returns all valid policies as a dict where the key is the policy name and the value is the policy
    """
    db = SessionLocal()

    # Gather all policies except PI0 and PI0_FAST
    policies = (
        db.query(PolicyModel)
        .filter(PolicyModel.unique_policy_name.notin_(["PI0", "PI0_FAST"]))
        .all()
    )
    logger.info(f"Found {len(policies)} valid policies.")

    return {
        policy.unique_policy_name: Policy(name=policy.unique_policy_name)
        for policy in policies
    }


def download_from_gcs(gcs_path: str, destination_path: str) -> None:
    """
    Downloads a file from GCS to local storage.
    """
    if os.path.exists(destination_path):
        return

    logger.info(f"Downloading {gcs_path} from GCS.")
    fs = fsspec.filesystem("gcs")
    if not fs.exists(gcs_path):
        raise FileNotFoundError(f"{gcs_path} does not exist in GCS.")

    # The slash is needed to download the contents of the folder to `destination_path`
    fs.get(gcs_path, destination_path)
    assert os.path.exists(
        destination_path
    ), f"Failed to download {gcs_path} to {destination_path}."


def populate_policy_episodes(policies: dict[str, Policy], gcs_bucket: str) -> None:
    """
    Populate the policy episodes with the data from the sessions.
    """
    db = SessionLocal()

    # Gather all valid evaluation sessions.
    # Valid eval sessions have VALID_SESSION: in the beginning of `evaluation_notes`
    # and check that episodes is not empty
    sessions = (
        db.query(SessionModel)
        .filter(
            SessionModel.evaluation_notes.like("VALID_SESSION:%"),
            SessionModel.episodes.any(),
        )
        .all()
    )
    logger.info(f"Found {len(sessions)} valid evaluation sessions.")

    for session_row in tqdm(sessions):
        assert session_row.episodes, "Session has no episodes."
        prompt: str = session_row.episodes[0].command
        session_id: str = session_row.session_uuid
        session = Session(
            id=session_id,
            location=session_row.evaluation_location,
            evaluator=session_row.evaluator_name,
            prompt=prompt,
        )

        # Get the name of the polices where we did the A/B evaluation
        policy_a_name: str = session_row.policyA_name
        policy_b_name: str = session_row.policyB_name

        # Example of evaluation_notes:
        # 'VALID_SESSION:\nPREFERENCE=B\nLONGFORM_FEEDBACK=policy A did not do anything
        # -- just froze. policy B actually picked up the red box at its third attempt.\n'
        raw_evaluation_notes: str = session_row.evaluation_notes
        ab_notes: str = raw_evaluation_notes.split("LONGFORM_FEEDBACK=")[-1].strip()

        # Process each episode and add it to the corresponding policy
        # If we match head-to-head policies, also add `HeadToHead` information
        for episode_row in session_row.episodes:
            policy_name: str = episode_row.policy_name
            head_to_head: HeadToHead | None
            if policy_name == policy_a_name:
                # If the policy is A, check if it won
                head_to_head = HeadToHead(
                    perspective_policy=policy_name,
                    was_policy_a=True,
                    won="PREFERENCE=A" in raw_evaluation_notes,
                    tied="PREFERENCE=TIE" in raw_evaluation_notes,
                    ab_notes=ab_notes,
                )
            elif policy_name == policy_b_name:
                # If the policy is B, check if it won
                head_to_head = HeadToHead(
                    perspective_policy=policy_name,
                    was_policy_a=False,
                    won="PREFERENCE=B" in raw_evaluation_notes,
                    tied="PREFERENCE=TIE" in raw_evaluation_notes,
                    ab_notes=ab_notes,
                )
            else:
                # If the policy is neither A nor B, set `head_to_head` to None
                head_to_head = None

            # Download the camera videos from GCS
            cameras = Cameras(
                left_gcs_path=episode_row.gcs_left_cam_path,
                right_gcs_path=episode_row.gcs_right_cam_path,
                wrist_gcs_path=episode_row.gcs_wrist_cam_path,
            )
            cameras.download(gcs_bucket)
            episode = Episode(
                session=session,
                partial_success_score=episode_row.partial_success,
                duration=episode_row.duration,
                cameras=cameras,
                head_to_head=head_to_head,
            )

            # Annotate each episode using a VLM
            episode.annotate()

            # Add the episode to the policy
            if policy_name in policies:
                policies[policy_name].add_episode(episode)


def analyze_head_to_head_evaluations_per_policy(policies: dict[str, Policy]) -> None:
    """
    Analyze the head-to-head evaluations for each policy.
    """
    logger.info("Analyzing head-to-head evaluations for each policy.")

    results: list[PolicyPerformanceAnalysis] = []

    for policy_name, policy in policies.items():
        logger.info(f"Generating full analysis report for policy {policy_name}.")

        # Gather all the head-to-head reports
        all_head_to_head_episodes: list[Episode] = (
            policy.get_all_head_to_head_episodes()
        )
        reports: list[str] = [episode.report for episode in all_head_to_head_episodes]
        session_id_to_video_path: dict[str, str] = {
            str(episode.session.id): os.path.relpath(
                episode.cameras.sample_video(), OUTPUT_DIR
            )
            for episode in all_head_to_head_episodes
        }
        session_id_to_prompt: dict[str, str] = {
            str(episode.session.id): episode.session.prompt
            for episode in all_head_to_head_episodes
        }

        # Construct the prompt to get the summary
        episode_text = "\n\n".join(
            f"========== Episode Report #{i + 1} ==========\n{report}"
            for i, report in enumerate(reports)
        )

        prompt = textwrap.dedent(
            f"""\
        We are evaluating a policy named {policy_name} deployed on a robot arm to perform various manipulation tasks.
        This policy was compared head-to-head against other policies across multiple episodes. Each episode includes:
        - A session ID
        - A task description
        - A scene and task analysis
        - Head-to-head evaluation results

        Using the episode data provided, generate a **structured and comprehensive summary report** in the format below:

        1. **Policy Overview**  
           A brief paragraph summarizing the general behavior, capabilities, and limitations of the policy.

        2. **Comparative Performance**  
           How the policy performed in head-to-head comparisons against other policies. Include overall win/loss/tie statistics when possible, and cite specific task outcomes using session IDs wrapped in `<ref>...</ref>` tags. Highlight task types where the policy consistently outperforms or underperforms others. Then go deep into the details and analyze the performance of the policy in each episode in respect to the other policy using the head-to-head evaluation notes. Summarize and list 5-10 bullet points with key insights. Make sure in this section every claim about the policy is in respect to other competing policies. Do not discuss the policy in isolation.

        3. **Strengths**  
           Bullet-pointed list of notable strengths in manipulation behavior or general reliability. Focus on generalizable behaviors like smooth trajectories, robust grasping, or adaptability. Use concrete examples and session ID citations.

        4. **Weaknesses**  
           Bullet-pointed list of recurring limitations or error patterns. Mention issues such as fine motor control, object confusion, multi-step failure, etc. Include session ID references with `<ref>` tags.

        5. **Instruction Following**  
           Analyze how well the policy understands and executes task instructions. Note sensitivity to language structure, ability to follow negated or relational commands, issues with ambiguous phrasing, abilty to handle typos, etc. Cite session-specific evidence.

        6. **Reasoning**  
           Evaluate the policy's ability to reason about both the **scene context** (e.g., spatial relationships, object visibility) and the **text instruction** (e.g., goal inference, conditional logic). Mention cases where reasoning appears strong or deficient. Use `<ref>` tags to support your analysis.

        7. **Manipulation Skills**  
           Describe the physical performance of the policy: grasping, placing, stacking, inserting, pouring, drawer use, and recovery from errors. Use examples to show when skills succeed or fail.

        8. **Robustness to Scene Variations**  
           Assess the policy's performance under different lighting, clutter levels, object positions, and camera views. Note any sensitivities to occlusion or distractors, etc.

        9. **Common Failure Modes**  
           List frequently observed failures (e.g., freezing mid-task, grabbing wrong item, failing passive commands). Provide short descriptions and supporting citations.

        **Instructions:**
        - Use `<ref>session_id</ref>` to wrap all cited session IDs so they can be linked in the UI. Try to cite as many session IDs as possible to support your claims, but ensure they are relevant.
        - Avoid generalizing from a single example; instead, focus on patterns across multiple episodes.
        - Keep the tone analytical and professional, emphasizing repeatable behaviors and insights.

        The episode reports are as follows:

        {episode_text}
        """
        )

        # Generate full report
        # Run inference Using a strong reasoning model to generate this analysis report.
        response, _ = openai_client.run_inference(
            model="o3-2025-04-16",
            messages=[{"role": "user", "content": prompt}],
            reasoning_effort="high",
            max_completion_tokens=100_000,
        )
        full_report: str = response["choices"][0]["message"]["content"]

        summary_prompt = textwrap.dedent(
            f"""\
Given the following full evaluation report of a robot manipulation policy, generate a concise, high-quality summary that captures the main findings from sections 2 through 8.

Each bullet should summarize the corresponding section in a few sentence fragments, focusing on the most important points. Avoid excessive detail, ensure clarity and correctness.

Use the following format exactly:

- Comparative Performance: <summary>

- Strengths: <summary>

- Weaknesses: <summary>

- Instruction Following: <summary>

- Reasoning: <summary>

- Manipulation Skills: <summary>

- Robustness to Scene Variations: <summary>

- Common Failure Modes: <summary>

Place a line break between each bullet point. Don't output anything before or after the bullet points.

Here is the full report to summarize:

{full_report}"""
        )

        # Summarize the full report
        summary_response, _ = openai_client.run_inference(
            model="o3-2025-04-16",
            messages=[{"role": "user", "content": summary_prompt}],
            reasoning_effort="high",
            max_completion_tokens=30_000,
        )
        summary: str = summary_response["choices"][0]["message"]["content"]

        result = PolicyPerformanceAnalysis(
            policy_name=policy_name,
            number_of_head_to_head_evaluations=len(all_head_to_head_episodes),
            full_report=full_report,
            summary=summary,
            episode_reports=reports,
            session_id_to_video_path=session_id_to_video_path,
            session_id_to_prompt=session_id_to_prompt,
        )
        results.append(result)

    # Save the full report and summary to a JSON file
    with open(ANALYSIS_JSON_PATH, "w") as f:
        json.dump([asdict(result) for result in results], f, indent=4)
        logger.info(f"Saved analysis results to {ANALYSIS_JSON_PATH}.")


@app.route("/")
def index():
    # pass the raw JSON into the template
    return render_template("analysis.html")


@app.route("/policy_analysis.json")
def serve_policy_json():
    return send_from_directory(OUTPUT_DIR, "policy_analysis.json")


@app.route("/videos/<path:video_path>")
def serve_video(video_path):
    """
    Serve videos stored under OUTPUT_DIR.
    """
    full_video_path = os.path.join(OUTPUT_DIR, video_path)
    if not os.path.isfile(full_video_path):
        abort(404)
    # Serve from output folder
    return send_from_directory(OUTPUT_DIR, video_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-refresh",
        action="store_true",
        help="Skip regeneration and use cached JSON file",
    )
    args = parser.parse_args()

    gcs_bucket_name: str = "distributed_robot_eval"
    output_path: str = "output"
    cache_path: str = os.path.join(output_path, "cache")
    os.makedirs(cache_path, exist_ok=True)

    # To make LLM inference calls for analysis
    with open("configs/llm_inference.yaml", "r") as f:
        llm_inference_config = yaml.safe_load(f)

    openai_client = OpenAIClient(
        api_key=llm_inference_config["openai_api_key"],
        cache_dir=cache_path,
    )

    # Connect to the database
    # TODO: use the localhost url
    database_url = "postgresql://centralserver:m3lxcf830x20g4@localhost:5432/real_eval"
    database_url = (
        "postgresql://centralserver:m3lxcf830x20g4@34.55.101.123:5432/real_eval"
    )
    SessionLocal = initialize_database_connection(database_url)
    logger.info(f"Database connection to {database_url} initialized.")

    if not args.skip_refresh:
        # Get all valid policies and populate them with the episodes across all sessions. Download recordings.
        valid_policies: dict[str, Policy] = get_all_valid_policies()
        populate_policy_episodes(valid_policies, gcs_bucket_name)

        # ANALYSIS
        # 1. For each policy, analyze all its head-to-head comparisons and notes from all of its episodes
        analyze_head_to_head_evaluations_per_policy(valid_policies)
        logger.info("Analysis completed.")
    else:
        assert os.path.exists(
            ANALYSIS_JSON_PATH
        ), f"Cached analysis JSON file not found: {ANALYSIS_JSON_PATH}"
        logger.info("Skipping analysis. Using cached analysis JSON file.")

    # Start the Flask server
    logger.info(f"Serving the analysis page.")
    app.run(host="0.0.0.0", port=8888, debug=False)
