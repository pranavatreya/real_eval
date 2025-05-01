from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import argparse
import json
import os
import pdb
import random
import textwrap
import yaml

from pydantic import BaseModel
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

# For the paper, make session cutoff time of April 30, 2025 18:00 UTC or April 30, 2025, 11:00 AM PDT
SESSION_CUTOFF_TIME = datetime(2025, 4, 30, 18, 0, 0, tzinfo=timezone.utc)

BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
OUTPUT_DIR = os.path.abspath(os.path.join(ROOT_DIR, "output"))
ANALYSIS_JSON_PATH = os.path.join(OUTPUT_DIR, "policy_analysis.json")
SIMPLE_ANALYSIS_JSON_PATH = os.path.join(OUTPUT_DIR, "simple_policy_analysis.json")

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

    def get_first_frame_paths(self) -> list[str]:
        """
        Get the first frame paths from all available cameras.
        """
        first_frame_paths = []
        if self.left_first_frame_local_path and os.path.exists(
            self.left_first_frame_local_path
        ):
            first_frame_paths.append(self.left_first_frame_local_path)
        if self.right_first_frame_local_path and os.path.exists(
            self.right_first_frame_local_path
        ):
            first_frame_paths.append(self.right_first_frame_local_path)
        if self.wrist_first_frame_local_path and os.path.exists(
            self.wrist_first_frame_local_path
        ):
            first_frame_paths.append(self.wrist_first_frame_local_path)
        return first_frame_paths


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

    metadata: dict | None = None
    """Simple metadata annotated by a VLM using the first frames of the cameras and the task at hand."""

    def annotate(self) -> None:
        """
        Annotate the episode using first frames from available cameras.
        """
        if not self.head_to_head:
            # Don't annotate the beginning of the episode if there is no head-to-head evaluation
            return

        image_paths = self.cameras.get_first_frame_paths()
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

    def compute_metadata(self) -> None:
        if not self.head_to_head:
            return

        image_paths = self.cameras.get_first_frame_paths()
        if not image_paths:
            logger.warning(
                f"No valid first-frame images found for session {self.session.id}"
            )
            return

        examples_text = (
            "Task categories and their examples:\n\n"
            "Pick and Place\n"
            "- pick up the ball and put it in the bowl\n"
            "- pick the blue box and put it in the dustpan\n"
            "- pick the carrot and place it in the yellow dish\n"
            "- pick the stuffed animal and put it in the sink\n"
            "- place the pineapple into the blue tray\n\n"
            "Open / Close\n"
            "- open the drawer\n"
            "- open the coffee machine\n"
            "- open the green book\n"
            "- close the black and pink glasses case\n"
            "- close the laptop screen\n\n"
            "Move / Slide\n"
            "- move the clipper into the jar\n"
            "- move the computer mouse to the left\n"
            "- move the deck of card to notebook\n"
            "- push the dustpan to the right\n"
            "- push the plate into the cup\n\n"
            "Knock Over / Topple\n"
            "- knock the brown bear off the box\n"
            "- knock the cup off the table\n"
            "- just knock off the green frog off the brown box and nothing else\n"
            "- push over the stacked blocks on the table\n"
            "- push over the white box\n\n"
            "Cover / Drape / Fold\n"
            "- drape the cloth over the box\n"
            "- drape the white cloth over the chair\n"
            "- cover the bowl with the blue plate\n"
            "- cover the yellow bowl with the towel\n"
            "- fold the blue towel\n\n"
            "Group / Organize / Stack\n"
            "- stack the bowls\n"
            "- stack the cups to form a pyramid\n"
            "- gather all items\n"
            "- place all items on an orange tile\n"
            "- place the blue block next to the green block\n\n"
            "Find / Search\n"
            "- find and pick up the pineapple on the shelf\n"
            "- find the bread\n"
            "- find the fruit\n"
            "- find the plant on the bookshelf and place into bowl\n"
            "- find the yellow object, pick it up, and place in the bowl\n"
            "- just touch the red box and nothing else\n"
            "- point at the kettle\n\n"
            "Minimal or No Action\n"
            "- do absolutely nothing. do not move\n"
            "- do not move\n"
            "- point your end gripper straight horizontally and freeze after\n\n"
            "Object Manipulation (Fine Motor Skills)\n"
            "- upright the cup\n"
            "- rotate the kettle 90 degrees clockwise\n"
            "- uncap the pen\n"
            "- unplug the black cable\n"
            "- remove the wrench from the beaker\n"
            "- pour the water from the mug into the silver bowl\n"
            "- pour the coffee out of the test tube on to the plate\n"
            "- pour the mug contents into the bowl\n"
            "- pour water from the teapot to the pot\n"
            "- pour the nuts from the red cup onto the plate\n\n"
            "Sorting / Classification\n"
            "- move the objects with similar color together\n"
            "- put all carrots into the bowl\n"
            "- put all cups into the yellow bowl\n"
            "- put all red items in the bowl\n"
            "- separate two different books by touching them (please touch two different books)\n\n"
            "Tool Use\n"
            "- use black eraser to clean white board\n"
            "- use the green marker to write on the white board\n"
            "- stir the pot\n"
            "- stir the pan with the spoon\n"
            "- clean the table\n\n"
        )

        prompt = (
            f"You are given first-frame images from a robot manipulation episode and a task description:\n\n"
            f"Task description:\n{self.session.prompt.strip()}\n\n"
            f"{examples_text}\n"
            f"Please infer the following metadata fields based on the task description and provided images:\n\n"
            f"- Task category: Choose the category that best describes the manipulation skill required, based on the examples above.\n\n"
            f"- Whether the camera view is clear: Determine if the images provide a full, unobstructed view of the objects and workspace necessary for the task. "
            f"Mark as clear if the view enables the robot to see all relevant elements; otherwise, mark as unclear if important parts are hidden, cropped, or poorly framed.\n\n"
            f"- Whether the lighting is good: Assess whether the lighting in the scene makes objects and their placement easy to perceive. "
            f"Good lighting means the scene is evenly lit without strong glare, harsh shadows, or dim areas that could affect task execution.\n\n"
            f"- Whether the task description is clear: Evaluate the clarity of the task text alone. A clear description is grammatically correct and unambiguous. "
            f"Mark it unclear if there are major spelling/grammar issues, missing context, or vague/inconsistent instructions.\n\n"
            f"- Whether the scene is simple or cluttered: Consider the overall environment shown in the images. Mark the scene as simple if it contains only the key objects and little to no clutter or distractors. "
            f"If the workspace includes many irrelevant or occluding items, mark it as cluttered.\n\n"
            f"- Whether the task requires reasoning: Indicate if the task involves higher-level reasoning (e.g., conditional logic, negation, relational or spatial inference, or concept-based grouping). "
            f"Examples that require reasoning include: 'pick up the non-read object' (negation), 'just touch the red box and nothing else' (selective suppression), "
            f"'find the object that is not stacked' (spatial reasoning), 'if there is a frog, knock it off' (conditional), and 'move similar-colored objects together' (conceptual grouping). "
            f"Tasks like 'pick up the red cup and place it in the bowl' do not require reasoning and should be marked as not requiring reasoning.\n\n"
            f"Only return a structured response matching the schema you're instructed to return."
        )

        max_retries = 5
        for attempt in range(max_retries):
            try:
                parsed, cached = openai_client.run_image_inference_structured(
                    model="gpt-4.5-preview-2025-02-27",
                    image_paths=image_paths,
                    text=prompt,
                    expected_structure=TaskSceneMetadata,
                    temperature=0,
                    max_tokens=1024,
                    force_recompute=(attempt > 0),
                )

                # Dump to JSON-like dict
                metadata = parsed.model_dump(mode="json")

                # Validate fields
                if metadata["task_category"] not in ALL_TASK_CATEGORIES:
                    raise ValueError(f"Invalid task_category: {metadata['task_category']}")

                for field in [
                    "clear_camera_view",
                    "good_lighting",
                    "clear_task_description",
                    "simple_scene",
                    "reasoning_required",
                ]:
                    if not isinstance(metadata[field], bool):
                        raise TypeError(f"Field {field} must be a boolean, got {type(metadata[field])} instead.")

                # If all checks pass
                self.metadata = metadata
                return
            except Exception as e:
                logger.warning(
                    f"Metadata annotation failed for session {self.session.id} on attempt {attempt + 1}: {e}"
                )

        # If we exhausted all retries, raise error
        raise RuntimeError(
            f"Failed to compute metadata for session {self.session.id} after {max_retries} attempts."
        )

    @property
    def report(self) -> str:
        """
        Generate a report that summarizes the episode with the head-to-head evaluation notes.
        """
        assert self.metadata and "task_category" in self.metadata and self.annotations
        assert (
            self.head_to_head
        ), "Episode has no head-to-head evaluation, so nothing to report."

        return textwrap.dedent(
            f"""Session ID: {self.session.id}
Task: {self.session.prompt}
Task category: {self.metadata['task_category']}

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


class TaskCategory(str, Enum):
    pick_and_place = "Pick and Place"
    open_close = "Open / Close"
    move_slide = "Move / Slide"
    knock_over_topple = "Knock Over / Topple"
    cover_drape_fold = "Cover / Drape / Fold"
    group_organize_stack = "Group / Organize / Stack"
    find_search = "Find / Search"
    minimal_or_no_action = "Minimal or No Action"
    object_manipulation = "Object Manipulation"
    sorting_classification = "Sorting / Classification"
    tool_use = "Tool Use"


class TaskSceneMetadata(BaseModel):
    task_category: TaskCategory
    clear_camera_view: bool
    good_lighting: bool
    clear_task_description: bool
    simple_scene: bool
    reasoning_required: bool


ALL_TASK_CATEGORIES = {category.value for category in TaskCategory}


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

    # Gather all valid evaluation sessions. that are before the cutoff time
    # Valid eval sessions have VALID_SESSION: in the beginning of `evaluation_notes`
    # and check that episodes is not empty
    sessions = (
        db.query(SessionModel)
        .filter(
            SessionModel.evaluation_notes.like("VALID_SESSION:%"),
            SessionModel.episodes.any(),
            SessionModel.session_creation_timestamp <= SESSION_CUTOFF_TIME,
        )
        .order_by(SessionModel.session_uuid)
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

            # Annotate each episode
            episode.annotate()
            episode.compute_metadata()

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
        logger.info(f"Generating full report for policy {policy_name} with {len(reports)} head-to-head evaluations.")
        episode_text = "\n\n".join(
            f"========== Episode Report #{i + 1} ==========\n{report}"
            for i, report in enumerate(reports)
        )

        prompt = textwrap.dedent(
            f"""\
We are evaluating a policy named {policy_name} deployed on a robot arm to perform various manipulation tasks.
This policy was compared head-to-head against other policies across multiple episodes. Each episode includes:
- A session ID
- A task description and the task category it belongs to. The possible task categories are: {', '.join(ALL_TASK_CATEGORIES)}.
- A scene and task analysis
- Head-to-head evaluation results

Using the episode data provided, generate a **structured and comprehensive summary report** in the format below:

1. **Policy Overview**  
A brief paragraph summarizing the general behavior, capabilities, and limitations of the policy.

2. **Comparative Performance**  
How the policy performed in head-to-head comparisons against other policies across the different task categories. For each task category, create a bullet point with a discussion of how the policy consistently outperformed or underperformed compared to all the other policies. Make sure in this section that every claim about the policy is with respect to other competing policies. When making a claim, always mention how the other policies performed in comparison to the current policy. Do not discuss the policy in isolation. Don't mention a task category unelss there is evidence of the policy performing well or poorly in that category across multiple episodes. Make your claims based on overall performance or underperformance for specific task categories rather than individual episodes. There is no need to reference specific session IDs in this section (no <ref> tags).

3. **Strengths**  
Bullet-pointed list of notable strengths in manipulation behavior or general reliability. Mention the task categories the policy is good at (if any) instead of basing a claim on a single instance. Focus on generalizable behaviors like smooth trajectories, robust grasping, or adaptability. Use concrete examples and session ID citations.

4. **Weaknesses**  
Bullet-pointed list of recurring limitations or error patterns. Mention the task categories the policy is poor at instead of basing a claim on a single instance. Mention issues such as fine motor control, object confusion, multi-step failure, etc. Include session ID references with `<ref>` tags.

5. **Instruction Following**  
Analyze how well the policy understands and executes task instructions. Note sensitivity to language structure, ability to follow negated or relational commands, issues with ambiguous phrasing, ability to handle typos, etc. Cite session-specific evidence.

6. **Reasoning**  
Evaluate the policy's ability to reason about both the **scene context** (e.g., spatial relationships, object visibility) and the **text instruction** (e.g., goal inference, conditional logic). Mention cases where reasoning appears strong or deficient. Use `<ref>` tags to support your analysis.

7. **Manipulation Skills**  
Describe the physical performance of the policy: grasping, placing, stacking, inserting, pouring, drawer use, and recovery from errors. Use examples to show when skills succeed or fail.

8. **Robustness to Scene Variations**  
Assess the policy's performance under different lighting, clutter levels, object positions, and camera views. Note any sensitivities to occlusion or distractors, etc.

9. **Common Failure Modes**  
List frequently observed failures (e.g., freezing mid-task, grabbing wrong item, failing passive commands). Provide short descriptions and supporting citations.

**Instructions:**
- When referring to a session, always cite the full session ID (UUID format, e.g., 16e5bbda-57c1-4e58-a24a-b39ee8142d41) exactly as provided. Do not shorten, truncate or modify it in any way.
- Always wrap session IDs inside <ref>...</ref> tags. Example: <ref>16e5bbda-57c1-4e58-a24a-b39ee8142d41</ref>
- Try to cite as many session IDs as possible to support your claims, but only if they are relevant to the point you're making.
- Avoid generalizing from a single episode unless there is clear evidence of a pattern.
- Keep the tone analytical and professional, emphasizing repeatable behaviors and insights.
- Do not invent session IDs. Only use session IDs present in the provided episode reports.
- There is no need to mention the specific number of episodes and wins/losses/ties in head-to-head evaluations in this report.

The individual episode reports are as follows:

{episode_text}"""
        )

        # Generate full report
        # Run inference Using a strong reasoning model to generate this analysis report.
        response, cached = openai_client.run_inference(
            model="o3-2025-04-16",
            messages=[{"role": "user", "content": prompt}],
            reasoning_effort="high",
            max_completion_tokens=100_000,
        )
        if cached:
            logger.info("Full report was generated before.")

        full_report: str = response["choices"][0]["message"]["content"]

        summary_prompt = textwrap.dedent(
            f"""\
Given the following full evaluation report of a robot manipulation policy, generate a concise, high-quality summary that captures the main findings from sections 1 through 9.

Each bullet should summarize the corresponding section in a few sentence fragments, focusing on the most important points. Avoid excessive detail, ensure clarity and correctness. Stick to the facts presented in the full report.

Use the following format exactly:

- Policy Overview: <summary>

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


def debug(task_category_to_commands: dict[str, set[str]], output_path: str) -> None:
    """
    Launch an interactive CLI tool to test how well LLM classifications align with human judgment.
    For each task category, show a random mix of 50 in-category and 50 out-of-category commands
    (or fewer if not enough examples exist), and ask the user to guess if each one belongs to that category.
    Log everything to output/debug.txt
    """
    log_path: str = os.path.join(output_path, "debug.txt")
    if os.path.exists(log_path):
        logger.warning(f"You already took the test. Check: {log_path}")

    all_categories = list(task_category_to_commands.keys())
    log_lines = []
    total = 0
    correct = 0

    print("\nStarting task category verification test. Answer y (yes) or n (no) for each prompt.\n")
    for target_category in all_categories:
        print(f"\n--- Testing category: {target_category} ---")
        in_category = task_category_to_commands[target_category]
        out_category = [
            cmd
            for cat, cmds in task_category_to_commands.items()
            if cat != target_category
            for cmd in cmds
        ]

        n_in = min(50, len(in_category))
        n_out = min(n_in, len(out_category))
        examples = [(cmd, True) for cmd in random.sample(list(in_category), n_in)] + \
                   [(cmd, False) for cmd in random.sample(list(out_category), n_out)]
        random.shuffle(examples)

        category_total = 0
        category_correct = 0

        for cmd, is_in_category in examples:
            print(f"\nTask description: {cmd}")
            while True:
                answer = input("Is this '{0}'? (y/n): ".format(target_category)).strip().lower()
                if answer in ("y", "n"):
                    break
                print("Please enter 'y' or 'n'.")

            user_thinks_in_category = (answer == "y")
            match = user_thinks_in_category == is_in_category
            total += 1
            category_total += 1
            if match:
                correct += 1
                category_correct += 1

            log_lines.append(
                f"[{target_category}] Prompt: {cmd} | Actual: {is_in_category} | User: {user_thinks_in_category} | Correct: {match}"
            )

        print(f"\nAccuracy for '{target_category}': {category_correct}/{category_total} ({category_correct / category_total:.1%})")
        log_lines.append(f"Category '{target_category}' accuracy: {category_correct}/{category_total}\n")

    print(f"\n== Final accuracy across all categories: {correct}/{total} ({correct / total:.1%}) ==")
    log_lines.append(f"\nOverall accuracy: {correct}/{total} ({correct / total:.1%})")

    with open(log_path, "w") as log_file:
        log_file.write("\n".join(log_lines))

    print(f"\nFull debug session saved to {log_path}")


def compare_best_against_others(all_valid_policies: dict[str, Policy]) -> None:
    from statistics import mean

    scores_by_category = defaultdict(
        lambda: {
            "pi0_fast_droid": [],
            "others_incl_paligemma": [],
            "others_excl_paligemma": [],
        }
    )

    for policy in all_valid_policies.values():
        for episode in policy.episodes:
            assert episode.metadata and "task_category" in episode.metadata, \
                f"Missing metadata for episode in policy {policy.name}, session {episode.session.id}"

            category = episode.metadata["task_category"]
            score = episode.partial_success_score

            if policy.name == "pi0_fast_droid":
                scores_by_category[category]["pi0_fast_droid"].append(score)
            else:
                scores_by_category[category]["others_incl_paligemma"].append(score)
                if policy.name != "paligemma_binning_droid":
                    scores_by_category[category]["others_excl_paligemma"].append(score)

    logger.info("\nAverage partial success scores by task category:\n")

    for category in sorted(scores_by_category.keys()):
        data = scores_by_category[category]

        def safe_avg(scores: list[float]) -> str:
            return f"{mean(scores):.3f}" if scores else "N/A"

        logger.info(f"{category}:")
        logger.info(f"  pi0_fast_droid:                        ({len(data['pi0_fast_droid']):>3} eps) {safe_avg(data['pi0_fast_droid'])}")
        logger.info(f"  all other policies:                   ({len(data['others_incl_paligemma']):>3} eps) {safe_avg(data['others_incl_paligemma'])}")
        logger.info(f"  all others (excl paligemma_binning): ({len(data['others_excl_paligemma']):>3} eps) {safe_avg(data['others_excl_paligemma'])}\n")


def compare_head_to_head_win_rates(all_valid_policies: dict[str, Policy]) -> None:
    outcomes_by_category = defaultdict(lambda: {
        "pi0_fast_droid": {"wins": 0, "total": 0},
        "others_incl_paligemma": {"wins": 0, "total": 0},
        "others_excl_paligemma": {"wins": 0, "total": 0},
    })

    for policy in all_valid_policies.values():
        for episode in policy.get_all_head_to_head_episodes():
            if not episode.metadata or "task_category" not in episode.metadata:
                raise ValueError(f"Missing task_category in episode metadata for policy {policy.name}, session {episode.session.id}")

            category = episode.metadata["task_category"]
            perspective = episode.head_to_head.perspective_policy
            won = episode.head_to_head.won

            if perspective == "pi0_fast_droid":
                outcomes_by_category[category]["pi0_fast_droid"]["total"] += 1
                if won:
                    outcomes_by_category[category]["pi0_fast_droid"]["wins"] += 1
            else:
                outcomes_by_category[category]["others_incl_paligemma"]["total"] += 1
                if won:
                    outcomes_by_category[category]["others_incl_paligemma"]["wins"] += 1

                if perspective != "paligemma_binning_droid":
                    outcomes_by_category[category]["others_excl_paligemma"]["total"] += 1
                    if won:
                        outcomes_by_category[category]["others_excl_paligemma"]["wins"] += 1

    logger.info("\nHead-to-head win rates by task category:\n")

    for category in sorted(outcomes_by_category.keys()):
        data = outcomes_by_category[category]

        def fmt(wins: int, total: int) -> str:
            return f"{wins}/{total} ({(wins / total * 100):.1f}%)" if total > 0 else "N/A"

        logger.info(f"{category}:")
        logger.info(f"  pi0_fast_droid:                        {fmt(**data['pi0_fast_droid'])}")
        logger.info(f"  all other policies:                   {fmt(**data['others_incl_paligemma'])}")
        logger.info(f"  all others (excl paligemma_binning): {fmt(**data['others_excl_paligemma'])}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-refresh",
        action="store_true",
        help="Skip regeneration and use cached JSON file",
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Whether to output policies and metadata",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode to evaluate task classification alignment",
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
    # database_url = "postgresql://centralserver:m3lxcf830x20g4@localhost:5432/real_eval"
    database_url = (
        "postgresql://centralserver:m3lxcf830x20g4@34.55.101.123:5432/real_eval"
    )

    SessionLocal = initialize_database_connection(database_url)
    logger.info(f"Database connection to {database_url} initialized.")

    if not args.skip_refresh:
        # Get all valid policies and populate them with the episodes across all sessions. Download recordings.
        valid_policies: dict[str, Policy] = get_all_valid_policies()
        populate_policy_episodes(valid_policies, gcs_bucket_name)

        # Build a mapping off prompt to task category
        prompt_to_task_category: dict[str, str] = {}
        task_category_to_commands: dict[str, set[str]] = defaultdict(set)
        for policy in valid_policies.values():
            for episode in policy.episodes:
                if episode.metadata and "task_category" in episode.metadata:
                    category = episode.metadata["task_category"]

                    prompt_to_task_category[episode.session.prompt] = category

                    command = episode.session.prompt.strip()
                    task_category_to_commands[category].add(command)

        # Use the mapping to set the task category for each episode of each policy when metadata doesn't exist
        for policy in valid_policies.values():
            for episode in policy.episodes:
                if not episode.metadata or "task_category" not in episode.metadata:
                    assert episode.session.prompt in prompt_to_task_category
                    task_category = prompt_to_task_category[episode.session.prompt]
                    episode.metadata = {"task_category": task_category}

        if args.debug:
            # Call the CLI debug tool
            debug(task_category_to_commands, output_path)
            exit(0)

        # ANALYSIS
        if not args.simple:
            # For each policy, analyze all its head-to-head comparisons and notes from all of its episodes
            analyze_head_to_head_evaluations_per_policy(valid_policies)
        else:
            compare_best_against_others(valid_policies)
            compare_head_to_head_win_rates(valid_policies)

            # Just dump the policies with their episodes that have head-to-head evaluations
            with open(SIMPLE_ANALYSIS_JSON_PATH, "w") as f:
                json.dump(
                    [
                        {
                            "name": policy.name,
                            "episodes": [asdict(ep) for ep in policy.episodes],
                        }
                        for policy in valid_policies.values()
                    ],
                    f,
                    indent=4,
                    default=str,
                )
        logger.info("Analysis completed.")
    else:
        assert os.path.exists(
            ANALYSIS_JSON_PATH
        ), f"Cached analysis JSON file not found: {ANALYSIS_JSON_PATH}"
        logger.info("Skipping analysis. Using cached analysis JSON file.")

    # Start the Flask server
    logger.info(f"Serving the analysis page.")
    app.run(host="0.0.0.0", port=8888, debug=False)
