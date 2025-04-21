from dataclasses import dataclass, field
import os
import pdb
import textwrap
import yaml

from tqdm import tqdm
import fsspec

from database.schema import PolicyModel, SessionModel
from database.connection import initialize_database_connection
from llm.openai_client import OpenAIClient
from logger import logger


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
    """Path to the left camera video on GCS."""

    left_local_path: str | None = None
    """Path to the left camera video on local storage."""

    right_gcs_path: str | None = None
    """Path to the right camera video on GCS."""

    right_local_path: str | None = None
    """Path to the right camera video on local storage."""

    wrist_gcs_path: str | None = None
    """Path to the wrist camera video on GCS."""

    wrist_local_path: str | None = None
    """Path to the wrist camera video on local storage."""

    def download(self, gcs_bucket: str) -> None:
        """
        Download the camera videos from GCS to local storage.
        """
        if self.left_gcs_path:
            local_path = os.path.join(output_path, self.left_gcs_path)
            download_from_gcs(f"gs://{gcs_bucket}/{self.left_gcs_path}", local_path)
        if self.right_gcs_path:
            local_path = os.path.join(output_path, self.right_gcs_path)
            download_from_gcs(f"gs://{gcs_bucket}/{self.right_gcs_path}", local_path)
        if self.wrist_gcs_path:
            local_path = os.path.join(output_path, self.wrist_gcs_path)
            download_from_gcs(f"gs://{gcs_bucket}/{self.wrist_gcs_path}", local_path)


@dataclass
class HeadToHead:
    session: Session
    """The session this head-to-head evaluation belongs to."""

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

    def report(self) -> str:
        """
        Generate a report that summarizes the head-to-head evaluation in perspective of `perspective_policy` policy`.
        """
        return textwrap.dedent(f"""
        Session #{self.session.id}
        Task: {self.session.prompt}
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

    head_to_head: HeadToHead | None
    """Head-to-head evaluation data for the episode (if exists)."""


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

    def get_all_head_to_head(self) -> list[HeadToHead]:
        """
        Get all head-to-head evaluations for this policy.
        """
        return [episode.head_to_head for episode in self.episodes if episode.head_to_head]


def get_all_valid_policies() -> dict[str, Policy]:
    """
    Returns all valid policies as a dict where the key is the policy name and the value is the policy
    """
    db = SessionLocal()

    # Gather all policies except PI0 and PI0_FAST
    policies = db.query(PolicyModel).filter(
        PolicyModel.unique_policy_name.notin_(["PI0", "PI0_FAST"])
    ).all()
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
    assert os.path.exists(destination_path), f"Failed to download {gcs_path} to {destination_path}."


def populate_policy_episodes(policies: dict[str, Policy], gcs_bucket: str) -> None:
    """
    Populate the policy episodes with the data from the sessions.
    """
    db = SessionLocal()

    # Gather all valid evaluation sessions.
    # Valid eval sessions have VALID_SESSION: in the beginning of `evaluation_notes`
    # and check that episodes is not empty
    sessions = db.query(SessionModel).filter(
        SessionModel.evaluation_notes.like("VALID_SESSION:%"),
        SessionModel.episodes.any(),
    ).all()
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
                    session=session,
                    perspective_policy=policy_name,
                    was_policy_a=True,
                    won="PREFERENCE=A" in raw_evaluation_notes,
                    tied="PREFERENCE=TIE" in raw_evaluation_notes,
                    ab_notes=ab_notes,
                )
            elif policy_name == policy_b_name:
                # If the policy is B, check if it won
                head_to_head = HeadToHead(
                    session=session,
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

            # Add the episode to the policy
            if policy_name in policies:
                policies[policy_name].add_episode(episode)


def analyze_head_to_head_evaluations_per_policy(policies: dict[str, Policy]) -> None:
    """
    Analyze the head-to-head evaluations for each policy.
    """
    logger.info("Analyzing head-to-head evaluations for each policy.")
    output_path = os.path.join(analysis_path, "head_to_head")
    os.makedirs(output_path, exist_ok=True)

    for policy_name, policy in policies.items():
        output_text_path: str = os.path.join(output_path, f"{policy_name}.txt")
        output_json_path: str = os.path.join(output_path, f"{policy_name}.json")

        # Gather all the head-to-head reports
        all_head_to_head: list[HeadToHead] = policy.get_all_head_to_head()
        reports: list[str] = [head_to_head.report() for head_to_head in all_head_to_head]

        # Construct the prompt to get the summary
        prompt = textwrap.dedent(f"""\
We are evaluating a policy named {policy_name} deployed on a robot arm to perform various tasks.
We have conducted head-to-head evaluations of this policy against other policies.
Given all the head-to-head evaluation reports for the policy {policy_name} against other policies, summarize the results.
Discuss the strengths and weaknesses of the policy in both the tasks it can perform and the tasks it cannot perform when compared to other policies.
If the report contains information about the policy's ability to reason, adhere to instructions, planning, etc., please include that information.
Try not to make general claims based on single instances, but rather focus on the overall performance of the policy across all head-to-head evaluations.
Cite specific session numbers and tasks when making claims about the policy's performance.

The head-to-head reports are as follows:

{'\n'.join(reports)}
""")

        # Run inference
        response, _ = openai_client.run_inference(
            model="gpt-4o-2024-11-20",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=2048,
        )
        summary: str = response["choices"][0]["message"]["content"]

        with open(output_text_path, "w") as f:
            f.write(f"Policy: {policy_name}\n")
            f.write(f"Number of episodes: {len(policy.episodes)}\n")
            f.write(f"Number of head-to-head evaluations: {len(all_head_to_head)}\n\n")
            f.write("Summary:\n")
            f.write(summary)
            f.write("\n\n")
            f.write("Head-to-head reports:\n")
            f.write("\n".join(reports))


if __name__ == "__main__":
    gcs_bucket_name: str = "distributed_robot_eval"
    output_path: str = "output"
    cache_path: str = os.path.join(output_path, "cache")
    os.makedirs(cache_path, exist_ok=True)
    analysis_path: str = os.path.join(output_path, "analysis")
    os.makedirs(analysis_path, exist_ok=True)

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
    database_url = "postgresql://centralserver:m3lxcf830x20g4@34.55.101.123:5432/real_eval"
    SessionLocal = initialize_database_connection(database_url)
    logger.info(f"Database connection to {database_url} initialized.")

    # Get all valid policies and populate them with the episodes across all sessions. Download recordings.
    valid_policies: dict[str, Policy] = get_all_valid_policies()
    populate_policy_episodes(valid_policies, gcs_bucket_name)

    # ANALYSIS
    # 1. For each policy, analyze all its head-to-head comparisons and notes from those sessions.
    analyze_head_to_head_evaluations_per_policy(valid_policies)

    logger.info("Analysis completed.")
