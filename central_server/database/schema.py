import datetime

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import UUID as PG_UUID


Base = declarative_base()


class PolicyModel(Base):
    """
    Stores info about each policy we can potentially evaluate.
    Columns:
        - id (PK)
        - unique_policy_name
        - ip_address
        - port
        - is_in_use
        - elo_score
        - times_in_ab_eval
        - last_time_evaluated
        - owner_name
        - robot_arm_type
        - created_at
    """

    __tablename__ = "policies"

    id = Column(Integer, primary_key=True)
    unique_policy_name = Column(String, unique=True, nullable=False)
    ip_address = Column(String, nullable=True)
    port = Column(Integer, nullable=True)
    is_in_use = Column(Boolean, default=False, nullable=False)
    elo_score = Column(Float, default=1200.0)  # Just a typical default
    times_in_ab_eval = Column(Integer, default=0)
    last_time_evaluated = Column(DateTime, nullable=True)
    owner_name = Column(String, nullable=True)
    robot_arm_type = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class SessionModel(Base):
    """
    The top-level entity for an evaluation session.
    Columns:
        - id (PK)
        - session_uuid
        - evaluation_type ("A/B" or "single-policy")
        - evaluation_location
        - evaluator_name
        - robot_name
        - session_creation_timestamp
        - session_completion_timestamp
        - evaluation_notes

        - policyA and policyB store references to the policy’s IP/port or unique name?
          They might store the unique_policy_name or just text data.
          For an A/B eval, we fill both columns; for single-policy, fill policyA only.
    """

    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True)
    session_uuid = Column(PG_UUID(as_uuid=True), unique=True, nullable=False)
    evaluation_type = Column(String, nullable=False)  # "A/B" or "single-policy"
    evaluation_location = Column(String, nullable=True)
    evaluator_name = Column(String, nullable=True)
    robot_name = Column(String, nullable=True)

    session_creation_timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    session_completion_timestamp = Column(DateTime, nullable=True)

    evaluation_notes = Column(Text, nullable=True)

    # Hard-coded references to two policies for an A/B session
    policyA_name = Column(String, nullable=True)
    policyB_name = Column(String, nullable=True)

    # Relationship: one Session -> many Episodes
    episodes = relationship("EpisodeModel", back_populates="parent_session", cascade="all, delete-orphan")


class EpisodeModel(Base):
    """
    Each row is one rollout or "episode" within a session.
    Columns (some are new):
        - id (PK)
        - session_id (FK)
        - policy_name
        - command
        - binary_success (0 or 1)
        - partial_success (0..1)
        - duration (# timesteps)
        - gcs_left_cam_path
        - gcs_right_cam_path
        - gcs_wrist_cam_path
        - npz_file_path
        - timestamp
        - policy_ip
        - policy_port
        - third_person_camera_type (e.g. "left" or "right")
        - third_person_camera_id (e.g. 223492)
        - feedback (text)
    """

    __tablename__ = "episodes"

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)

    policy_name = Column(String, nullable=False)
    command = Column(Text, nullable=True)

    binary_success = Column(Integer, nullable=True)  # 0 or 1
    partial_success = Column(Float, nullable=True)  # 0.0 to 1.0

    duration = Column(Integer, nullable=True)  # number of timesteps

    gcs_left_cam_path = Column(String, nullable=True)
    gcs_right_cam_path = Column(String, nullable=True)
    gcs_wrist_cam_path = Column(String, nullable=True)
    npz_file_path = Column(String, nullable=True)

    # Which policy IP/port
    policy_ip = Column(String, nullable=True)
    policy_port = Column(Integer, nullable=True)

    third_person_camera_type = Column(String, nullable=True)  # e.g. "left" or "right"
    third_person_camera_id = Column(Integer, nullable=True)

    feedback = Column(Text, nullable=True)

    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    parent_session = relationship("SessionModel", back_populates="episodes")
