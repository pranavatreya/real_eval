import datetime

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import UUID as PG_UUID


Base = declarative_base()


class PolicyModel(Base):
    __tablename__ = "policies"
    id = Column(Integer, primary_key=True)
    unique_policy_name = Column(String, unique=True, nullable=False)
    ip_address = Column(String, nullable=True)
    port = Column(Integer, nullable=True)
    is_in_use = Column(Boolean, default=False, nullable=False)  # We DO NOT use this anymore
    elo_score = Column(Float, default=1200.0)
    times_in_ab_eval = Column(Integer, default=0)
    last_time_evaluated = Column(DateTime, nullable=True)
    owner_name = Column(String, nullable=True)
    robot_arm_type = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class SessionModel(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True)
    session_uuid = Column(PG_UUID(as_uuid=True), unique=True, nullable=False)

    evaluation_type = Column(String, nullable=False)  # Always "A/B"

    evaluation_location = Column(String, nullable=True)
    evaluator_name = Column(String, nullable=True)
    robot_name = Column(String, nullable=True)

    session_creation_timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    session_completion_timestamp = Column(DateTime, nullable=True)

    # We use evaluation_notes to store validity, preference, textual feedback, etc.
    evaluation_notes = Column(Text, nullable=True)

    # We only store 2 policies in the table itself (A & B), but more can be used/returned
    policyA_name = Column(String, nullable=True)
    policyB_name = Column(String, nullable=True)

    episodes = relationship(
        "EpisodeModel", back_populates="parent_session", cascade="all, delete-orphan"
    )


class EpisodeModel(Base):
    __tablename__ = "episodes"
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    policy_name = Column(String, nullable=False)
    command = Column(Text, nullable=True)

    binary_success = Column(Integer, nullable=True)
    partial_success = Column(Float, nullable=True)
    duration = Column(Integer, nullable=True)

    gcs_left_cam_path = Column(String, nullable=True)
    gcs_right_cam_path = Column(String, nullable=True)
    gcs_wrist_cam_path = Column(String, nullable=True)
    npz_file_path = Column(String, nullable=True)

    policy_ip = Column(String, nullable=True)
    policy_port = Column(Integer, nullable=True)

    third_person_camera_type = Column(String, nullable=True)
    third_person_camera_id = Column(Integer, nullable=True)

    # We re-purpose `feedback` to store BOTH policy letter and average latency, e.g. "B;avg_latency=0.24"
    feedback = Column(Text, nullable=True)

    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    parent_session = relationship("SessionModel", back_populates="episodes")

