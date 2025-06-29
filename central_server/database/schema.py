import datetime

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey,
    Text,
    Boolean,
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

Base = declarative_base()


# --------------------------------------------------------------------------- #
#  Users                                                                      #
# --------------------------------------------------------------------------- #
class UserModel(Base):
    """
    Registered benchmark users.

    • `email` is the primary key (unique).
    • Weekly evaluation-credit accrues automatically.
    """

    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, nullable=False)

    participating_corl = Column(Boolean, default=False)
    eval_credit = Column(Integer, default=0)
    last_credit_update = Column(DateTime, default=datetime.datetime.utcnow)


# --------------------------------------------------------------------------- #
#  Policies                                                                   #
# --------------------------------------------------------------------------- #
class PolicyModel(Base):
    """
    Policy servers.

    Column notes
    ------------
    • `owner_name`  : Stores the **owner’s email** (legacy column name kept).
    • `is_in_use`   : ‼️ Repurposed to mean **open-source flag**.
                      True  → policy + source code are open-sourced
                      False → closed-source
    """

    __tablename__ = "policies"

    id = Column(Integer, primary_key=True)

    # Human-readable unique identifier
    unique_policy_name = Column(String, unique=True, nullable=False)

    # Network location
    ip_address = Column(String, nullable=True)
    port = Column(Integer, nullable=True)

    # ↳ now represents “open-source?” (see docstring above)
    is_in_use = Column(Boolean, default=False, nullable=False)

    # Ranking / bookkeeping
    elo_score = Column(Float, default=1200.0)
    times_in_ab_eval = Column(Integer, default=0)
    last_time_evaluated = Column(DateTime, nullable=True)

    owner_name = Column(String, nullable=True)  # stores owner e-mail
    robot_arm_type = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.datetime.utcnow)


# --------------------------------------------------------------------------- #
#  Sessions & Episodes                                                        #
# --------------------------------------------------------------------------- #
class SessionModel(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True)
    session_uuid = Column(PG_UUID(as_uuid=True), unique=True, nullable=False)

    evaluation_type = Column(String, nullable=False)  # always "A/B"

    evaluation_location = Column(String, nullable=True)
    evaluator_name = Column(String, nullable=True)  # evaluator e-mail
    robot_name = Column(String, nullable=True)

    session_creation_timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    session_completion_timestamp = Column(DateTime, nullable=True)

    evaluation_notes = Column(Text, nullable=True)

    policyA_name = Column(String, nullable=True)
    policyB_name = Column(String, nullable=True)

    episodes = relationship(
        "EpisodeModel",
        back_populates="parent_session",
        cascade="all, delete-orphan",
    )


class EpisodeModel(Base):
    __tablename__ = "episodes"

    id = Column(Integer, primary_key=True)
    session_id = Column(
        Integer, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )

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

    # Stores policy letter + avg latency, e.g. "A;avg_latency=0.25"
    feedback = Column(Text, nullable=True)

    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    parent_session = relationship("SessionModel", back_populates="episodes")
