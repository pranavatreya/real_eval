import uuid
import datetime
import random
import json
import requests  # Keep for any future expansions, though we removed health checks
from flask import Flask, request, jsonify
from google.cloud import storage

# SQLAlchemy imports
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, ForeignKey,
    Text, Boolean
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

# 1) SQLAlchemy Setup
DB_URL = "postgresql://centralserver:m3lxcf830x20g4@localhost:5432/real_eval"
engine = create_engine(DB_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
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

    # We will always set this to "A/B"
    evaluation_type = Column(String, nullable=False)

    evaluation_location = Column(String, nullable=True)
    evaluator_name = Column(String, nullable=True)
    robot_name = Column(String, nullable=True)

    session_creation_timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    session_completion_timestamp = Column(DateTime, nullable=True)

    # We are re-purposing evaluation_notes to store:
    #   - "VALID_SESSION: " prefix if user said everything was good
    #   - "PREFERENCE=A,B,or tie" for the user preference
    #   - any textual feedback they typed
    #   - "TIMED_OUT" or any other reason for invalid session
    evaluation_notes = Column(Text, nullable=True)

    # We only store two policies in the table itself (A and B) – even though we may evaluate more than two in a session.
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

    # We are now re-purposing `feedback` to store the policy letter (A, B, C, D, etc.).
    feedback = Column(Text, nullable=True)

    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    parent_session = relationship("SessionModel", back_populates="episodes")


# 3) Flask App Setup
app = Flask(__name__)

BUCKET_NAME = "distributed_robot_eval"
BUCKET_PREFIX = "evaluation_data"

def get_gcs_client():
    return storage.Client()

Base.metadata.create_all(engine)

# -- Utility / Cleanup: handle stale sessions --

# Currently set to 0.5 hours. If a session times out, we mark it invalid by storing "TIMED_OUT".
SESSION_TIMEOUT_HOURS = 0.5

def cleanup_stale_sessions():
    """
    Finds sessions that are older than SESSION_TIMEOUT_HOURS but have no
    completion_timestamp set. Mark them as completed, and set evaluation_notes to "TIMED_OUT"
    (thus marking them invalid).
    """
    db = SessionLocal()
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=SESSION_TIMEOUT_HOURS)
    try:
        stale_sessions = db.query(SessionModel).filter(
            SessionModel.session_completion_timestamp.is_(None),
            SessionModel.session_creation_timestamp < cutoff
        ).all()

        for sess in stale_sessions:
            sess.session_completion_timestamp = datetime.datetime.utcnow()
            # Mark it invalid by adding a note
            if sess.evaluation_notes:
                sess.evaluation_notes += "\nTIMED_OUT"
            else:
                sess.evaluation_notes = "TIMED_OUT"

        if stale_sessions:
            db.commit()
    finally:
        db.close()


@app.route("/get_policies_to_compare", methods=["GET"])
def get_policies_to_compare():
    """
    Returns JSON specifying up to 'num_policies_needed' policy IPs for an A/B style evaluation.
    We do 4 by default, labeling them A, B, C, D. The first two (A and B) also get stored
    in the session table.
    """

    # 1) Clean up stale sessions
    cleanup_stale_sessions()

    # 2) We'll collect some user info from query args so that it can be stored in SessionModel
    evaluation_location = request.args.get("eval_location", "")
    evaluator_name = request.args.get("evaluator_name", "")
    robot_name = request.args.get("robot_name", "DROID")

    num_policies_needed = 4  # can be changed to 5 if we want more

    db = SessionLocal()
    try:
        # Grab all candidate policies that have an IP/port (we no longer filter by is_in_use).
        candidates = db.query(PolicyModel).filter(
            PolicyModel.ip_address.isnot(None),
            PolicyModel.port.isnot(None),
        ).all()

        random.shuffle(candidates)

        chosen = []
        for pol in candidates:
            # We remove is_policy_server_alive calls because we assume servers stay up.
            # If the server is not healthy, that is beyond the scope here.
            chosen.append(pol)
            if len(chosen) == num_policies_needed:
                break

        if len(chosen) < num_policies_needed:
            return jsonify({"error": f"Not enough available policies to retrieve {num_policies_needed}."}), 400

        # For each chosen policy, increment times_in_ab_eval (since we are always doing A/B evaluations)
        for p in chosen:
            if p.times_in_ab_eval is None:
                p.times_in_ab_eval = 0
            p.times_in_ab_eval += 1

        # 3) Create the new session in DB
        session_uuid_ = uuid.uuid4()
        # We only store the first two in the session itself
        policyA_name = chosen[0].unique_policy_name
        policyB_name = chosen[1].unique_policy_name

        new_session = SessionModel(
            session_uuid=session_uuid_,
            evaluation_type="A/B",  # Always A/B
            evaluation_location=evaluation_location,
            evaluator_name=evaluator_name,
            robot_name=robot_name,
            policyA_name=policyA_name,
            policyB_name=policyB_name
            # We do NOT store C and D in the session table
        )
        db.add(new_session)
        db.commit()

        labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]  # could be extended
        resp_data = {
            "session_id": str(session_uuid_),
            "evaluation_type": "A/B",
            "policies": []
        }
        for i in range(num_policies_needed):
            pol = chosen[i]
            resp_data["policies"].append({
                "label": labels[i],
                "policy_name": pol.unique_policy_name,
                "ip": pol.ip_address,
                "port": pol.port
            })

        return jsonify(resp_data), 200

    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


@app.route("/upload_eval_data", methods=["POST"])
def upload_eval_data():
    """
    Receives a single evaluation episode’s data.
    Uploads any video files + 1 npz to GCS, then stores the metadata in EpisodeModel.
    We also store the policy letter in the "feedback" field now.
    """
    if not request.form:
        return jsonify({"error": "Must send data in multipart form-data."}), 400

    session_id = request.form.get("session_id", None)
    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400

    # Retrieve text fields
    policy_name = request.form.get("policy_name", "")
    command = request.form.get("command", "")
    binary_success_str = request.form.get("binary_success", None)
    partial_success_str = request.form.get("partial_success", None)
    duration_str = request.form.get("duration", None)
    policy_ip = request.form.get("policy_ip", None)
    policy_port_str = request.form.get("policy_port", None)
    camera_type = request.form.get("third_person_camera_type", None)
    camera_id_str = request.form.get("third_person_camera_id", None)
    policy_letter = request.form.get("policy_letter", "")  # The newly-added field
    timestamp_str = request.form.get("timestamp", "")

    try:
        binary_success = int(binary_success_str) if binary_success_str is not None else None
    except:
        binary_success = None

    try:
        partial_success = float(partial_success_str) if partial_success_str else None
    except:
        partial_success = None

    try:
        duration = int(duration_str) if duration_str else None
    except:
        duration = None

    try:
        policy_port = int(policy_port_str) if policy_port_str else None
    except:
        policy_port = None

    try:
        third_person_camera_id = int(camera_id_str) if camera_id_str else None
    except:
        third_person_camera_id = None

    try:
        if timestamp_str:
            timestamp_dt = datetime.datetime.strptime(timestamp_str, "%Y_%m_%d_%H_%M_%S")
        else:
            timestamp_dt = datetime.datetime.utcnow()
    except:
        timestamp_dt = datetime.datetime.utcnow()

    db = SessionLocal()
    try:
        # 1) Check that session exists
        session_obj = db.query(SessionModel).filter_by(session_uuid=session_id).first()
        if not session_obj:
            return jsonify({"error": f"No session with ID {session_id}"}), 400

        # 2) Upload files to GCS
        storage_client = get_gcs_client()
        bucket = storage_client.bucket(BUCKET_NAME)

        def upload_file_if_present(file_key: str, extension: str):
            f = request.files.get(file_key, None)
            if not f:
                return None
            gcs_path = f"{BUCKET_PREFIX}/{session_id}/{policy_name}_{timestamp_str}_{file_key}.{extension}"
            blob = bucket.blob(gcs_path)
            blob.upload_from_file(f)
            return gcs_path

        gcs_left_cam_path = upload_file_if_present("video_left", "mp4")
        gcs_right_cam_path = upload_file_if_present("video_right", "mp4")
        gcs_wrist_cam_path = upload_file_if_present("video_wrist", "mp4")
        npz_file_path = upload_file_if_present("npz_file", "npz")

        # Insert new EpisodeModel row.
        new_episode = EpisodeModel(
            session_id=session_obj.id,
            policy_name=policy_name,
            command=command,
            binary_success=binary_success,
            partial_success=partial_success,
            duration=duration,
            gcs_left_cam_path=gcs_left_cam_path,
            gcs_right_cam_path=gcs_right_cam_path,
            gcs_wrist_cam_path=gcs_wrist_cam_path,
            npz_file_path=npz_file_path,
            timestamp=timestamp_dt,
            policy_ip=policy_ip,
            policy_port=policy_port,
            third_person_camera_type=camera_type,
            third_person_camera_id=third_person_camera_id,
            feedback=policy_letter  # storing the letter here
        )
        db.add(new_episode)
        db.commit()

        return jsonify({"status": "success"}), 200
    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


@app.route("/terminate_session", methods=["POST"])
def terminate_session():
    """
    Called by the evaluation_client when the entire evaluation is done.
      - Mark session completion timestamp
      - Store final evaluation notes in 'evaluation_notes'
      - We do *not* bother with freeing up is_in_use (we no longer use that).
      - Mark last_time_evaluated for the *two* policies
    """
    form_data = request.form if request.form else request.json
    if not form_data:
        return jsonify({"error": "Missing form or JSON data."}), 400

    form_session_id = form_data.get("session_id")
    if not form_session_id:
        return jsonify({"error": "Missing session_id"}), 400

    # We expect "evaluation_notes" from the client describing preference, long feedback, etc.
    final_notes = form_data.get("evaluation_notes", "")

    db = SessionLocal()
    try:
        session_obj = db.query(SessionModel).filter_by(session_uuid=form_session_id).first()
        if not session_obj:
            return jsonify({"error": f"No session with ID {form_session_id}"}), 404

        # Mark the session complete
        session_obj.session_completion_timestamp = datetime.datetime.utcnow()

        # Append the final notes
        session_obj.evaluation_notes = final_notes

        # We can update last_time_evaluated for policyA and policyB if we like
        used_policy_names = []
        if session_obj.policyA_name:
            used_policy_names.append(session_obj.policyA_name)
        if session_obj.policyB_name:
            used_policy_names.append(session_obj.policyB_name)
        # We do *not* track policyC_name, policyD_name in the SessionModel; they are ephemeral.
        # However information about them is stored in the episodes table, and is linked by the same FK session number

        for pname in used_policy_names:
            pol = db.query(PolicyModel).filter_by(unique_policy_name=pname).first()
            if pol:
                pol.last_time_evaluated = datetime.datetime.utcnow()

        db.commit()
        return jsonify({
            "status": "terminated",
            "session_id": form_session_id
        }), 200

    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

