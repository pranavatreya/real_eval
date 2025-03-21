import uuid
import datetime
import json
import random
import requests

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

# 2) Define ORM Models

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
    evaluation_type = Column(String, nullable=False)        # "A/B" or "single-policy"
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
    episodes = relationship(
        "EpisodeModel", back_populates="parent_session", cascade="all, delete-orphan"
    )

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
    partial_success = Column(Float, nullable=True)   # 0.0 to 1.0

    duration = Column(Integer, nullable=True)        # number of timesteps

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


# 3) Flask App Setup
app = Flask(__name__)

BUCKET_NAME = "distributed_robot_eval"
BUCKET_PREFIX = "evaluation_data"

def get_gcs_client():
    return storage.Client()

Base.metadata.create_all(engine)

# -- Utility / Cleanup: handle stale sessions --

SESSION_TIMEOUT_HOURS = 4  # Example: after 4 hours, we mark an un-terminated session as "done" and free any policies

def cleanup_stale_sessions():
    """
    Finds sessions that are older than SESSION_TIMEOUT_HOURS but have no
    completion_timestamp set. Mark them as completed, and free up policies if needed.
    """
    db = SessionLocal()
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=SESSION_TIMEOUT_HOURS)
    try:
        stale_sessions = db.query(SessionModel).filter(
            SessionModel.session_completion_timestamp.is_(None),
            SessionModel.session_creation_timestamp < cutoff
        ).all()

        for sess in stale_sessions:
            # Mark session as completed now
            sess.session_completion_timestamp = datetime.datetime.utcnow()

            # Free up policies
            # If policyA_name or policyB_name are not None, set them is_in_use=False
            # in the PolicyModel
            if sess.policyA_name:
                polA = db.query(PolicyModel).filter_by(unique_policy_name=sess.policyA_name).first()
                if polA and polA.is_in_use:
                    polA.is_in_use = False
            if sess.policyB_name:
                polB = db.query(PolicyModel).filter_by(unique_policy_name=sess.policyB_name).first()
                if polB and polB.is_in_use:
                    polB.is_in_use = False

        if stale_sessions:
            db.commit()
    finally:
        db.close()

def is_policy_server_alive(ip_address, port, timeout=3.0): # TODO: make this function call the correct endpoint
    """
    Try to ping the policy server with a simple GET or other minimal request.
    Returns True if we got a 200, otherwise False.
    """
    if not ip_address or not port:
        return False
    url = f"http://{ip_address}:{port}/ping"  # Or whatever minimal endpoint
    try:
        r = requests.get(url, timeout=timeout)
        return r.ok
    except:
        return False


@app.route("/get_policies_to_compare", methods=["GET"])
def get_policies_to_compare():
    """
    Returns JSON specifying up to two policy IPs for an A/B evaluation session.
    (Could also handle single-policy if "eval_type=single-policy".)
    Steps:
      1) Clean up stale sessions.
      2) Randomly pick the required # of policies from the policy table, skipping those in_use or missing IP/port.
      3) For each candidate, do a quick "ping" check. If the server is not up, set IP/port to None in DB and skip it.
      4) If we find enough policies, create a SessionModel row with type "A/B", set them in_use = True, return them.
    """
    # e.g. user can specify in query args:
    eval_type = request.args.get("eval_type", "A/B")  # "A/B" or "single-policy"
    eval_location = request.args.get("eval_location", "Berkeley")
    evaluator_name = request.args.get("evaluator_name", "John Doe")
    robot_name = request.args.get("robot_name", "DROID")
    evaluation_notes = request.args.get("evaluation_notes", "") # TODO: evaluation_notes should be passed in at the termination of a session, not at the beginning (for now we can assume it's empty)

    # 1) Clean up stale sessions
    cleanup_stale_sessions()

    db = SessionLocal()
    try:
        num_policies_needed = 2 if eval_type == "A/B" else 1

        # Query for candidate policies
        candidates = db.query(PolicyModel).filter(
            PolicyModel.is_in_use == False,
            PolicyModel.ip_address.isnot(None),
            PolicyModel.port.isnot(None)
        ).all()

        random.shuffle(candidates)

        chosen = []
        for pol in candidates:
            # TODO: uncomment once we have a ping endpoint
            # # Check if policy server is up
            # alive = is_policy_server_alive(pol.ip_address, pol.port)
            # if not alive:
            #     # Mark IP/port = None => policy no longer valid for picks
            #     pol.ip_address = None
            #     pol.port = None
            #     db.commit()
            #     continue

            # If alive, we pick it
            chosen.append(pol)
            if len(chosen) == num_policies_needed:
                break

        if len(chosen) < num_policies_needed:
            return jsonify({"error": f"Not enough available policies for {eval_type} evaluation."}), 400

        # Mark chosen policies as in use
        for p in chosen:
            p.is_in_use = True
            # If it's A/B, we also increment times_in_ab_eval
            if eval_type == "A/B":
                p.times_in_ab_eval += 1

        # Create the new session in DB
        session_uuid_ = uuid.uuid4()
        if eval_type == "A/B":
            policyA_name = chosen[0].unique_policy_name
            policyB_name = chosen[1].unique_policy_name
        else:
            policyA_name = chosen[0].unique_policy_name
            policyB_name = None

        new_session = SessionModel(
            session_uuid=session_uuid_,
            evaluation_type=eval_type,
            evaluation_location=eval_location,
            evaluator_name=evaluator_name,
            robot_name=robot_name,
            evaluation_notes=evaluation_notes,
            policyA_name=policyA_name,
            policyB_name=policyB_name
        )
        db.add(new_session)
        db.commit()

        # Return the IP/ports to the client
        resp_data = {
            "session_id": str(session_uuid_),
            "evaluation_type": eval_type,
            "policyA": {
                "policy_name": chosen[0].unique_policy_name,
                "ip": chosen[0].ip_address,
                "port": chosen[0].port
            }
        }
        if eval_type == "A/B":
            resp_data["policyB"] = {
                "policy_name": chosen[1].unique_policy_name,
                "ip": chosen[1].ip_address,
                "port": chosen[1].port
            }

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
    Uploads up to 3 video streams + 1 npz to GCS, then stores the metadata in EpisodeModel.

    Expected form fields:
      session_id, policy_name, command,
      binary_success, partial_success,
      duration, policy_ip, policy_port,
      third_person_camera_type, third_person_camera_id,
      feedback, [plus video_left, video_right, video_wrist, npz_file]
    """
    # We can accept either form-data or JSON, but videos typically come in form-data
    if not request.form:
        return jsonify({"error": "Must send data in form format."}), 400

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
    feedback = request.form.get("feedback", "")
    timestamp_str = request.form.get("timestamp", "")

    # Convert numeric fields
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

    # Convert timestamp if provided
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

        # 2) Upload up to 3 videos + 1 npz
        #    We'll store them as separate keys in the GCS bucket:
        #    e.g. evaluation_data/<session_uuid>/<policy_name>_<timestamp>_left.mp4
        #         evaluation_data/<session_uuid>/<policy_name>_<timestamp>_npz.npz
        storage_client = get_gcs_client()
        bucket = storage_client.bucket(BUCKET_NAME)

        def upload_file_if_present(file_key: str, extension: str):
            """
            Helper: if there's an uploaded file in request.files[file_key],
            upload to GCS, return the path. Otherwise return None.
            """
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

        # 3) Insert new EpisodeModel row
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
            feedback=feedback
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
    Called by the evaluation_client when the entire A/B (or single-policy) evaluation is done.
      - Mark session completion timestamp
      - Free up the policies used (set is_in_use=False, update last_time_evaluated)
    """
    form_session_id = request.form.get("session_id") or request.json.get("session_id")
    if not form_session_id:
        return jsonify({"error": "Missing session_id"}), 400

    db = SessionLocal()
    try:
        session_obj = db.query(SessionModel).filter_by(session_uuid=form_session_id).first()
        if not session_obj:
            return jsonify({"error": f"No session with ID {form_session_id}"}), 404

        # Mark the session complete
        session_obj.session_completion_timestamp = datetime.datetime.utcnow()
        db.commit()

        # Free up policies used by this session
        # For A/B session, there might be two policies. For single-policy, maybe just one in policyA_name
        used_policy_names = []
        if session_obj.policyA_name:
            used_policy_names.append(session_obj.policyA_name)
        if session_obj.policyB_name:
            used_policy_names.append(session_obj.policyB_name)

        for pname in used_policy_names:
            pol = db.query(PolicyModel).filter_by(unique_policy_name=pname).first()
            if pol and pol.is_in_use:
                pol.is_in_use = False
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
    # Run the Flask app in debug mode for development
    app.run(host="0.0.0.0", port=5000, debug=True)
