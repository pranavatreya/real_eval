import uuid
import datetime
import random
import requests  # Keep for any future expansions, though we removed health checks
from flask import Flask, request, jsonify
from google.cloud import storage

from database.schema import PolicyModel, SessionModel, EpisodeModel
from database.connection import initialize_database_connection
from logger import logger


#  Flask App Setup
app = Flask(__name__)

BUCKET_NAME = "distributed_robot_eval"
BUCKET_PREFIX = "evaluation_data"

def get_gcs_client():
    return storage.Client()

# -- Utility / Cleanup: handle stale sessions --

SESSION_TIMEOUT_HOURS = 0.5

def cleanup_stale_sessions():
    db = SessionLocal()
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=SESSION_TIMEOUT_HOURS)
    try:
        stale_sessions = db.query(SessionModel).filter(
            SessionModel.session_completion_timestamp.is_(None),
            SessionModel.session_creation_timestamp < cutoff
        ).all()

        for sess in stale_sessions:
            sess.session_completion_timestamp = datetime.datetime.utcnow()
            if sess.evaluation_notes:
                sess.evaluation_notes += "\nTIMED_OUT"
            else:
                sess.evaluation_notes = "TIMED_OUT"

        if stale_sessions:
            db.commit()
    finally:
        db.close()


# -------------------------------
# Version-check endpoint
# -------------------------------
SERVER_VERSION = "1.1"  # Make sure to bump this up whenever you update the server/client code

@app.route("/version_check", methods=["POST"])
def version_check():
    """
    The client will send e.g. JSON: {"client_version": "1.0"}
    We compare with SERVER_VERSION. If mismatch, return an error.
    """
    data = request.get_json() or {}
    client_version = data.get("client_version", None)
    if client_version == SERVER_VERSION:
        return jsonify({"status": "ok", "message": "Versions match"}), 200
    else:
        return jsonify({"status": "error", "message": "Version mismatch"}), 400
# -------------------------------


@app.route("/get_policies_to_compare", methods=["GET"])
def get_policies_to_compare():
    """
    (1) Now, instead of returning only 4 policies, we return *all* policies
    that have non-null IP/port.
    (2) We still create a new session with policyA_name / policyB_name set to the first two,
        if at least 2 exist. If fewer than 2 exist, we still error.
    """

    # Clean up stale sessions
    cleanup_stale_sessions()

    evaluation_location = request.args.get("eval_location", "")
    evaluator_name = request.args.get("evaluator_name", "")
    robot_name = request.args.get("robot_name", "DROID")

    db = SessionLocal()
    try:
        # Grab all candidate policies that have an IP/port
        candidates = db.query(PolicyModel).filter(
            PolicyModel.ip_address.isnot(None),
            PolicyModel.port.isnot(None),
        ).all()

        random.shuffle(candidates)

        if len(candidates) < 2:
            return jsonify({"error": "We need at least 2 policies to do an A/B session."}), 400

        # Increment times_in_ab_eval for each
        for pol in candidates:
            if pol.times_in_ab_eval is None:
                pol.times_in_ab_eval = 0
            pol.times_in_ab_eval += 1

        session_uuid_ = uuid.uuid4()
        # policyA_name is the first, policyB_name is the second
        # We'll do that by default
        policyA_name = candidates[0].unique_policy_name
        policyB_name = candidates[1].unique_policy_name

        new_session = SessionModel(
            session_uuid=session_uuid_,
            evaluation_type="A/B",
            evaluation_location=evaluation_location,
            evaluator_name=evaluator_name,
            robot_name=robot_name,
            policyA_name=policyA_name,
            policyB_name=policyB_name
        )
        db.add(new_session)
        db.commit()

        # Return *all* policies
        resp_data = {
            "session_id": str(session_uuid_),
            "evaluation_type": "A/B",
            "policies": []
        }

        labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for i, pol in enumerate(candidates):
            label = labels[i] if i < len(labels) else f"X{i}"
            resp_data["policies"].append({
                "label": label,
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
    if not request.form:
        return jsonify({"error": "Must send data in multipart form-data."}), 400

    session_id = request.form.get("session_id", None)
    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400

    policy_name = request.form.get("policy_name", "")
    command = request.form.get("command", "")
    binary_success_str = request.form.get("binary_success", None)
    partial_success_str = request.form.get("partial_success", None)
    duration_str = request.form.get("duration", None)
    policy_ip = request.form.get("policy_ip", None)
    policy_port_str = request.form.get("policy_port", None)
    camera_type = request.form.get("third_person_camera_type", None)
    camera_id_str = request.form.get("third_person_camera_id", None)
    # We'll store policy_letter plus average latency inside the same text
    # E.g. "B;avg_latency=0.123"
    policy_letter_and_latency = request.form.get("policy_letter", "")  
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
        session_obj = db.query(SessionModel).filter_by(session_uuid=session_id).first()
        if not session_obj:
            return jsonify({"error": f"No session with ID {session_id}"}), 400

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
            feedback=policy_letter_and_latency  # Store the combined policy letter + average latency
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
    form_data = request.form if request.form else request.json
    if not form_data:
        return jsonify({"error": "Missing form or JSON data."}), 400

    form_session_id = form_data.get("session_id")
    if not form_session_id:
        return jsonify({"error": "Missing session_id"}), 400

    final_notes = form_data.get("evaluation_notes", "")

    db = SessionLocal()
    try:
        session_obj = db.query(SessionModel).filter_by(session_uuid=form_session_id).first()
        if not session_obj:
            return jsonify({"error": f"No session with ID {form_session_id}"}), 404

        session_obj.session_completion_timestamp = datetime.datetime.utcnow()
        session_obj.evaluation_notes = final_notes

        used_policy_names = []
        if session_obj.policyA_name:
            used_policy_names.append(session_obj.policyA_name)
        if session_obj.policyB_name:
            used_policy_names.append(session_obj.policyB_name)

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
    # 1) Initialize the database connection
    database_url: str = "postgresql://centralserver:m3lxcf830x20g4@localhost:5432/real_eval"
    SessionLocal = initialize_database_connection(database_url)
    logger.info(f"Database connection to {database_url} initialized.")

    app.run(host="0.0.0.0", port=5000, debug=True)
