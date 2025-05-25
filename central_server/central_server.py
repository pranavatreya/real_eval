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

# ---------------------------------------------------------------------------
# 🆕 Canonical-university mapping + helper
# ---------------------------------------------------------------------------
UNI_CANONICAL = {
    "UNIVERSITY OF CALIFORNIA BERKELEY": "Berkeley",
    "UCB": "Berkeley",
    "BERKELEY": "Berkeley",
    "STANFORD": "Stanford",
    "U PENN": "UPenn",
    "UNIVERSITY OF PENNSYLVANIA": "UPenn",
    "UPENN": "UPenn",
    "UNIVERSITY OF WASHINGTON": "UW",
    "UNVERSITY OF WASHINGTON": "UW",
    "UNIVERSITY OF WASHGINTON": "UW",
    "UW": "UW",
    "MILA": "UMontreal",
    "UNIVERSITY MONTREAL": "UMontreal",
    "UOF MONTREAL": "UMontreal",
    "YONSEI": "Yonsei",
    "UT AUSTIN": "UT Austin",
    "UNIVERSITY OF TEXAS AT AUSTIN": "UT Austin",
}
TARGET_UNIS = [
    "Berkeley",
    "Stanford",
    "UW",
    "UPenn",
    "UMontreal",
    "Yonsei",
    "UT Austin",
]


def canonicalize_uni(raw: str | None) -> str:
    """Return canonical university name or 'Other'."""
    if not raw:
        return "Other"
    upper = raw.upper().strip()
    for k, canon in UNI_CANONICAL.items():
        if k in upper:
            return canon
    return "Other"


# ---------------------------------------------------------------------------
# 🆕 Endpoint: list completed/VALID A-B evaluation sessions
# ---------------------------------------------------------------------------
@app.route("/list_ab_evaluations", methods=["GET"])
def list_ab_evaluations():
    """
    Return _all_ completed and VALID A/B-evaluation sessions, **newest first**.
    Only the fields needed by the React UI are included.

    Response JSON schema
    --------------------
    {
        "evaluations": [
            {
                "session_id": str,
                "university": str,
                "completion_time": "2025-05-24T21:14:03.123456Z",
                "preference": "A" | "B" | "TIE" | null,
                "longform_feedback": str | null,
                "language_instruction": str | null,
                "policyA": {
                    "name": str,
                    "partial_success": float | null,
                    "wrist_video_url": str | null,
                    "third_person_video_url": str | null
                },
                "policyB": { …same keys as policyA… }
            },
            …
        ]
    }
    """
    db = SessionLocal()
    try:
        sessions = (
            db.query(SessionModel)
            .filter(
                SessionModel.evaluation_type == "A/B",
                SessionModel.session_completion_timestamp.isnot(None),
                SessionModel.evaluation_notes.isnot(None),
            )
            .order_by(SessionModel.session_completion_timestamp.desc())
            .all()
        )

        out = []
        for s in sessions:
            # Only keep sessions explicitly marked as VALID
            if "VALID_SESSION" not in s.evaluation_notes.upper():
                continue

            # ------------------------------------
            # Parse evaluation_notes
            # ------------------------------------
            pref = None
            feedback = None
            for line in (s.evaluation_notes or "").splitlines():
                line = line.strip()
                if line.upper().startswith("PREFERENCE="):
                    pref = line.split("=", 1)[1].strip().upper()
                elif line.upper().startswith("LONGFORM_FEEDBACK="):
                    feedback = line.split("=", 1)[1].strip()

            # ------------------------------------
            # Fetch episodes for policies A & B
            # ------------------------------------
            episodes = (
                db.query(EpisodeModel)
                .filter(
                    EpisodeModel.session_id == s.id,
                    EpisodeModel.policy_name.in_([s.policyA_name, s.policyB_name]),
                )
                .all()
            )
            ep_map = {ep.policy_name: ep for ep in episodes}
            if s.policyA_name not in ep_map or s.policyB_name not in ep_map:
                # Incomplete data – skip
                continue

            def _policy_block(policy_name: str) -> dict:
                ep = ep_map[policy_name]

                # Pick the preferred third-person camera video
                cam_type = (ep.third_person_camera_type or "").lower()
                if "left" in cam_type and ep.gcs_left_cam_path:
                    third_rel = ep.gcs_left_cam_path
                elif "right" in cam_type and ep.gcs_right_cam_path:
                    third_rel = ep.gcs_right_cam_path
                else:
                    # Fallback if camera_type missing / misspelled
                    third_rel = ep.gcs_left_cam_path or ep.gcs_right_cam_path

                def _url(rel_path: str | None) -> str | None:
                    if not rel_path:
                        return None
                    return f"https://storage.googleapis.com/{BUCKET_NAME}/{rel_path}"

                return {
                    "name": policy_name,
                    "partial_success": ep.partial_success,
                    "wrist_video_url": _url(ep.gcs_wrist_cam_path),
                    "third_person_video_url": _url(third_rel),
                }

            policyA_block = _policy_block(s.policyA_name)
            policyB_block = _policy_block(s.policyB_name)

            # Any episode will do for getting the language command
            lang_instr = ep_map[s.policyA_name].command or ep_map[s.policyB_name].command

            out.append(
                {
                    "session_id": str(s.session_uuid),
                    "university": canonicalize_uni(s.evaluation_location),
                    "completion_time": s.session_completion_timestamp.isoformat() + "Z",
                    "evaluator_name": s.evaluator_name,
                    "preference": pref,
                    "longform_feedback": feedback,
                    "language_instruction": lang_instr,
                    "policyA": policyA_block,
                    "policyB": policyB_block,
                }
            )

        return jsonify({"evaluations": out}), 200

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
