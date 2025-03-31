import argparse
import uuid
import datetime
import random

from flask import Flask, jsonify, request, render_template

from database.schema import PolicyModel, SessionModel, EpisodeModel
from database.connnection import initialize_database_connection
from logger import logger
from server_config import load_config, ServerConfig
from server_utils import cleanup_stale_sessions, get_gcs_bucket


app = Flask(__name__, template_folder="frontend/templates", static_folder="frontend/static")


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
    evaluation_notes = request.args.get(
        "evaluation_notes", ""
    )  # TODO: evaluation_notes should be passed in at the termination of a session, not at the beginning (for now we can assume it's empty)

    db = SessionLocal()

    # 1) Clean up stale sessions
    cleanup_stale_sessions(database_session=db, range_in_hours=ServerSetting.eval_session.timeout_hours)

    try:
        num_policies_needed: int = 2 if eval_type == "A/B" else 1

        # Query for candidate policies
        candidates = (
            db.query(PolicyModel)
            .filter(PolicyModel.is_in_use == False, PolicyModel.ip_address.isnot(None), PolicyModel.port.isnot(None))
            .all()
        )

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
                if p.times_in_ab_eval is None:
                    p.times_in_ab_eval = 0
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
            policyB_name=policyB_name,
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
                "port": chosen[0].port,
            },
        }
        if eval_type == "A/B":
            resp_data["policyB"] = {
                "policy_name": chosen[1].unique_policy_name,
                "ip": chosen[1].ip_address,
                "port": chosen[1].port,
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

    def upload_file_if_present(file_key: str, extension: str) -> str | None:
        """
        Helper: if there's an uploaded file in request.files[file_key],
        upload to GCS, return the path. Otherwise, return None.
        """
        f = request.files.get(file_key, None)
        if not f:
            return None

        gcs_path = f"evaluation_data/{session_id}/{policy_name}_{timestamp_str}_{file_key}.{extension}"
        blob = BlobStorage.blob(gcs_path)
        blob.upload_from_file(f)
        return gcs_path

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
            feedback=feedback,
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
    form_session_id = request.form.get("session_id") or request.json.get("session_id")  # type: ignore
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

        return jsonify({"status": "terminated", "session_id": form_session_id}), 200
    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


@app.route("/leaderboard", methods=["GET"])
def leaderboard():
    """
    Displays the leaderboard of policies, ordered by `elo_score` in descending order.
    """
    db = SessionLocal()
    try:
        policies = db.query(PolicyModel).order_by(PolicyModel.elo_score.desc()).all()
        return render_template("leaderboard.html", policies=policies)
    except Exception as e:
        return f"An error occurred loading the leaderboard: {str(e)}", 500
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the central evaluation server.")
    parser.add_argument("config_path", type=str, help="Path to the evaluation config YAML file")
    args = parser.parse_args()

    # Load the server configurations
    ServerSetting: ServerConfig = load_config(args.config_path)
    logger.info(f"Server configuration loaded: {ServerSetting}")

    # Initialize the database connection
    SessionLocal = initialize_database_connection(ServerSetting.database_url)
    logger.info(f"Database connection to {ServerSetting.database_url} initialized.")

    # The GCS bucket used to store episode videos and npz files
    BlobStorage = get_gcs_bucket(ServerSetting.gcs_bucket_name)

    # Run the Flask app
    app.run(host=ServerSetting.host, port=ServerSetting.port, debug=ServerSetting.debug_mode)
