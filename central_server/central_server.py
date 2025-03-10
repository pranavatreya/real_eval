import uuid
import datetime
import json

from flask import Flask, request, jsonify
from google.cloud import storage

# SQLAlchemy imports
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

# 1) SQLAlchemy Setup

# Format: postgresql://username:password@host:port/database
DB_URL = "postgresql://pranav:taranga@localhost:5432/real_eval_devel"

engine = create_engine(DB_URL, echo=False)  # echo=True to see SQL logs
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# 2) Define ORM Models

class SessionModel(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True)
    # We'll store the "session_id" as a UUID in the database:
    session_uuid = Column(PG_UUID(as_uuid=True), unique=True, nullable=False)
    policyA = Column(String, nullable=False)
    policyB = Column(String, nullable=False)
    eval_location = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationship: one Session -> many Episodes
    episodes = relationship("EpisodeModel", back_populates="parent_session", cascade="all, delete-orphan")

class EpisodeModel(Base):
    __tablename__ = "episodes"

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)

    # E.g. "policy_A" or "policy_B"
    policy_name = Column(String, nullable=False)
    command = Column(Text, nullable=True)
    success = Column(Float, nullable=True)
    duration = Column(Integer, nullable=True)  # number of timesteps
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    # GCS path to the uploaded video
    gcs_video_path = Column(String, nullable=True)

    parent_session = relationship("SessionModel", back_populates="episodes")


# 3) Flask App Setup

app = Flask(__name__)

BUCKET_NAME = "rail-tpus-pranav"
BUCKET_PREFIX = "real_eval"

def get_gcs_client():
    """
    Creates and returns a Google Cloud Storage client.
    """
    return storage.Client()

# Ensure DB schema is created, in case it currently isn't
Base.metadata.create_all(engine)

# 4) Endpoints

@app.route("/get_policies_to_compare", methods=["GET"])
def get_policies_to_compare():
    """
    Returns JSON specifying two policy IP addresses for the A/B evaluation.
    Creates a new session in the DB & GCS folder.
    """
    # Create a unique session_id (UUID)
    session_uuid = uuid.uuid4()

    #policyA_ip = "10.103.116.247:8000"
    #policyB_ip = "10.103.116.247:8000"

    policyA_ip = "128.32.175.81:8000"
    policyB_ip = "128.32.175.81:8000"

    # Insert into DB
    db = SessionLocal()
    try:
        new_session = SessionModel(
            session_uuid=session_uuid,
            policyA=policyA_ip,
            policyB=policyB_ip,
            eval_location="berkeley_droid",
        )
        db.add(new_session)
        db.commit()
        db.refresh(new_session)
    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()

    # We can also add metadata to GCS, albeit this is redundant
    try:
        storage_client = get_gcs_client()
        bucket = storage_client.bucket(BUCKET_NAME)
        metadata_blob = bucket.blob(f"{BUCKET_PREFIX}/{session_uuid}/metadata.json")
        metadata_blob.upload_from_string(
            json.dumps({
                "session_id": str(session_uuid),
                "policyA": policyA_ip,
                "policyB": policyB_ip,
                "eval_location": "berkeley_droid",
                "created_at": datetime.datetime.utcnow().isoformat(),
            }),
            content_type="application/json"
        )
    except Exception as e:
        print(f"Warning: Could not upload metadata to GCS: {e}")

    return jsonify({
        "session_id": str(session_uuid),
        "policyA": policyA_ip,
        "policyB": policyB_ip
    })

@app.route("/upload_eval_data", methods=["POST"])
def upload_eval_data():
    """
    Receives a single evaluation episode’s data (video, success, etc.)
    Writes the video to GCS, stores metadata in the DB.
    """
    # We expect session_id in form-data or JSON
    if request.files:
        session_id = request.form.get("session_id", None)
        policy_name = request.form.get("policy_name", None)
        command = request.form.get("command", "")
        success_str = request.form.get("success", "")
        duration_str = request.form.get("duration", "0")
        timestamp_str = request.form.get("timestamp", "")

        # Convert success to float
        try:
            success = float(success_str)
        except:
            success = None

        # Convert duration to int
        try:
            duration = int(duration_str)
        except:
            duration = None

        # Convert timestamp to datetime if needed
        try:
            if timestamp_str:
                # timestamp_str e.g. "2023_03_15_19_22_05"
                # or "2023-03-15T19:22:05"
                # We'll do a naive approach:
                timestamp_dt = datetime.datetime.strptime(timestamp_str, "%Y_%m_%d_%H_%M_%S")
            else:
                timestamp_dt = datetime.datetime.utcnow()
        except:
            timestamp_dt = datetime.datetime.utcnow()

        if not session_id:
            return jsonify({"error": "Missing session_id"}), 400

        # 1) Check the session in DB
        db = SessionLocal()
        try:
            session_obj = db.query(SessionModel).filter_by(session_uuid=session_id).first()
            if not session_obj:
                return jsonify({"error": f"No session with ID {session_id}"}), 400

            # 2) Upload video file to GCS
            video_file = request.files.get("video", None)
            gcs_video_path = None
            if video_file:
                # The path in GCS for this session:
                gcs_path = f"{BUCKET_PREFIX}/{session_id}/{policy_name}_{timestamp_str}.mp4"
                try:
                    storage_client = get_gcs_client()
                    bucket = storage_client.bucket(BUCKET_NAME)
                    blob = bucket.blob(gcs_path)
                    blob.upload_from_file(video_file, content_type="video/mp4")
                    gcs_video_path = gcs_path
                except Exception as e:
                    return jsonify({"error": f"Failed to upload video: {str(e)}"}), 500

            # 3) Insert episode metadata in DB
            new_episode = EpisodeModel(
                session_id=session_obj.id,
                policy_name=policy_name,
                command=command,
                success=success,
                duration=duration,
                timestamp=timestamp_dt,
                gcs_video_path=gcs_video_path
            )
            db.add(new_episode)
            db.commit()
            db.refresh(new_episode)

        except Exception as e:
            db.rollback()
            return jsonify({"error": str(e)}), 500
        finally:
            db.close()

        return jsonify({"status": "success"}), 200

    else:
        return jsonify({"error": "No files or data provided"}), 400

@app.route("/terminate_session", methods=["POST"])
def terminate_session():
    """
    Called by the evaluation_client when the entire A/B evaluation is done.
    """
    form_session_id = request.form.get("session_id") or request.json.get("session_id")
    if not form_session_id:
        return jsonify({"error": "Missing session_id"}), 400

    db = SessionLocal()
    try:
        session_obj = db.query(SessionModel).filter_by(session_uuid=form_session_id).first()
        if not session_obj:
            return jsonify({"error": f"No session with ID {form_session_id}"}), 404

        # If we want to do anything final with the session here, we can do it...

        # For example, we could store a "closed_at" field if added to the schema:
        # session_obj.closed_at = datetime.datetime.utcnow()
        # db.commit()

        # We could also delete the session, but usually this is not done:
        # db.delete(session_obj)
        # db.commit()

    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()

    return jsonify({"status": "terminated", "session_id": form_session_id}), 200

if __name__ == "__main__":
    # Run the Flask app (note: while we're using flask, this server is for development only)
    app.run(host="0.0.0.0", port=5000, debug=True)
