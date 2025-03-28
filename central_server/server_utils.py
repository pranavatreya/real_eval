import datetime
import requests

from google.cloud import storage

from database.schema import PolicyModel, SessionModel


def cleanup_stale_sessions(database_session, range_in_hours: float = 4):
    """
    Finds sessions that are older than SESSION_TIMEOUT_HOURS but have no
    completion_timestamp set. Mark them as completed, and free up policies if needed.
    """
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=range_in_hours)
    try:
        stale_sessions = database_session.query(SessionModel).filter(
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
                policy_a = database_session.query(PolicyModel).filter_by(unique_policy_name=sess.policyA_name).first()
                if policy_a and policy_a.is_in_use:
                    policy_a.is_in_use = False
            if sess.policyB_name:
                policy_b = database_session.query(PolicyModel).filter_by(unique_policy_name=sess.policyB_name).first()
                if policy_b and policy_b.is_in_use:
                    policy_b.is_in_use = False

        if stale_sessions:
            database_session.commit()
    finally:
        database_session.close()


def is_policy_server_alive(ip_address, port, timeout=3.0):  # TODO: make this function call the correct endpoint
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
    except Exception as e:
        return False


def get_gcs_client():   # TODO: create a wrapper class
    return storage.Client()

