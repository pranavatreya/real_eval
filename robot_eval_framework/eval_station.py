import requests
import random
import time
from typing import Dict, Any

SCHEDULER_URL = "http://localhost:8000"
EVAL_STATION_ID = "robot_arm_east_coast"
EVAL_USER_ID = "robot_operator_1"

def run_ab_evaluation():
    """
    1) Request two policies from the scheduler.
    2) For each policy, call the policy server's /query to get an action.
    3) Simulate real-world success/failure.
    4) Upload results.
    """
    # Step 1: sample_eval
    resp = requests.get(
        f"{SCHEDULER_URL}/sample_eval",
        params={"eval_station": EVAL_STATION_ID, "eval_user_id": EVAL_USER_ID}
    )
    data = resp.json()
    if data.get("status") == "error":
        print("Could not retrieve policies:", data)
        return

    eval_id = data["eval_id"]
    policy_ids = data["policy_ids"]
    policy_addrs = data["policy_addrs"]

    # Step 2: do an A/B test. We simulate with random success.
    # In a real system, you'd command the robot, gather success/failure, etc.
    successA = random.choice([True, False])
    successB = random.choice([True, False])

    # Step 3: Construct an example result payload
    # We'll say if successA == successB, it's a tie (scoreA=0.5). Otherwise 1 or 0.
    if successA and not successB:
        scoreA, scoreB = 1, 0
    elif not successA and successB:
        scoreA, scoreB = 0, 1
    else:
        scoreA, scoreB = 0.5, 0.5

    # Additional feedback, etc.
    feedback = (
        "Policy A succeeded, B failed" if successA and not successB else
        "Policy B succeeded, A failed" if successB and not successA else
        "Tie or both had the same outcome"
    )

    eval_data = {
        "policyA_id": policy_ids[0],
        "policyB_id": policy_ids[1],
        "task_label": "Pick and place test",
        "successA": successA,
        "successB": successB,
        "scoreA": scoreA,
        "scoreB": scoreB,
        "feedback": feedback
    }

    # Step 4: upload_eval_results
    upload_resp = requests.post(
        f"{SCHEDULER_URL}/upload_eval_results",
        json={
            "eval_id": eval_id,
            "eval_data": eval_data
        }
    )
    print("Upload response:", upload_resp.json())

if __name__ == "__main__":
    # Example usage: continuously request evaluations
    for _ in range(5):
        run_ab_evaluation()
        time.sleep(2)
