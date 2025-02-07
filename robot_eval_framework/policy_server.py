from fastapi import FastAPI, Body
from typing import Any, Dict

app = FastAPI(title="Policy Server")

# In-memory representation of the currently loaded policy.
current_policy = {
    "policy_id": None,
    "params": None
}

@app.get("/test")
def test_server():
    return {"status": "ok"}

@app.post("/load")
def load_policy(policy_id: str, policy_params: Dict[str, Any] = Body(...)):
    """
    Load policy parameters onto this server.
    """
    current_policy["policy_id"] = policy_id
    current_policy["params"] = policy_params
    return {"status": "loaded", "policy_id": policy_id}

@app.post("/query")
def query_policy(observation: Dict[str, Any] = Body(...)):
    """
    Query the currently loaded policy with an observation
    and return an action (dummy for this example).
    """
    # In a real system, run your neural network inference here
    # For demonstration, return a random or trivial action
    if current_policy["policy_id"] is None:
        return {"error": "No policy loaded"}
    return {
        "action": "dummy_action",
        "policy_id": current_policy["policy_id"]
    }
