from fastapi import FastAPI, Body
from typing import Dict, Any, Optional
import uuid
import random
import datetime

from .db import DatabaseClient
from .elo import update_elo_ratings

app = FastAPI(title="Central Scheduling Server")
db_client = DatabaseClient()

# In-memory lock/tracking for "reserved" policies
reserved_policies = {}

@app.post("/register_user")
def register_user(email: str, name: str):
    user_id = db_client.register_user(email, name)
    return {"user_id": user_id}

@app.post("/register_policy")
def register_policy(policy_addr: str, eval_environment: str, user_id: str):
    policy_id = db_client.register_policy(policy_addr, eval_environment, user_id)
    return {"policy_id": policy_id}

@app.get("/sample_eval")
def sample_eval(eval_station: str, eval_user_id: str):
    """
    Called by an eval station to request two policies to A/B test.
    Return: two policy server addresses and a unique eval_id
    """
    # In a real system, implement your advanced scheduling logic:
    # - policies with fewer evals or uncertain ELO
    # - policies the user has credits to test
    # - tasks for max info gain
    # ...
    policies = db_client.list_policies()

    # Filter out currently reserved policies:
    available_policies = [
        p_id for p_id, p_data in policies.items() 
        if p_data.get("active", True) and p_id not in reserved_policies
    ]

    # For demonstration, randomly pick two distinct policies if available
    if len(available_policies) < 2:
        return {
            "status": "error",
            "message": "Not enough available policies to sample."
        }

    chosen_two = random.sample(available_policies, 2)
    # Reserve them
    for p_id in chosen_two:
        reserved_policies[p_id] = {
            "reserved_at": datetime.datetime.now(),
            "eval_station": eval_station
        }

    # Create a unique eval_id
    eval_id = str(uuid.uuid4())

    # In a real system, you'd also send a "load" signal to policy servers
    # For demonstration, we just return the addresses
    policy_addrs = [policies[p]["policy_addr"] for p in chosen_two]

    return {
        "eval_id": eval_id,
        "policy_ids": chosen_two,
        "policy_addrs": policy_addrs
    }

@app.post("/upload_eval_results")
def upload_eval_results(eval_id: str, eval_data: Dict[str, Any] = Body(...)):
    """
    Called by eval station to register results with the central server
    """
    # Example eval_data might contain:
    # {
    #   "policyA_id": "...", "policyB_id": "...",
    #   "task_label": "...",
    #   "successA": true, "successB": false,
    #   "scoreA": 1, "scoreB": 0,
    #   "feedback": "Policy 1 did better because..."
    #   ...
    # }
    db_client.store_eval_result(eval_id, eval_data)

    # Mark policies free again
    policyA_id = eval_data.get("policyA_id")
    policyB_id = eval_data.get("policyB_id")
    if policyA_id in reserved_policies:
        del reserved_policies[policyA_id]
    if policyB_id in reserved_policies:
        del reserved_policies[policyB_id]

    # ELO update example
    if "scoreA" in eval_data and "scoreB" in eval_data:
        # Typically, you only need one score, e.g. scoreA in {0,0.5,1}
        # We'll assume scoreB = 1 - scoreA in a standard match scenario
        eloA = db_client.get_elo(policyA_id)
        eloB = db_client.get_elo(policyB_id)
        newA, newB = update_elo_ratings(eloA, eloB, eval_data["scoreA"])
        db_client.update_elo(policyA_id, newA)
        db_client.update_elo(policyB_id, newB)

    return {"status": "success", "message": f"Eval {eval_id} results uploaded."}

@app.post("/update_stats")
def update_stats():
    """
    Could recalculate all ELOs or broadcast periodic stats to policy owners
    For now, does nothing in this skeleton.
    """
    return {"status": "stats updated"}

@app.get("/leaderboard")
def leaderboard():
    """
    Return a leaderboard sorted by ELO.
    """
    policies = db_client.list_policies()
    elos = [(p_id, db_client.get_elo(p_id)) for p_id in policies]
    # sort descending by ELO
    elos_sorted = sorted(elos, key=lambda x: x[1], reverse=True)
    return {"leaderboard": elos_sorted}
