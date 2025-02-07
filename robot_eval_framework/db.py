import os
import datetime
from typing import Dict, Any

# If using Firestore:
# from google.cloud import firestore

class DatabaseClient:
    def __init__(self, project_id: str = "YOUR_GCP_PROJECT_ID"):
        # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/your/service_account.json"
        # self.client = firestore.Client(project=project_id)
        self.project_id = project_id
        self.mock_storage = {
            "evals": [],
            "policies": {},
            "users": {},
            "elo_scores": {}
        }

    def register_user(self, email: str, name: str) -> str:
        """Registers a new user in the database and returns user_id."""
        user_id = f"user_{len(self.mock_storage['users']) + 1}"
        self.mock_storage["users"][user_id] = {
            "email": email,
            "name": name,
            "credits": 0
        }
        return user_id

    def register_policy(self, policy_addr: str, eval_env: str, user_id: str) -> str:
        """Registers a policy and returns policy_id."""
        policy_id = f"policy_{len(self.mock_storage['policies']) + 1}"
        self.mock_storage["policies"][policy_id] = {
            "policy_addr": policy_addr,
            "eval_env": eval_env,
            "owner": user_id,
            "active": True
        }
        return policy_id

    def store_eval_result(self, eval_id: str, eval_data: Dict[str, Any]) -> None:
        """Stores an evaluation result in the database."""
        # In Firestore, you might do something like:
        # doc_ref = self.client.collection('evals').document(eval_id)
        # doc_ref.set(eval_data, merge=True)

        # For demonstration, store in a simple list/dict:
        self.mock_storage["evals"].append({eval_id: eval_data})

    def update_elo(self, policy_id: str, new_elo: float) -> None:
        self.mock_storage["elo_scores"][policy_id] = new_elo

    def get_elo(self, policy_id: str) -> float:
        return self.mock_storage["elo_scores"].get(policy_id, 1500.0)  # Default ELO

    def get_policy(self, policy_id: str) -> Dict[str, Any]:
        return self.mock_storage["policies"].get(policy_id, {})

    def list_policies(self):
        return self.mock_storage["policies"]

    def list_evals(self):
        return self.mock_storage["evals"]

    def list_users(self):
        return self.mock_storage["users"]
