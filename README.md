# real_eval

To run the scheduling server:
```
uvicorn robot_eval_framework.scheduling_server:app --host 0.0.0.0 --port 8000
```

To run a policy server:
```
uvicorn robot_eval_framework.policy_server:app --host 0.0.0.0 --port 8100
```

To run the eval station:
```
python -m robot_eval_framework.eval_station
```