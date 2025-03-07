# Distributed Real Robot Evaluation Benchmark

As of now, there are three components that need to be launched independently:
1. The robot-side evaluation client `evaluation_client/main.py`
2. The policy server (could be any server, right now using openpi_droid_dct)
3. Central scheduling server `central_server/central_server.py`


### Setup

```shell
pip install -r requirements.txt
```

### Testing evaluation client at Stanford
The policy server for openpi_droid_fast and the central server are running on my end, so as a first step we can try just running the robot evaluation client at Stanford and seeing if things run smoothly.

To launch the eval client:
1. Update the camera IDs in `evaluation_client/main.py` wherever it says `modify camera ID to match setup`
2. Run `python evaluation_client/main.py --left_image/right_image` selecting one of either `left_image` or `right_image` for the 3rd person camera