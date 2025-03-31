# Distributed Real Robot Evaluation Benchmark

As of now, there are three components that need to be launched independently:
1. The robot-side evaluation client `evaluation_client/main.py`
2. The policy server (could be any server, right now using openpi_droid_dct)
3. Central scheduling server `central_server/serve.py`

## Setup

### Install DROID and it's dependencies:

From the root of your [DROID](https://github.com/droid-dataset/droid) checkout, run the following:

```shell
pip install -e .
```

### Install RealEval specific dependencies

From the root directory of the repository, run the following:

```shell
pip install -r requirements.txt
```

## Running the evaluation

To run the evaluation:
1. Add an evaluation config following the other config files in `configs/`
2. Run `python evaluation_client/main.py <path to config file>`. For example, `python evaluation_client/main.py configs/berkeley.yaml`

## Central server deployment

Update the `configs/central_server.yaml` file with the appropriate configuration. Then run the following command to start the server:

```shell
python3 central_server/serve.py configs/central_server.yaml
```

## Contributing

Please ensure to run `./pre-commit.sh` before pushing changes to `main`.

TODO: Add pre-commit checks to the CI/GHA pipeline.
