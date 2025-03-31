from dataclasses import dataclass

import yaml


@dataclass(frozen=True)
class EvalSessionSetting:
    timeout_hours: float


@dataclass(frozen=True)
class ServerConfig:
    host: str
    port: int
    database_url: str
    gcs_bucket_name: str
    debug_mode: bool
    eval_session: EvalSessionSetting


def load_config(config_file_path: str) -> ServerConfig:
    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)

    server_settings: dict = config["server"]
    eval_session_settings: dict = config["eval_session"]

    return ServerConfig(
        host=server_settings["host"],
        port=server_settings["port"],
        database_url=server_settings["database_url"],
        gcs_bucket_name=server_settings["gcs_bucket_name"],
        debug_mode=server_settings["debug_mode"],
        eval_session=EvalSessionSetting(timeout_hours=eval_session_settings["timeout_hours"]),
    )
