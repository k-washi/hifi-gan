import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv(verbose=True)


@dataclass
class Env:
    aws_access_key_id: str
    aws_region: str


def load_env(env_path: str) -> Env:
    """
    env_pathの.envファイルを優先して、環境変数を呼び出す

    Args:
        env_path (str): .envファイルのパス

    Returns:
        Env: 環境変数のデータ
    """
    load_dotenv(dotenv_path=env_path)

    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID", "default")
    aws_region = os.environ.get("AWS_REGION", "ap-northeast-1")

    return Env(
        aws_access_key_id=aws_access_key_id,
        aws_region=aws_region,
    )


if __name__ == "__main__":
    env_path = "./.env"
    print(load_env(env_path=env_path))
