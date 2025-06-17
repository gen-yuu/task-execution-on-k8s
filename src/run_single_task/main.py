# src/run_single_task/main.py

import argparse
import logging
import os

import torch

from ..common.logger import setup_logger

# 同じパッケージ内の他モジュールから、必要な関数をインポート
from .executor import run_and_measure_task
from .utils import get_system_info, init_minio_client, load_model


def main():
    """
    メイン実行関数
    """
    # 1. ロガーをセットアップ
    logger = setup_logger(__name__)

    # 2. コマンドライン引数を解析
    parser = argparse.ArgumentParser(description="Run a single object detection task.")
    parser.add_argument(
        "--task-name",
        type=str,
        required=True,
        help="Name of the task directory in MinIO.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to run on."
    )
    parser.add_argument(
        "--minio-endpoint", type=str, required=True, help="MinIO server endpoint."
    )
    parser.add_argument(
        "--data-bucket", type=str, default="videos", help="MinIO bucket for input data."
    )
    parser.add_argument(
        "--result-bucket",
        type=str,
        default="results",
        help="MinIO bucket for output results.",
    )
    args = parser.parse_args()

    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")
    if not access_key or not secret_key:
        raise ValueError

    logger.info(
        "Starting task",
        extra={
            "task_name": args.task_name,
            "device": args.device,
        },
    )

    try:
        if torch.cuda.is_available() and "cuda" in args.device:  # GPUが利用可能か確認
            device = torch.device(args.device)
            gpu_name = torch.cuda.get_device_name(device)
            logger.info(
                "GPU is available. Using GPU.",
                extra={"gpu_name": gpu_name},
            )
        else:
            device = torch.device("cpu")
            gpu_name = "CPU"
            logger.warning("GPU is not available. Using CPU.")
        system_info = get_system_info(device)
        minio_client = init_minio_client(args.minio_endpoint, access_key, secret_key)
        model, transform = load_model(device)

        # 4. メインの計測処理を実行
        measurement = run_and_measure_task(
            task_name=args.task_name,
            minio_client=minio_client,
            data_bucket=args.data_bucket,
            model=model,
            transform=transform,
            device=device,
        )

        # 5. 結果を処理
        if measurement:
            # ... (結果を整形し、MinIOにアップロードする処理) ...
            logger.info(f"Task {args.task_name} completed successfully.")
        else:
            logger.error(f"Task {args.task_name} failed or produced no results.")

    except Exception as e:
        logger.error(
            f"An unexpected error occurred during task {args.task_name}", exc_info=True
        )


if __name__ == "__main__":
    main()
