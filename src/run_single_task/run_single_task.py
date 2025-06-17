import argparse
import os
import socket
import time
from io import BytesIO

import pandas as pd
import torch
import torchvision
import torchvision.transforms as T
from minio import Minio
from PIL import Image
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)

from .logger import setup_logger
from .utils import get_system_info


def init_minio_client(endpoint: str, access_key: str, secret_key: str) -> Minio:
    """MinIOクライアントを初期化します。"""
    print(f"Initializing MinIO client for endpoint: {endpoint}")
    client = Minio(
        endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=False,  # k8sクラスタ内通信など、非HTTPSの場合はFalseに設定
    )
    return client


def load_model(device: torch.device):
    """事前学習済みモデルをロードします。"""
    print("Loading pre-trained Faster R-CNN model...")
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
    return model, weights.transforms()


def run_and_measure_task(
    task_name: str,
    minio_client: Minio,
    data_bucket: str,
    model: torch.nn.Module,
    transform: T.Compose,
    device: torch.device,
) -> dict | None:
    """
    MinIOからデータを読み込み、単一タスクのGPU時間とウォールクロック時間を計測します。
    """
    # === ウォールクロック時間計測開始 ===
    wall_start_time = time.perf_counter()

    # 1. MinIOから画像オブジェクトのリストを取得
    prefix = f"{task_name}/"
    try:
        objects = minio_client.list_objects(data_bucket, prefix=prefix, recursive=True)
        image_object_names = [
            obj.object_name
            for obj in objects
            if obj.object_name.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    except Exception as e:
        print(f"Error listing objects from MinIO for task {task_name}: {e}")
        return None

    if not image_object_names:
        print(
            f"Warning: No images found in bucket '{data_bucket}' with prefix '{prefix}'."
        )
        return None

    print(f"Found {len(image_object_names)} images for task '{task_name}'.")

    # 2. ウォームアップ (I/OとGPUカーネル初期化のため)
    print("Warming up...")
    try:
        response = minio_client.get_object(data_bucket, image_object_names[0])
        img_bytes = response.read()
        response.close()
        response.release_conn()
        with torch.no_grad():
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            _ = model(tensor)
        torch.cuda.synchronize(device)
    except Exception as e:
        print(f"Warning: Warmup failed, continuing without it. Error: {e}")

    # 3. 本計測
    total_gpu_time_ms = 0.0
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for object_name in image_object_names:
        # MinIOから画像データをダウンロード
        response = minio_client.get_object(data_bucket, object_name)
        img_bytes = response.read()
        response.close()
        response.release_conn()

        # データ前処理
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)

        # GPU実行時間を計測
        start_event.record()
        with torch.no_grad():
            _ = model(tensor)
        end_event.record()
        torch.cuda.synchronize(device)

        total_gpu_time_ms += start_event.elapsed_time(end_event)

    # === ウォールクロック時間計測終了 ===
    wall_end_time = time.perf_counter()
    total_wall_time_s = wall_end_time - wall_start_time

    return {
        "num_images": len(image_object_names),
        "total_gpu_time_ms": total_gpu_time_ms,
        "total_wall_time_s": total_wall_time_s,
    }


def main(args):
    """メイン実行関数"""
    logger = setup_logger(__name__)

    # デバイス設定
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

    logger.info(f"Starting Task '{args.task_name}'")

    system_info = get_system_info(device)
    logger.info("System Information:", extra={"system_info": system_info})

    # MinIOクライアントとモデルの初期化
    # minio_client = init_minio_client(
    #     args.minio_endpoint, args.minio_access_key, args.minio_secret_key
    # )
    # model, transform = load_model(device)

    # タスク実行と計測
    # measurement = run_and_measure_task(
    #     args.task_name, minio_client, args.data_bucket, model, transform, device
    # )

    # # 結果の処理とアップロード
    # if measurement:
    #     num_images = measurement["num_images"]
    #     avg_gpu_time_ms = (
    #         measurement["total_gpu_time_ms"] / num_images if num_images > 0 else 0
    #     )
    #     avg_wall_time_s = (
    #         measurement["total_wall_time_s"] / num_images if num_images > 0 else 0
    #     )

    #     final_result = {
    #         "hostname": socket.gethostname(),
    #         "gpu_name": gpu_name,
    #         "task_name": args.task_name,
    #         "model_name": "FasterRCNN_ResNet50_FPN",
    #         **measurement,
    #         "avg_gpu_time_ms_per_image": avg_gpu_time_ms,
    #         "avg_wall_time_s_per_image": avg_wall_time_s,
    #     }

    #     print("\n--- Measurement Summary ---")
    #     for key, value in final_result.items():
    #         print(f"{key}: {value}")
    #     print("-------------------------\n")

    #     # 結果をCSVとしてMinIOにアップロード
    #     df = pd.DataFrame([final_result])
    #     csv_bytes = df.to_csv(index=False).encode("utf-8")
    #     result_object_name = f"{args.task_name}.csv"

    #     try:
    #         minio_client.put_object(
    #             args.result_bucket,
    #             result_object_name,
    #             data=BytesIO(csv_bytes),
    #             length=len(csv_bytes),
    #             content_type="application/csv",
    #         )
    #         print(
    #             f"✅ Successfully uploaded result to '{args.result_bucket}/{result_object_name}'"
    #         )
    #     except Exception as e:
    #         print(f"❌ Failed to upload result to MinIO. Error: {e}")
    # else:
    # print(f"❌ Task '{args.task_name}' failed or produced no results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a single object detection task using data from MinIO."
    )
    # Task setting
    parser.add_argument(
        "--task-name",
        type=str,
        required=True,
        help='Name of the task directory in MinIO (e.g., "video0001").',
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help='Device to run on (e.g., "cuda:0", "cpu").',
    )
    # MinIO settings
    parser.add_argument(
        "--minio-endpoint",
        type=str,
        required=True,
        help='MinIO server endpoint (e.g., "minio-service:9000").',
    )
    parser.add_argument(
        "--minio-access-key", type=str, required=True, help="MinIO access key."
    )
    parser.add_argument(
        "--minio-secret-key", type=str, required=True, help="MinIO secret key."
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
    # 環境変数からキーを読み込むことも可能（k8s Secret利用時に便利）
    args.minio_access_key = os.getenv("MINIO_ACCESS_KEY", args.minio_access_key)
    args.minio_secret_key = os.getenv("MINIO_SECRET_KEY", args.minio_secret_key)
    main(args)
