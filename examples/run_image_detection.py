import argparse
import os
import time
from io import BytesIO
from pathlib import Path
from typing import List

import torch
from dotenv import load_dotenv
from PIL import Image

from common.logger import setup_logger
from common.storage import TaskStorage
from common.utils import get_system_info, load_model
from run_single_task.debug_utils import ensure_dir, save_annotated_image, save_json

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def run_inference(
    model,
    imgs: List[bytes],
    transform,
    device,
    model_name: str,
    debug_dir: Path | None = None,
):
    """
    単一モデルで画像リストを推論し、合計時間を返す。
    measure_transfer=True の場合、最初の .to(device) を計測に含める。
    """
    if debug_dir:
        out_dir = debug_dir / model_name
        ensure_dir(out_dir)

    t0 = time.perf_counter()
    model = model.to(device)

    # 推論
    for idx, img_bytes in enumerate(imgs):
        img = Image.open(BytesIO(img_bytes)).convert("RGB")  # PIL へ変換
        if transform is not None:  # torchvision 系
            tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(tensor)
            result_dict = outputs[0]
            detections = {
                "boxes": result_dict["boxes"].cpu().tolist(),
                "scores": result_dict["scores"].cpu().tolist(),
                "labels": result_dict["labels"].cpu().tolist(),
            }
        else:  # YOLOv8 系
            results = model.predict(img, device=device, verbose=False)
            results_obj = results[0]
            detections = {
                "boxes": results_obj.boxes.xyxy.cpu().tolist(),
                "scores": results_obj.boxes.conf.cpu().tolist(),
                "labels": results_obj.boxes.cls.cpu().tolist(),
            }

        if debug_dir:
            stem = f"frame_{idx:06d}"
            save_json(
                detections,
                out_dir / f"{stem}.json",
            )
            save_annotated_image(
                img_bytes,
                detections,
                out_dir / f"{stem}.png",
                0.5,
            )

    elapsed = time.perf_counter() - t0
    return elapsed


def main(args):
    try:
        logger = setup_logger()
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

        system_info = get_system_info(device)
        logger.info("System Information:", extra={"system_info": system_info})

        # MinIOクライアントとモデルの初期化
        minio_client = TaskStorage(
            args.minio_endpoint,
            args.minio_access_key,
            args.minio_secret_key,
            args.data_bucket,
            args.result_bucket,
        )

        prefix = "videos/video0001"
        logger.info(f"Loading images from {prefix}")
        image_paths = minio_client.get_task_image_paths(prefix)
        images = [minio_client.get_image_bytes(p) for p in image_paths]
        logger.info(f"Loaded {len(images)} images")

        model_names = ["faster_rcnn", "ssd300_vgg16", "yolov8s"]
        debug_output_root = Path("./debug_outputs")
        for name in model_names:
            model, transform = load_model(name)
            logger.info(f"Running model: {name}")
            latency = run_inference(
                model=model,
                imgs=images,
                transform=transform,
                device=device,
                model_name=name,
                debug_dir=debug_output_root,  # None にすれば保存しない
            )
            logger.info(
                "Inference finished",
                extra={
                    "model": name,
                    "latency_sec": latency,
                    "images": len(images),
                    "device": gpu_name,
                },
            )
    except Exception as e:
        logger.error(
            "An error occurred",
            extra={
                "error": str(e),
            },
            exc_info=True,
        )
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a single object detection task using data from MinIO."
    )
    # Task setting
    parser.add_argument(
        "--task-name",
        type=str,
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
        help='MinIO server endpoint (e.g., "minio-service:9000").',
    )
    parser.add_argument("--minio-access-key", type=str, help="MinIO access key.")
    parser.add_argument("--minio-secret-key", type=str, help="MinIO secret key.")
    parser.add_argument(
        "--data-bucket",
        type=str,
        default="gpu-perf-predictor",
        help="MinIO bucket for input data.",
    )
    parser.add_argument(
        "--result-bucket",
        type=str,
        default="results",
        help="MinIO bucket for output results.",
    )

    args = parser.parse_args()
    # 環境変数からキーを読み込むことも可能（k8s Secret利用時に便利）
    args.minio_endpoint = os.getenv("MINIO_ENDPOINT", args.minio_endpoint)
    args.minio_access_key = os.getenv("MINIO_ACCESS_KEY", args.minio_access_key)
    args.minio_secret_key = os.getenv("MINIO_SECRET_KEY", args.minio_secret_key)
    main(args)
