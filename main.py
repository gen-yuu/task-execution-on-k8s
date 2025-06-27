import csv
import datetime
import io
import json
import logging
import os
import statistics
import time

import torch

from common.utils import get_system_info, load_model, sanitize_for_path
from library.aws.client import S3Client
from library.aws.error import ImageDecodeError, ObjectNotFoundError
from run_single_task.feature_extractor import ModelFeatureExtractor

logger = logging.getLogger(__name__)

COLOR_SPACE_TO_CHANNELS = {
    "sRGB": 3,
    "RGB": 3,
    "Grayscale": 1,
    "Gray": 1,
}

# msの定数
MILLISECONDS_PER_SECOND = 1000


def load_frames_iteratively(s3_client, bucket_name, frame_keys, logger):
    """
    フレームを1つずつロードしてyieldするジェネレータ関数

    Args:
        s3_client (S3Client): S3クライアントオブジェクト
        bucket_name (str): バケット名
        frame_keys (list): フレームキーのリスト
        logger (logging.Logger): ロガー

    Yields:
        PIL.Image.Image: ロードした画像
    """
    for key in frame_keys:
        try:
            pil_image = s3_client.get_image(
                bucket_name=bucket_name,
                object_key=key,
            )
            yield pil_image
        except (ObjectNotFoundError, ImageDecodeError) as e:
            logger.warning("Skipping frame '%s' due to an error: %s", key, e)


def main(config):
    logger.info("Starting main function")
    # タスク情報と環境変数の取得
    task_index = config.get("task_index", 0)
    s3_endpoint_url = config.get("s3_endpoint_url")
    s3_access_key = config.get("s3_access_key")
    s3_secret_key = config.get("s3_secret_key")
    s3_bucket = config.get("s3_bucket")
    logger.info(
        "Task information and environment variables have been collected.",
        extra={
            "task_index": task_index,
            "s3_endpoint_url": s3_endpoint_url,
            "s3_bucket": s3_bucket,
        },
    )

    # GPU環境の確認（本実験はGPUが使えない場合はerrorとする）
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logger.info(
            "GPU is available. Using GPU.",
            extra={"gpu_name": torch.cuda.get_device_name(device)},
        )
    else:
        logger.error("GPU is not available.")
        raise Exception("GPU is not available.")

    # 実行環境の取得
    logger.info("Collecting system information")
    system_info = get_system_info(device)

    s3_client = S3Client(
        endpoint_url=s3_endpoint_url,
        access_key=s3_access_key,
        secret_key=s3_secret_key,
    )

    # タスクリストをS3から取得
    logger.info("Downloading task list from S3")
    try:
        tasks_obj = s3_client.get_object(
            bucket_name=s3_bucket,
            object_key="tasks.csv",
        )
        logger.info(
            f"Task list '{tasks_obj.key}' downloaded"
            f"({tasks_obj.content_length} bytes)."
        )

        # ヘルパーメソッドを使ってデコード
        tasks_str = tasks_obj.decode()
        tasks_list = list(csv.reader(tasks_str.splitlines()))
    except ObjectNotFoundError as e:
        logger.error(
            "Task list not found in S3.",
            extra={
                "error": str(e),
            },
            exc_info=True,
        )
        raise Exception("Task list not found in S3.")
    except Exception as e:
        logger.error(
            "Failed to download task list from S3.",
            extra={"error": str(e)},
            exc_info=True,
        )
        raise Exception("Failed to download task list from S3.")

    # ヘッダーを除き、自分のインデックスのタスクを取得
    num_tasks = len(tasks_list) - 1  # ヘッダーを除くタスク数
    if task_index >= num_tasks:
        logger.error(
            f"Task index {task_index} is out of range. Total tasks: {num_tasks}"
        )
        raise IndexError
    task_info = tasks_list[task_index + 1]
    model_name, video_path = task_info[1], task_info[2]

    try:
        # 動画のメタデータ取得
        logger.info("Downloading metadata from S3")
        metadata_key = f"{video_path}/metadata.json"
        metadata_obj = s3_client.get_object(
            bucket_name=s3_bucket,
            object_key=metadata_key,
        )

        metadata_str = metadata_obj.decode()
        data = json.loads(metadata_str)

        sequence_data = data.get("sequence", {})

        # メタデータにキーがない場合は、最も一般的な'sRGB'をデフォルト値
        color_space_str = sequence_data.get("color_space", "sRGB")
        num_channels = COLOR_SPACE_TO_CHANNELS.get(color_space_str, 3)

        # FLOPs計算用に解像度を変数として保持
        im_width = sequence_data.get("imWidth")
        im_height = sequence_data.get("imHeight")

        video_info = {
            "name": sequence_data.get("name"),
            "seqLength": sequence_data.get("seqLength"),
            "imWidth": im_width,
            "imHeight": im_height,
            "numChannels": num_channels,
        }

        logger.info(
            f"Metadata '{metadata_key}' downloaded"
            f"({metadata_obj.content_length} bytes)."
        )
    except ObjectNotFoundError as e:
        logger.warning(
            "Metadata not found in S3.",
            extra={
                "error": str(e),
            },
            exc_info=True,
        )  # のちにない場合は取得するスクリプトでfallback
        raise
    except Exception as e:
        logger.error(
            "Failed to download metadata from S3.",
            extra={"error": str(e)},
            exc_info=True,
        )
        raise

    # モデルの読み込み
    try:
        model, transform = load_model(model_name)
        logger.info(f"Model '{model_name}' loaded successfully.")
    except Exception as e:
        logger.error(
            f"Failed to load model '{model_name}'",
            extra={
                "error": str(e),
            },
            exc_info=True,
        )
        raise Exception

    # 特徴量抽出
    logger.info(
        "Extracting model features on CPU.",
        extra={"model_name": model_name},
    )
    try:
        input_resolution = (num_channels, im_height, im_width)
        extractor = ModelFeatureExtractor(
            model=model,
            input_resolution=input_resolution,
        )
        model_features = extractor.extract_features()
        logger.info("Model features extracted successfully on CPU.")
    except Exception as e:
        logger.error(
            "Failed to extract model features on CPU.",
            extra={
                "error": str(e),
            },
            exc_info=True,
        )
        raise

    logger.info("Proceeding to inference phase.")

    frame_prefix = f"{video_path}/frames/"

    frame_keys = s3_client.list_object_keys(
        bucket_name=s3_bucket,
        prefix=frame_prefix,
    )
    frame_keys.sort()

    logger.info(f"Found {len(frame_keys)} frames in '{frame_prefix}'.")

    frame_generator = load_frames_iteratively(s3_client, s3_bucket, frame_keys, logger)

    # for key in frame_keys:
    #     try:
    #         # PIL Imageオブジェクトを取得
    #         pil_image = s3_client.get_image(
    #             bucket_name=s3_bucket,
    #             object_key=key,
    #         )
    #         frames_in_memory.append(pil_image)
    #     except (ObjectNotFoundError, ImageDecodeError) as e:
    #         logger.warning("Skipping frame '%s' due to an error: %s", key, e)

    # モデルのGPU転送と、その時間の計測
    logger.info(f"Transferring model to device: {str(device)}")
    transfer_start_event = torch.cuda.Event(enable_timing=True)
    transfer_end_event = torch.cuda.Event(enable_timing=True)

    transfer_start_event.record()
    model.to(device)
    transfer_end_event.record()
    torch.cuda.synchronize()  # 転送の完了を待つ
    model_transfer_time_ms = transfer_start_event.elapsed_time(transfer_end_event)
    logger.info(f"Model transferred to GPU in {model_transfer_time_ms} ms.")

    # 各フレームで推論を実行し、時間を計測
    logger.info(
        "Starting inference loop for each frame.",
        extra={"model_name": model_name},
    )
    wall_times_per_frame = []
    gpu_compute_times_per_frame = []
    model.eval()
    try:
        for i, pil_image in enumerate(frame_generator):
            torch.cuda.synchronize()
            wall_start_time = time.time()

            # データの前処理 (CPU上)
            if transform:
                # torchvision系のモデルなど、transformが定義されている場合
                input_tensor_cpu = transform(pil_image)
                input_tensor_cpu = input_tensor_cpu.unsqueeze(0)
            else:
                # YOLOv8sなど、transformがNoneの場合
                # YOLOv8はPIL Imageを直接扱えるが、ここでは他のモデルと型を揃えるため
                # 最低限のテンソル化を行う
                from torchvision import transforms

                input_tensor_cpu = transforms.ToTensor()(pil_image)
                input_tensor_cpu = input_tensor_cpu.unsqueeze(0)

            # フレームのデータ転送 (CPU -> GPU)
            input_tensor_gpu = input_tensor_cpu.to(device)

            # GPU演算時間の計測
            gpu_start_event = torch.cuda.Event(enable_timing=True)
            gpu_end_event = torch.cuda.Event(enable_timing=True)
            gpu_start_event.record()

            with torch.no_grad():
                if transform:
                    # --- torchvision系のモデル ---
                    input_tensor_cpu = transform(pil_image).unsqueeze(0)
                    input_tensor_gpu = input_tensor_cpu.to(device)

                    gpu_start_event.record()
                    results = model(input_tensor_gpu)
                    gpu_end_event.record()

                    # 結果をCPUに戻す
                    results_cpu = [{k: v.cpu() for k, v in r.items()} for r in results]

                else:
                    # --- YOLOv8系のモデル ---
                    gpu_start_event.record()

                    # model.predict()が内部でデータ転送を行う
                    results = model.predict(pil_image, device=device, verbose=False)

                    gpu_end_event.record()

                    # 結果オブジェクトから必要な情報を抽出してCPUに戻す
                    results_obj = results[0]
                    detections_cpu = {
                        "boxes": results_obj.boxes.xyxy.cpu().tolist(),
                        "scores": results_obj.boxes.conf.cpu().tolist(),
                        "labels": results_obj.boxes.cls.cpu().tolist(),
                    }

            # 全てのGPU処理の完了を待つ
            torch.cuda.synchronize()
            wall_end_time = time.time()

            wall_times_per_frame.append(
                (wall_end_time - wall_start_time) * MILLISECONDS_PER_SECOND
            )
            gpu_compute_times_per_frame.append(
                gpu_start_event.elapsed_time(gpu_end_event)
            )
        logger.info("Inference loop finished.")
        # ピークGPUメモリ使用量を取得 (バイト単位)
        peak_gpu_memory_bytes = torch.cuda.max_memory_allocated(device)
        peak_gpu_memory_mb = peak_gpu_memory_bytes / (1024**2)
        logger.info(
            "Peak GPU memory usage during inference: %.2f MB",
            peak_gpu_memory_mb,
        )
        # 計測後にリセットしておく
        torch.cuda.reset_peak_memory_stats(device)
    except Exception as e:
        logger.error(
            "An error occurred during inference.",
            extra={"error": str(e)},
            exc_info=True,
        )
        raise

    logger.info("Aggregating results and preparing for upload.")
    # 合計時間を計算
    total_wall_time_ms = sum(wall_times_per_frame) + model_transfer_time_ms
    total_gpu_compute_time_ms = sum(gpu_compute_times_per_frame)

    # 処理したフレーム数を記録（念の為）
    num_processed_frames = len(wall_times_per_frame)
    # 1フレームあたりの平均時間と標準偏差を計算
    avg_wall_time_ms = (
        statistics.mean(wall_times_per_frame) if wall_times_per_frame else 0
    )
    std_wall_time_ms = (
        statistics.stdev(wall_times_per_frame) if num_processed_frames > 1 else 0
    )
    avg_gpu_compute_time_ms = (
        statistics.mean(gpu_compute_times_per_frame)
        if gpu_compute_times_per_frame
        else 0
    )
    std_gpu_compute_time_ms = (
        statistics.stdev(gpu_compute_times_per_frame) if num_processed_frames > 1 else 0
    )

    final_result = {
        # --- 静的情報 ---
        **system_info,
        **video_info,
        **model_features,
        "model_name": model_name,
        "task_index": config.get("task_index"),
        # --- 計測結果 ---
        "num_processed_frames": num_processed_frames,
        "model_transfer_time_ms": model_transfer_time_ms,
        "total_wall_time_sec": total_wall_time_ms / 1000.0,  # 合計時間は秒単位
        "total_gpu_compute_time_sec": total_gpu_compute_time_ms / 1000.0,
        "avg_wall_time_ms": avg_wall_time_ms,
        "std_wall_time_ms": std_wall_time_ms,
        "avg_gpu_compute_time_ms": avg_gpu_compute_time_ms,
        "std_gpu_compute_time_ms": std_gpu_compute_time_ms,
        "peak_gpu_memory_mb": peak_gpu_memory_mb if peak_gpu_memory_mb else -1,
    }

    # パフォーマンスサマリーをログに出力
    logger.info(
        "Performance summary: Avg Wall Time=%.2f ms, Avg GPU Compute Time=%.2f ms",
        final_result["avg_wall_time_ms"],
        final_result["avg_gpu_compute_time_ms"],
        extra={
            "total_wall_time_sec": final_result["total_wall_time_sec"],
            "std_wall_time_ms": final_result["std_wall_time_ms"],
            "std_gpu_compute_time_ms": final_result["std_gpu_compute_time_ms"],
        },
    )

    # CSV形式に変換してS3にアップロード
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=final_result.keys())
    writer.writeheader()
    writer.writerow(final_result)
    # system_infoからCPUとGPUのモデル名を取得
    cpu_name_raw = system_info.get("cpu_model", "unknown-cpu")
    gpu_name_raw = system_info.get("gpu_model", "unknown-gpu")
    hw_config_name = (
        f"cpu_{sanitize_for_path(cpu_name_raw)}_gpu_{sanitize_for_path(gpu_name_raw)}"
    )
    logger.info(f"Generated hardware config name for results: {hw_config_name}")

    # task_indexを4桁でゼロパディング (例: 0 -> "0000", 12 -> "0012")
    result_key = f"results/{hw_config_name}/result_{task_index:04d}.csv"
    try:
        s3_client.put_object(
            bucket_name=s3_bucket,
            object_key=result_key,
            data=output.getvalue().encode("utf-8"),
            content_type="text/csv",
        )
        logger.info(f"Uploaded: s3://{s3_bucket}/{result_key}")
    except Exception as e:
        logger.error(
            "Failed to upload results",
            extra={"error": str(e)},
            exc_info=True,
        )
        raise
    logger.info("Task finished successfully.")

    # ログfileをS3にアップロード
    logfile_path = config.get("logfile_path")
    # logfile_pathが設定されており、かつファイルが実際に存在する場合のみ実行
    if logfile_path and os.path.exists(logfile_path):
        logger.info("Uploading log file to S3...")
        try:
            log_basename = os.path.basename(logfile_path)
            s3_log_key = f"logs/{hw_config_name}/{log_basename}"
            # ログファイルをバイナリモードで読み込む
            with open(logfile_path, "rb") as f:
                log_data = f.read()

            # S3にアップロード
            s3_client.put_object(
                bucket_name=s3_bucket,
                object_key=s3_log_key,
                data=log_data,
                content_type="text/plain",
            )
            logger.info(
                "Successfully uploaded log file to s3://%s/%s", s3_bucket, s3_log_key
            )
        except Exception as e:
            logger.error(
                "Failed to upload log file.",
                extra={"error": str(e)},
                exc_info=True,
            )
    else:
        logger.info("Log file not found or path not specified, skipping log upload.")
    return 0


if __name__ == "__main__":
    import argparse

    from common.logger import setup_logger

    # --- コマンドライン引数のパーサーを定義 ---
    parser = argparse.ArgumentParser(
        description="Run a single task for GPU performance prediction."
    )

    # --test-mode フラグ
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode, using command-line arguments\
              instead of environment variables.",
    )

    # テスト時に上書きしたい設定を引数として定義
    parser.add_argument("--task-index", type=int, help="Task index to run.")
    parser.add_argument("--s3-endpoint-url", type=str, help="S3 endpoint URL.")
    parser.add_argument("--s3-access-key", type=str, help="S3 access key.")
    parser.add_argument("--s3-secret-key", type=str, help="S3 secret key.")
    parser.add_argument("--s3-bucket", type=str, help="S3 bucket name.")

    args = parser.parse_args()

    # 本番環境をデフォルト値としてconfigを構築
    config = {
        "task_index": int(os.environ.get("JOB_COMPLETION_INDEX", 0)),
        "s3_endpoint_url": os.environ.get("S3_ENDPOINT_URL"),
        "s3_access_key": os.environ.get("S3_ACCESS_KEY"),
        "s3_secret_key": os.environ.get("S3_SECRET_KEY"),
        "s3_bucket": os.environ.get("S3_BUCKET"),
    }

    # テストモードの場合、コマンドライン引数の値でconfigを上書き
    if args.test_mode:
        if args.task_index is not None:
            config["task_index"] = args.task_index
        if args.s3_endpoint_url:
            config["s3_endpoint_url"] = args.s3_endpoint_url
        if args.s3_access_key:
            config["s3_access_key"] = args.s3_access_key
        if args.s3_secret_key:
            config["s3_secret_key"] = args.s3_secret_key
        if args.s3_bucket:
            config["s3_bucket"] = args.s3_bucket

    # ログファイルパスを決定し、ロガーをセットアップ
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # テストモードか本番モードかでファイル名のプレフィックスを切り替える
    log_prefix = "test_run_" if args.test_mode else ""

    logfile_name = f"{log_prefix}{timestamp}_task_{config['task_index']:04d}.log"
    logfile_path = os.path.join(log_dir, logfile_name)

    # アプリケーション全体のロガーを設定
    setup_logger(level=logging.INFO, logfile_path=logfile_path)

    # アップロードのために、生成したログパスをconfigに追加
    config["logfile_path"] = logfile_path

    try:
        if args.test_mode:
            logger.info("Running in TEST MODE with the following config:")
        else:
            logger.info("Running in PRODUCTION MODE with the following config:")

        # パスワードなどはログに出力しないようにする
        safe_config_to_log = {k: v for k, v in config.items() if "key" not in k.lower()}
        logger.info("Config contents", extra=safe_config_to_log)
        main(config)

    except Exception as e:
        # main関数で捕捉されなかった最終的なクラッシュを記録
        logger.critical(
            "The main function has CRASHED with an unhandled exception.",
            extra={"error": str(e)},
            exc_info=True,
        )
        exit(1)
