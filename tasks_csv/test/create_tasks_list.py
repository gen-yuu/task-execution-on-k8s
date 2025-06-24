import csv
import io
import logging

import boto3
from botocore.exceptions import ClientError

from common.logger import setup_logger

# --- 設定項目 ---
S3_ENDPOINT_URL = "http://192.168.0.50:9000"  # MinIOなどローカルS3を使う場合
S3_ACCESS_KEY = "minioadmin"
S3_SECRET_KEY = "Ashushu0810@"

SOURCE_BUCKET = "gpu-perf-predictor"  # コピー元のバケット
DEST_BUCKET = "test-bucket"  # コピー先のバケット (旧 BUCKET_NAME)

logger = setup_logger(level=logging.INFO)


def copy_s3_data(
    s3_resource, source_bucket_name: str, dest_bucket_name: str, video_paths: list[str]
):
    """
    指定されたvideo_pathsに含まれる全てのオブジェクトを、sourceからdestバケットにコピーする。
    """
    logger.info(
        "--- Starting data copy from '%s' to '%s' ---",
        source_bucket_name,
        dest_bucket_name,
    )

    source_bucket = s3_resource.Bucket(source_bucket_name)
    dest_bucket = s3_resource.Bucket(dest_bucket_name)

    total_copied_count = 0
    for prefix in video_paths:
        copied_count_per_prefix = 0
        logger.info("Copying objects with prefix: %s", prefix)
        try:
            # プレフィックスに一致する全てのオブジェクトをループ
            for obj in source_bucket.objects.filter(Prefix=prefix):
                copy_source = {"Bucket": obj.bucket_name, "Key": obj.key}
                # 新しいキーも同じパスにする
                new_key = obj.key
                dest_bucket.copy(copy_source, new_key)
                copied_count_per_prefix += 1

            if copied_count_per_prefix > 0:
                logger.info(
                    "-> Copied %d objects for prefix '%s'",
                    copied_count_per_prefix,
                    prefix,
                )
            else:
                logger.warning(
                    "-> No objects found for prefix '%s' in source bucket.", prefix
                )

            total_copied_count += copied_count_per_prefix

        except ClientError as e:
            logger.error("An error occurred while copying prefix '%s': %s", prefix, e)
            continue  # エラーが発生しても次のプレフィックスのコピーを試みる

    logger.info(
        "--- Finished data copy. Total objects copied: %d ---", total_copied_count
    )


def create_task_list_csv(s3_client, dest_bucket_name: str, video_paths: list[str]):
    """
    タスクリスト(tasks.csv)を生成し、指定されたバケットにアップロードする。
    """
    logger.info("--- Generating and uploading task list (tasks.csv) ---")

    models = ["faster_rcnn", "ssd300_vgg16", "yolov8s"]
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["task_id", "model_name", "video_path"])

    task_id_counter = 1
    for video_path in video_paths:
        for model_name in models:
            writer.writerow([task_id_counter, model_name, video_path])
            task_id_counter += 1

    try:
        s3_client.put_object(
            Bucket=dest_bucket_name, Key="tasks.csv", Body=output.getvalue()
        )
        logger.info("Uploaded: s3://%s/tasks.csv", dest_bucket_name)
    except ClientError as e:
        logger.error("Failed to upload tasks.csv: %s", e)


def setup_test_environment():
    """
    テスト環境をセットアップするメイン関数。
    """
    # 接続情報
    s3_connection_args = {
        "endpoint_url": S3_ENDPOINT_URL,
        "aws_access_key_id": S3_ACCESS_KEY,
        "aws_secret_access_key": S3_SECRET_KEY,
    }

    try:
        # clientはオブジェクトのput/getなど単発の操作に使う
        s3_client = boto3.client("s3", **s3_connection_args)
        # resourceはオブジェクトのリスト取得やバッチ操作など高レベルな操作に使う
        s3_resource = boto3.resource("s3", **s3_connection_args)
        logger.info("Successfully connected to S3.")
    except Exception as e:
        logger.error("Failed to connect to S3: %s", e)
        return

    # コピー先バケットが存在しない場合は作成
    try:
        s3_client.head_bucket(Bucket=DEST_BUCKET)
        logger.info("Destination bucket '%s' already exists.", DEST_BUCKET)
    except ClientError:
        logger.info("Destination bucket '%s' not found. Creating it...", DEST_BUCKET)
        s3_client.create_bucket(Bucket=DEST_BUCKET)

    # 対象となる動画ディレクトリのリスト
    video_paths_to_process = [f"videos/video{i:04d}" for i in range(1, 11)]

    # 1. データをコピーする
    copy_s3_data(s3_resource, SOURCE_BUCKET, DEST_BUCKET, video_paths_to_process)

    # 2. タスクリストを作成する
    create_task_list_csv(s3_client, DEST_BUCKET, video_paths_to_process)

    logger.info("--- Test environment setup finished. ---")


if __name__ == "__main__":
    setup_test_environment()
