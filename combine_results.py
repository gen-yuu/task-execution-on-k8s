import io
import logging

import pandas as pd

from library.aws.client import S3Client
from library.aws.error import StorageError

logger = logging.getLogger(__name__)

# --- 設定項目 ---
# main.pyやrun_all_tasks.pyと同じ設定を使用
config = {
    "s3_bucket": "test-bucket",
    "s3_endpoint_url": "192.168.0.50:9000",
    "s3_access_key": "minioadmin",
    "s3_secret_key": "Ashushu0810@",
}

# 入力と出力のパス
RESULTS_PREFIX = "results/"
OUTPUT_KEY = "summary/all_results.csv"


def combine_results():
    """S3上の個別の結果CSVを1つにまとめる"""
    logger.info("--- Starting result combination process ---")

    s3_client = S3Client(
        endpoint_url=config["s3_endpoint_url"],
        access_key=config["s3_access_key"],
        secret_key=config["s3_secret_key"],
    )

    # 1. results/ 配下の全CSVファイルのキーを取得
    try:
        logger.info(
            "Listing all result files from s3://%s/%s",
            config["s3_bucket"],
            RESULTS_PREFIX,
        )
        result_keys = s3_client.list_object_keys(
            bucket_name=config["s3_bucket"], prefix=RESULTS_PREFIX
        )

        # .csvファイルのみを対象にする
        csv_keys = [key for key in result_keys if key.endswith(".csv")]

        if not csv_keys:
            logger.warning("No result CSV files found. Exiting.")
            return

        logger.info("Found %d result files to combine.", len(csv_keys))

    except StorageError:
        logger.critical("Failed to list result files from S3. Aborting.", exc_info=True)
        return

    # 2. 各CSVを読み込み、DataFrameのリストとして保持
    df_list = []
    for key in csv_keys:
        try:
            logger.info("Reading file: %s", key)
            s3_obj = s3_client.get_object(
                bucket_name=config["s3_bucket"], object_key=key
            )
            # bytes -> str -> ファイルライクオブジェクトに変換してpandasで読み込む
            csv_content = s3_obj.decode()
            df = pd.read_csv(io.StringIO(csv_content))
            df_list.append(df)
        except Exception:
            logger.error(
                "Failed to read or parse file: %s. Skipping.", key, exc_info=True
            )

    if not df_list:
        logger.error("Could not read any valid CSV files. Aborting.")
        return

    # 3. 全てのDataFrameを1つに結合
    logger.info("Combining %d DataFrames into one...", len(df_list))
    combined_df = pd.concat(df_list, ignore_index=True)
    logger.info("Successfully combined. Total rows: %d", len(combined_df))

    # 4. 結合したDataFrameを1つのCSVとしてS3にアップロード
    try:
        # DataFrameをCSV形式の文字列に変換
        output_csv_str = combined_df.to_csv(index=False)

        output_path = f"s3://{config['s3_bucket']}/{OUTPUT_KEY}"
        logger.info("Uploading combined CSV to %s", output_path)

        s3_client.put_object(
            bucket_name=config["s3_bucket"],
            object_key=OUTPUT_KEY,
            data=output_csv_str.encode("utf-8"),
            content_type="text/csv",
        )
        logger.info("Successfully uploaded combined results.")
    except Exception:
        logger.critical("Failed to upload the combined CSV file.", exc_info=True)

    logger.info("--- Result combination process finished. ---")


if __name__ == "__main__":
    combine_results()
