import csv
import datetime
import logging
import os
import time

from common.logger import setup_logger

# 必要なS3Clientとエラー、ロガー設定関数もインポート
from library.aws.client import S3Client
from library.aws.error import ObjectNotFoundError

# main.pyからmain関数をインポート
from main import main

config = {
    "s3_bucket": "test-bucket",
    "s3_endpoint_url": "192.168.0.50:9000",
    "s3_access_key": "minioadmin",
    "s3_secret_key": "Ashushu0810@",
}


def run_all_tasks():
    """
    S3からタスクリストを取得し、全てのタスクを逐次実行する。
    """
    # --- 1. このスクリプト全体のロガーを設定 ---
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = os.path.join(log_dir, f"batch_run_{timestamp}.log")

    # ルートロガーを設定することで、importした他のモジュールのログも同じ設定になる
    setup_logger(level=logging.INFO, logfile_path=logfile)
    logger = logging.getLogger(__name__)

    logger.info("======== STARTING BATCH TASK EXECUTION ========")

    # --- 2. S3からtasks.csvを読み込み、タスク総数を取得 ---
    try:
        s3_client = S3Client(
            endpoint_url=config["s3_endpoint_url"],
            access_key=config["s3_access_key"],
            secret_key=config["s3_secret_key"],
        )
        tasks_obj = s3_client.get_object(
            bucket_name=config["s3_bucket"], object_key="tasks.csv"
        )
        tasks_list = list(csv.reader(tasks_obj.decode().splitlines()))
        total_tasks = len(tasks_list) - 1  # ヘッダーを除く
        logger.info("Found %d tasks in tasks.csv.", total_tasks)
    except (ObjectNotFoundError, Exception):
        logger.critical(
            "Failed to read tasks.csv from S3. Aborting batch run.", exc_info=True
        )
        return

    # --- 3. 各タスクをループで実行 ---
    success_count = 0
    failure_count = 0

    overall_start_time = time.time()

    for i in range(total_tasks):
        if i % 3 != 2:  # yoloだけ実行
            continue
        logger.info("----------------------------------------------------")
        logger.info(">>> Starting task %d / %d", i, total_tasks - 1)
        logger.info("----------------------------------------------------")

        # main関数に渡すためのタスクごとの設定を作成
        task_config = config.copy()
        task_config["task_index"] = i

        try:
            # main.pyのmain関数を直接呼び出す
            return_code = main(task_config)
            if return_code == 0:
                logger.info(">>> Task %d finished SUCCESSFULLY.", i)
                success_count += 1
            else:
                # main関数が意図的にエラーコードを返した場合
                logger.error(
                    ">>> Task %d finished with a non-zero return code: %s",
                    i,
                    return_code,
                )
                failure_count += 1
        except Exception:
            # main関数内で捕捉されなかった例外が発生した場合
            logger.critical(
                "!!! Task %d CRASHED with an unhandled exception."
                "Continuing to next task.",
                i,
                exc_info=True,
            )
            failure_count += 1

    overall_end_time = time.time()
    total_duration_sec = overall_end_time - overall_start_time

    # --- 4. 最終結果のサマリーをログに出力 ---
    logger.info("======== BATCH TASK EXECUTION FINISHED ========")
    logger.info("Total execution time: %.2f seconds", total_duration_sec)
    logger.info("Success: %d tasks", success_count)
    logger.info("Failures: %d tasks", failure_count)
    logger.info("==============================================")


if __name__ == "__main__":
    run_all_tasks()
