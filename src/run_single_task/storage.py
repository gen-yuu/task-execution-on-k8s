import logging

import pandas as pd

from common.error import StorageError
from library.aws.client import S3Client

logger = logging.getLogger(__name__)


class TaskStorage:
    """
    S3Clientを利用して、このアプリケーション固有のストレージ操作を行うクラス
    """

    def __init__(self, s3_client: S3Client, data_bucket: str, result_bucket: str):
        """
        TaskStorageを初期化する

        Args:
            s3_client (S3Client): 通信に使用するS3クライアントのインスタンス。
            data_bucket (str): 入力データが格納されているバケット名。
            result_bucket (str): 結果を保存するバケット名。
        """
        self.client = s3_client
        self.data_bucket = data_bucket
        self.result_bucket = result_bucket
        logger.info(
            f"TaskStorage initialized for data_bucket='{data_bucket}',\
                  result_bucket='{result_bucket}'"
        )

    def get_task_image_paths(self, task_name: str) -> list[str]:
        """
        タスク名から、処理対象となる画像ファイルのパスリストを取得します。

        Raises:
            StorageError: オブジェクトの一覧取得に失敗した場合。
        """
        prefix = f"{task_name}/"
        try:
            all_objects = self.client.list_objects(self.data_bucket, prefix=prefix)

            # 画像ファイルのみをフィルタリングする
            image_paths = [
                path
                for path in all_objects
                if path.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            logger.info(f"Found {len(image_paths)} images for task '{task_name}'.")
            return image_paths
        except StorageError as e:
            logger.error(
                f"Could not get image paths for task '{task_name}'.",
                extra={
                    "error": str(e),
                },
            )
            raise
        except Exception as e:
            logger.error(
                f"An unexpected error occurred while getting image paths \
                    for task '{task_name}'.",
                extra={
                    "error": str(e),
                },
            )
            raise

    def get_image_bytes(self, image_path: str) -> bytes:
        """
        単一の画像パスからバイトデータを取得します。

        Raises:
            ObjectNotFoundError: 画像が見つからない場合。
            StorageError: その他の取得エラーが発生した場合。
        """
        return self.client.get_object(self.data_bucket, image_path)

    def upload_result_df(self, task_name: str, hostname: str, result_df: pd.DataFrame):
        """
        計測結果のDataFrameを受け取り、CSVとしてアップロードします。

        Raises:
            StorageError: アップロードに失敗した場合。
        """
        # アップロードするオブジェクト名を生成
        object_name = f"results/{task_name}_{hostname}.csv"

        # DataFrameをCSVに変換
        csv_string = result_df.to_csv(index=False)
        csv_bytes = csv_string.encode("utf-8")

        try:
            self.client.put_object(
                bucket_name=self.result_bucket,
                object_name=object_name,
                data=csv_bytes,
                content_type="text/csv",
            )
        except StorageError as e:
            logger.error(
                f"Failed to upload result for task '{task_name}'.",
                extra={
                    "error": str(e),
                },
            )
            raise
        except Exception as e:
            logger.error(
                f"An unexpected error occurred while uploading result \
                    for task '{task_name}'.",
                extra={
                    "error": str(e),
                },
            )
            raise
