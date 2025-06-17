import logging
from typing import List

import pandas as pd

from common.error import ObjectNotFoundError, StorageError
from library.aws.client import S3Client

logger = logging.getLogger(__name__)


class TaskStorage:
    """
    S3Clientを利用して、このアプリケーション固有のストレージ操作を行うクラス。
    """

    def __init__(self, s3_client: S3Client, data_bucket: str, result_bucket: str):
        """
        TaskStorageを初期化します。

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

    def get_task_image_paths(self, task_name: str) -> List[str]:
        """
        タスク名から、処理対象となる画像ファイルのパスリストを取得します。

        Raises:
            StorageError: オブジェクトの一覧取得に失敗した場合。
        """
        prefix = f"{task_name}/"
        try:
            # 汎用メソッドを呼び出して、プレフィックスに一致する全オブジェクトを取得
            all_objects = self.client.list_objects(self.data_bucket, prefix=prefix)

            # アプリケーション固有のロジック: 画像ファイルのみをフィルタリングする
            image_paths = [
                path
                for path in all_objects
                if path.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            logger.info(f"Found {len(image_paths)} images for task '{task_name}'.")
            return image_paths
        except StorageError as e:
            logger.error(f"Could not get image paths for task '{task_name}'.")
            raise  # 例外を再度送出して、呼び出し元に失敗を伝える

    def get_image_bytes(self, image_path: str) -> bytes:
        """
        単一の画像パスからバイトデータを取得します。

        Raises:
            ObjectNotFoundError: 画像が見つからない場合。
            StorageError: その他の取得エラーが発生した場合。
        """
        # 単純な呼び出しの委譲
        return self.client.get_object(self.data_bucket, image_path)

    def upload_result_df(self, task_name: str, hostname: str, result_df: pd.DataFrame):
        """
        計測結果のDataFrameを受け取り、CSVとしてアップロードします。

        Raises:
            StorageError: アップロードに失敗した場合。
        """
        # アプリケーション固有のロジック: オブジェクト名を生成
        object_name = f"results/{task_name}_{hostname}.csv"

        # アプリケーション固有のロジック: DataFrameをCSV(bytes)に変換
        csv_string = result_df.to_csv(index=False)
        csv_bytes = csv_string.encode("utf-8")

        try:
            # 汎用メソッドを呼び出してアップロード
            self.client.put_object(
                bucket_name=self.result_bucket,
                object_name=object_name,
                data=csv_bytes,
                content_type="text/csv",
            )
        except StorageError as e:
            logger.error(f"Failed to upload result for task '{task_name}'.")
            raise
