import io
import logging
import time
from functools import wraps

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from PIL import Image, UnidentifiedImageError

from library.aws.error import ImageDecodeError, ObjectNotFoundError, StorageError

from .s3_object import S3Object

logger = logging.getLogger(__name__)


def _retry_on_storage_error(max_retries: int = 2, delay_sec: float = 1.0):
    """
    StorageErrorが発生した場合に処理をリトライするデコレーター

    Args:
        max_retries (int): 最大リトライ回数
        delay_sec (float): リトライ間の待機時間（sec）
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except StorageError as e:
                    last_exception = e
                    logger.warning(
                        f"StorageError in '{func.__name__}'. \
                            Retrying in {delay_sec}s... ({attempt + 1}/{max_retries})",
                        extra={
                            "error": str(e),
                            "traceback": e.__traceback__,
                        },
                    )
                    time.sleep(delay_sec)

            # 最後のリトライでも失敗した場合
            logger.error(
                f"'{func.__name__}' failed after {max_retries} retries.",
            )
            raise last_exception

        return wrapper

    return decorator


class S3Client:
    """
    S3にアクセスするためのクライアントクラス
    """

    def __init__(self, endpoint_url: str, access_key: str, secret_key: str):
        """
        S3クライアントを初期化する

        Args:
            endpoint_url (str): S3エンドポイントURL
            access_key (str): S3アクセスキー
            secret_key (str): S3シークレットキー
        """
        try:
            self.client = boto3.client(
                "s3",
                endpoint_url=f"http://{endpoint_url}",
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                config=Config(signature_version="s3v4"),
            )
            logger.info(f"S3Client initialized for endpoint: {endpoint_url}")
        except Exception as e:
            logger.error(
                "Failed to initialize S3Client",
                extra={
                    "endpoint_url": endpoint_url,
                    "error": str(e),
                },
                exc_info=True,
            )
            raise

    @_retry_on_storage_error()
    def list_object_keys(self, bucket_name: str, prefix: str = "") -> list[str]:
        """
        指定されたバケット内のオブジェクトキーを一覧取得する
        末尾が'/'のディレクトリを示すオブジェクトは除外する

        Args:
            bucket_name (str): 取得するバケット名
            prefix (str): 取得するオブジェクトの前方一致パターン

        Returns:
            list[str]: 取得したオブジェクトのキーのリスト

        Raises:
            StorageError: リトライしてもS3関連エラーが解消しなかった場合
            Exception: その他の予期せぬ例外
        """
        object_keys = []
        paginator = self.client.get_paginator("list_objects_v2")
        try:
            pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

            object_keys = [
                obj["Key"]
                for page in pages
                for obj in page.get("Contents", [])
                if not obj["Key"].endswith("/")  # ディレクトリ除外
            ]

            logger.info(
                f"Found {len(object_keys)} objects"
                f"with prefix '{prefix}' in bucket '{bucket_name}'.",
            )
            return object_keys
        except ClientError as e:
            logger.warning(
                f"Failed to list objects in bucket '{bucket_name}'"
                f" with prefix '{prefix}'",
                extra={
                    "error": str(e),
                },
                exc_info=True,
            )
            raise StorageError from e
        except Exception as e:
            logger.warning(
                f"Unexpected error listing objects in bucket '{bucket_name}'",
                extra={
                    "error": str(e),
                },
                exc_info=True,
            )
            raise

    @_retry_on_storage_error()
    def get_object(self, bucket_name: str, object_key: str) -> S3Object:
        """
        指定されたバケットからオブジェクトを取得し、メタデータと共に返す

        Args:
            bucket_name (str): 取得するバケット名
            object_key (str): 取得するオブジェクトキー

        Returns:
            S3Object: 取得したオブジェクトの内容とメタデータ

        Raises:
            ObjectNotFoundError: 指定されたオブジェクトが見つからなかった場合
            StorageError: リトライしてもS3関連エラーが解消しなかった場合
            Exception: その他の予期せぬ例外
        """
        try:
            response = self.client.get_object(Bucket=bucket_name, Key=object_key)
            return S3Object(
                key=object_key,
                content=response["Body"].read(),
                content_type=response.get("ContentType", ""),
                content_length=response.get("ContentLength", 0),
                metadata=response.get("Metadata", {}),
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                logger.error(
                    "Object not found in bucket.",
                    extra={
                        "object_name": object_key,
                        "bucket_name": bucket_name,
                        "error": str(e),
                    },
                    exc_info=True,
                )
                raise ObjectNotFoundError
            else:
                logger.error(
                    "An unexpected S3 error occurred.",
                    extra={
                        "object_name": object_key,
                        "bucket_name": bucket_name,
                        "error": str(e),
                    },
                    exc_info=True,
                )
                raise StorageError from e
        except Exception as e:
            logger.error(
                "Unexpected error getting object",
                extra={
                    "object_name": object_key,
                    "bucket_name": bucket_name,
                    "error": str(e),
                },
                exc_info=True,
            )
            raise

    @_retry_on_storage_error()
    def put_object(
        self,
        bucket_name: str,
        object_key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ):
        """
        指定されたバケットにオブジェクトをアップロードする

        Args:
            bucket_name (str): アップロードするバケット名
            object_key (str): アップロードするオブジェクト名
            data (bytes): アップロードするデータ
            content_type (str): アップロードするオブジェクトのコンテンツタイプ

        Raises:
            StorageError: その他のS3関連エラーが発生した場合
        """
        try:
            self.client.put_object(
                Bucket=bucket_name,
                Key=object_key,
                Body=data,
                ContentType=content_type,
            )
            logger.info(f"Successfully uploaded object to '{bucket_name}/{object_key}'")
        except ClientError as e:
            logger.error(
                f"Failed to upload object to '{bucket_name}/{object_key}'",
                extra={
                    "error": str(e),
                },
                exc_info=True,
            )
            raise StorageError from e
        except Exception as e:
            logger.error(
                "Unexpected error uploading object",
                extra={
                    "object_name": object_key,
                    "bucket_name": bucket_name,
                    "error": str(e),
                },
                exc_info=True,
            )
            raise

    def get_image(self, bucket_name: str, object_key: str) -> Image.Image:
        """
        S3からオブジェクトを取得し、PIL Imageオブジェクトとしてデコードする
        画像は自動的にRGB形式に変換される

        Args:
            bucket_name (str): バケット名
            object_key (str): オブジェクトキー

        Returns:
            Image.Image: PILのImageオブジェクト

        Raises:
            ObjectNotFoundError: オブジェクトが見つからない場合
            ImageDecodeError: オブジェクトが有効な画像でない場合
        """
        s3_obj = self.get_object(bucket_name, object_key)
        try:
            # bytesをファイルライクオブジェクトに変換して画像として開く
            image_bytes = s3_obj.content
            image = Image.open(io.BytesIO(image_bytes))

            # 常にRGB形式で扱うために変換して返す
            return image.convert("RGB")
        except UnidentifiedImageError as e:
            logger.error(
                "Failed to decode object as an image.",
                extra={
                    "bucket": bucket_name,
                    "key": object_key,
                    "error": str(e),
                },
                exc_info=True,
            )
            raise ImageDecodeError from e
