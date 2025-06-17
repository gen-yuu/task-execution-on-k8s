import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from common.error import ObjectNotFoundError, StorageError
from common.logger import setup_logger

logger = setup_logger(__name__)


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
            )
            raise

    def list_objects(self, bucket_name: str, prefix: str = "") -> list[str]:
        """
        指定されたバケット内のオブジェクトを一覧取得する

        Args:
            bucket_name (str): 取得するバケット名
            prefix (str): 取得するオブジェクトの前方一致パターン

        Returns:
            List[str]: 取得したオブジェクトのキーのリスト

        Raises:
            StorageError: その他のS3関連エラーが発生した場合
        """
        object_keys = []
        paginator = self.client.get_paginator("list_objects_v2")
        try:
            pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
            for page in pages:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        object_keys.append(obj["Key"])
            logger.info(
                f"Found {len(object_keys)} objects with prefix \
                    '{prefix}' in bucket '{bucket_name}'"
            )
        except ClientError as e:
            logger.error(
                f"Failed to list objects in bucket '{bucket_name}'",
                extra={
                    "error": str(e),
                },
            )
            raise StorageError
        except Exception as e:
            logger.error(
                "Unexpected error listing objects in bucket",
                extra={
                    "bucket_name": bucket_name,
                    "error": str(e),
                },
            )
            raise
        return object_keys

    def get_object(self, bucket_name: str, object_name: str) -> bytes:
        """
        指定されたバケットからオブジェクトを取得する

        Args:
            bucket_name (str): 取得するバケット名
            object_name (str): 取得するオブジェクト名

        Returns:
            bytes: 取得したオブジェクトの内容

        Raises:
            ObjectNotFoundError: 指定されたオブジェクトが見つからなかった場合
            StorageError: その他のS3関連エラーが発生した場合
        """
        try:
            response = self.client.get_object(Bucket=bucket_name, Key=object_name)
            return response["Body"].read()
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                logger.warning(
                    "Object '{object_name}' not found in bucket '{bucket_name}'.",
                    extra={
                        "error": str(e),
                    },
                )
                raise ObjectNotFoundError
            else:
                logger.error(
                    "An unexpected S3 error occurred.",
                    extra={
                        "error": str(e),
                    },
                )
                # その他のS3エラーは汎用的なストレージエラーとして送出
                raise StorageError
        except Exception as e:
            logger.error(
                "Unexpected error getting object",
                extra={
                    "object_name": object_name,
                    "bucket_name": bucket_name,
                    "error": str(e),
                },
            )
            raise

    def put_object(
        self,
        bucket_name: str,
        object_name: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ):
        """
        指定されたバケットにオブジェクトをアップロードする

        Args:
            bucket_name (str): アップロードするバケット名
            object_name (str): アップロードするオブジェクト名
            data (bytes): アップロードするデータ
            content_type (str): アップロードするオブジェクトのコンテンツタイプ

        Raises:
            StorageError: その他のS3関連エラーが発生した場合
        """
        try:
            self.client.put_object(
                Bucket=bucket_name, Key=object_name, Body=data, ContentType=content_type
            )
            logger.info(
                f"Successfully uploaded object to '{bucket_name}/{object_name}'"
            )
        except ClientError as e:
            logger.error(
                f"Failed to upload object to '{bucket_name}/{object_name}'",
                extra={"error": str(e)},
            )
            raise StorageError
        except Exception as e:
            logger.error(
                "Unexpected error uploading object",
                extra={
                    "object_name": object_name,
                    "bucket_name": bucket_name,
                    "error": str(e),
                },
            )
            raise
