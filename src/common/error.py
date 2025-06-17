class StorageError(Exception):
    """ストレージ操作に関する基底例外"""

    pass


class ObjectNotFoundError(StorageError):
    """オブジェクトが見つからなかった場合の例外"""

    pass
