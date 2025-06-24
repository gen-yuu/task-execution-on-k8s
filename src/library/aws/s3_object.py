from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)  # immutable
class S3Object:
    """
    S3から取得したオブジェクトの情報を格納するデータクラス
    """

    key: str
    content: bytes
    content_type: str
    content_length: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def decode(self, encoding: str = "utf-8-sig") -> str:
        """
        コンテンツを文字列としてデコードするヘルパーメソッド
        """
        return self.content.decode(encoding)
