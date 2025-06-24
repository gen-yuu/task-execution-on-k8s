import logging

import torch

from common.utils import load_model
from run_single_task.feature_extractor import ModelFeatureExtractor

# ロガーをセットアップ

logger = logging.getLogger(__name__)


def check_features():
    """
    モデルの特徴量をチェックする
    """
    # テストしたいモデルのリスト
    model_names = ["faster_rcnn", "ssd300_vgg16", "yolov8s"]

    # GPUが利用可能かチェック
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 適当な入力解像度をダミーとして用意
    # 実際の解像度はデータ特徴量から取得するが、ここでは仮置き
    dummy_resolution = (3, 640, 640)

    for name in model_names:
        logger.info(f"Analyzing model: {name}")
        try:
            # モデルを生成
            model, transform = load_model(name)

            # 特徴量抽出器を初期化
            extractor = ModelFeatureExtractor(model, dummy_resolution, device)

            # 特徴量を抽出して表示
            features = extractor.extract_features()

            logger.info(f"Features for {name}:")
            for key, value in features.items():
                logger.info(f"  {key}: {value}")

        except Exception as e:
            logger.error(
                f"Failed to analyze model {name}",
                extra={
                    "error": str(e),
                },
                exc_info=True,
            )


if __name__ == "__main__":
    check_features()
