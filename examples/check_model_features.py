# examples/check_model_features.py

import logging

import torch

from src.common.logger import setup_logger
from src.run_single_task.feature_extractor import ModelFeatureExtractor
from src.run_single_task.utils import create_model

# ロガーをセットアップ
setup_logger(__name__)
logger = logging.getLogger(__name__)


def check_features():
    # テストしたいモデルのリスト
    model_names = ["faster_rcnn", "ssd300_vgg16", "yolov8s"]

    # GPUが利用可能かチェック
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 適当な入力解像度をダミーとして用意
    # 実際の解像度はデータ特徴量から取得するが、ここでは仮置き
    dummy_resolution = (3, 640, 640)

    for name in model_names:
        print("-" * 50)
        logger.info(f"Analyzing model: {name}")
        try:
            # 1. モデルを生成
            model, _ = create_model(name, device)

            # 2. 特徴量抽出器を初期化
            extractor = ModelFeatureExtractor(model, dummy_resolution)

            # 3. 特徴量を抽出して表示
            features = extractor.extract_features()

            print(f"Features for {name}:")
            for key, value in features.items():
                print(f"  {key}: {value}")

        except Exception as e:
            logger.error(
                f"Failed to analyze model {name}",
                extra={
                    "error": str(e),
                },
            )


if __name__ == "__main__":
    # プロジェクトをインストール済み(`pip install -e .`)であることが前提
    check_features()
