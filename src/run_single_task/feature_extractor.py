from typing import Any

import torch
import torch.nn as nn
from thop import profile

from common.logger import setup_logger

logger = setup_logger(__name__)


class ModelFeatureExtractor:
    """
    モデルの静的特徴量を抽出するクラス
    """

    def __init__(self, model: nn.Module, input_resolution: tuple[int, int, int]):
        """
        モデル特徴量抽出クラスの初期化

        Args:
            model (nn.Module): モデル
            input_resolution (tuple[int, int, int]): 入力画像の解像度
        """
        # YOLOv8のようなラッパーオブジェクトの場合、実際のnn.Moduleは.model属性にある
        self.nn_model = model.model if hasattr(model, "model") else model
        # (チャンネル数, 高さ, 幅) の形式
        self.input_resolution = input_resolution
        self.device = next(model.parameters()).device

    def _analyze_with_fx(self) -> dict[str, int]:
        """
        torch.fxを使ってモデルのグラフを解析し、レイヤー数を数える
        """
        features = {
            "num_conv2d": 0,
            "num_linear": 0,
            "num_batchnorm": 0,
            "num_add_op": 0,
        }
        try:
            # symbolic_traceはモデルによっては失敗することがある
            graph_module = torch.fx.symbolic_trace(self.model)  # type: ignore
            for node in graph_module.graph.nodes:
                if node.op == "call_module":
                    module = dict(self.nn_model.named_modules())[node.target]
                    if isinstance(module, nn.Conv2d):
                        features["num_conv2d"] += 1
                    elif isinstance(module, nn.Linear):
                        features["num_linear"] += 1
                    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                        features["num_batchnorm"] += 1
                elif node.op == "call_function" and "add" in str(node.target):
                    features["num_add_op"] += 1
        except Exception as e:
            logger.warning(f"torch.fx analysis failed: {e}. Returning zero counts.")
        return features

    def _calculate_flops_params(self) -> dict[str, Any]:
        """thopを使ってFLOPsとパラメータ数を計算する"""
        dummy_input = torch.randn(1, *self.input_resolution).to(self.device)
        try:
            # モデルによってはカスタムOpの登録が必要な場合がある
            profile_result = profile(
                self.nn_model, inputs=(dummy_input,), verbose=False
            )
            flops = profile_result[0]
            params = profile_result[1]
            return {"model_flops_g": flops / 1e9, "model_params_m": params / 1e6}
        except Exception as e:
            logger.warning(f"thop profile failed: {e}. Returning zero.")
            return {"model_flops_g": 0, "model_params_m": 0}

    def extract_features(self) -> dict[str, Any]:
        """
        すべてのモデル特徴量を抽出して辞書として返す

        Returns:
            dict[str, Any]: モデル特徴量
        """
        model_features = {}

        # レイヤー数の解析
        # layer_counts = self._analyze_with_fx()
        layer_counts = self._analyze_with_export()
        model_features.update(layer_counts)

        # FLOPsとパラメータ数の計算
        flops_params = self._calculate_flops_params()
        model_features.update(flops_params)

        # メモリサイズとデータ型の取得
        total_memory_bytes = sum(
            p.numel() * p.element_size() for p in self.nn_model.parameters()
        )
        model_features["model_memory_mb"] = total_memory_bytes / (1024**2)

        # 最初のパラメータのデータ型を代表として使う
        input_dtype = next(self.nn_model.parameters()).dtype
        model_features["model_input_dtype_bytes"] = (
            torch.finfo(input_dtype).bits // 8 if input_dtype.is_floating_point else 1
        )

        return model_features
