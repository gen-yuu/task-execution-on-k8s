import collections
import logging
from typing import Any, Tuple

import torch
import torch.nn as nn
from thop import profile  # type: ignore

logger = logging.getLogger(__name__)

# 解析対象レイヤーの定義
FEATURE_LAYER_TYPES: Tuple[type[nn.Module], ...] = (
    nn.Conv2d,  # 畳み込み層：cuDNN の Winograd／GEMM カーネルで実装
    nn.BatchNorm2d,  # バッチ正規化層：cuDNN の BN 前向きカーネル
    nn.ReLU,  # 活性化層（ReLU）：要素毎演算 vectorized_elementwise_kernel
    nn.SiLU,  # 活性化層（SiLU）：YOLO-v8 等で使われる SiLU カーネル
    nn.Linear,  # 全結合層（線形層）：BLAS/GEMM ベースの行列積カーネル
)

GIGA = 1e9
MEGA = 1e6
MEGABYTE = 1024**2


class ModelFeatureExtractor:
    """
    モデルの静的特徴量を抽出するクラス
    """

    def __init__(
        self,
        model: nn.Module,
        input_resolution: tuple[int, int, int],
    ):
        """
        モデル特徴量抽出クラスの初期化

        Args:
            model (nn.Module): モデル
            input_resolution (tuple[int, int, int]): 入力画像の解像度
        """
        # Ultralytics/YOLOのラッパー対応
        self.nn_model = model.model if hasattr(model, "model") else model
        # (チャンネル数, 高さ, 幅) の形式
        self.input_resolution = input_resolution

    def _calculate_flops_params(self) -> dict[str, Any]:
        """
        thopを使ってFLOPsとParamsを計算する

        Returns:
            dict[str, Any]: FLOPsとパラメータ数
        """
        logger.info("Calculating FLOPs and Params")

        c, h, w = self.input_resolution
        stride = 32
        h_adj = (h // stride) * stride
        w_adj = (w // stride) * stride

        if h_adj == 0 or w_adj == 0:
            adjusted_resolution = self.input_resolution
        else:
            adjusted_resolution = (c, h_adj, w_adj)

        dummy_input = torch.randn(1, *adjusted_resolution)

        logger.info(
            "Using adjusted resolution for FLOPs calculation: %s", adjusted_resolution
        )
        try:
            profile_result = profile(
                self.nn_model, inputs=(dummy_input,), verbose=False
            )
            flops = profile_result[0]
            params = profile_result[1]
            return {
                "model_flops_g": flops / GIGA,
                "model_params_m": params / MEGA,
            }
        except Exception as e:
            logger.warning(
                "thop profile failed. Returning zero.",
                extra={
                    "error": str(e),
                    "traceback": e.__traceback__,
                    "model": self.nn_model.__class__.__name__,
                },
                exc_info=True,
            )
            return {"model_flops_g": 0, "model_params_m": 0}

    def _count_layers(self) -> dict[str, int]:
        """
        FEATURE_LAYER_TYPESに基づいてレイヤー数をカウントし、固定長ベクトルとして辞書化して返す

        Returns:
            dict[str, int]: レイヤー数
        """
        logger.info("Counting layers")
        # 出現数をカウント
        counter = collections.Counter(type(m) for m in self.nn_model.modules())

        # タプル順を保った固定長ベクトル
        layer_counts = {
            f"num_{cls.__name__.lower()}": counter.get(cls, 0)
            for cls in FEATURE_LAYER_TYPES
        }
        return layer_counts

    def extract_features(self) -> dict[str, Any]:
        """
        すべてのモデル特徴量を抽出して辞書として返す

        Returns:
            dict[str, Any]: モデル特徴量
                キーと値の例は以下の通りです。
                    # --- レイヤー数 (int) ---
                    "num_conv2d": 53,
                    "num_linear": 1,
                    "num_batchnorm2d": 53,
                    "num_relu": 28,
                    "num_maxpool2d": 1,
                    "num_adaptiveavgpool2d": 1,
                    "num_upsample": 2,
                    "num_silu": 28,
                    # --- 計算量とパラメータ数 (float) ---
                    "model_flops_g": 41.2,  # GFLOPs (ギガFLOPs) 単位
                    "model_params_m": 25.5, # Million (百万) 単位
                    # --- メモリ使用量 (float) ---
                    "model_memory_mb": 97.5, # MB (メガバイト) 単位
                    # --- データ型のバイト数 (int) ---
                    "model_input_dtype_bytes": 4 # 例: float32なら4, float16なら2
                }
        """
        logger.info("Extracting model features")
        features = {}

        # レイヤー出現数
        features.update(self._count_layers())

        # FLOPs / Params
        features.update(self._calculate_flops_params())

        # メモリ使用量
        total_bytes = sum(
            p.numel() * p.element_size() for p in self.nn_model.parameters()
        )
        features["model_memory_mb"] = total_bytes / MEGABYTE

        # dtype (byte 数): 最初のパラメータのデータ型を代表として使う
        dtype = next(self.nn_model.parameters()).dtype
        if dtype.is_floating_point:
            features["model_input_dtype_bytes"] = torch.finfo(dtype).bits // 8
        else:
            features["model_input_dtype_bytes"] = torch.iinfo(dtype).bits // 8
        logger.info(
            "Extracted model features",
            extra=features,
        )
        return features
