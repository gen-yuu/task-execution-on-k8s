# ===== src/run_single_task/debug_utils.py =====
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2  # type: ignore
import numpy as np
from PIL import Image


def load_class_names(filepath: str) -> list[str]:
    """
    クラス名が記述されたテキストファイルを読み込み、リストとして返す。
    """
    with open(filepath, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names


COCO_CLASSES = load_class_names("./coco_classes.txt")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_annotated_image(
    img: bytes | np.ndarray | Image.Image,
    detections: Dict[str, List[Any]],
    path: Path,
    score_threshold: float = 0.5,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> None:
    """
    スコアのしきい値でフィルタリングし、クラス名を表示できるアノテータ。
    """
    class_names = COCO_CLASSES
    # --- 入力画像の形式をOpenCVで扱える形式(BGRのNumpy配列)に統一 ---
    if isinstance(img, bytes):
        img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
    elif isinstance(img, Image.Image):
        img = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

    # --- ここが最重要：スコアに基づいて検出結果をフィルタリング ---
    filtered_results = []
    for box, score, cls in zip(
        detections["boxes"], detections["scores"], detections["labels"]
    ):
        if score >= score_threshold:
            filtered_results.append((box, score, cls))

    # --- フィルタリングされた結果のみを描画 ---
    for box, score, cls in filtered_results:
        x1, y1, x2, y2 = map(int, box)

        # クラスIDをクラス名に変換（class_namesが提供されていれば）
        if class_names and int(cls) < len(class_names):
            label = class_names[int(cls)]
        else:
            label = str(int(cls))  # 提供がなければIDをそのまま表示

        # ボックス（矩形）を描画
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # ラベルとスコアを描画
        text = f"{label}: {score:.2f}"
        cv2.putText(
            img,
            text,
            (x1, max(y1 - 5, 10)),  # ボックスの少し上に表示
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            lineType=cv2.LINE_AA,
        )

    # 画像をファイルに保存
    cv2.imwrite(str(path), img)
