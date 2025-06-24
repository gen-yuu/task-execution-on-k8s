import cpuinfo
import distro
import psutil

try:
    import pynvml
except ImportError:
    pynvml = None
import logging
import platform
import socket
import traceback

import torch
import torchvision
from ultralytics import YOLO  # type: ignore

logger = logging.getLogger(__name__)


def get_system_info(device: torch.device) -> dict:
    """
    実行環境のハードウェア・ソフトウェア情報を収集する

    Args:
        device (torch.device): 使用するデバイス

    Returns:
        dict[str, Any]: 収集した情報
            キーと値の例:
            - "os_platform" (str): OSプラットフォーム (e.g., "Linux")
            - "os_distro" (str, optional): Linuxディストリビューション名
            - "kernel_version" (str): カーネルのバージョン
            - "architecture" (str): CPUアーキテクチャ
            - "hostname" (str): ホスト名
            - "ip_address" (str): IPアドレス
            - "cpu_model" (str): CPUモデル名
            - "cpu_cores_physical" (int): 物理CPUコア数
            - "cpu_cores_logical" (int): 論理CPUコア数
            - "cpu_max_freq_mhz" (str): CPUの最大周波数
            - "ram_total_gb" (float): 合計RAM (GB)
            - "python_version" (str): Pythonのバージョン
            - "pytorch_version" (str): PyTorchのバージョン
            - "torchvision_version" (str): Torchvisionのバージョン
            - "torch_cuda_version" (str, optional): PyTorchが使用するCUDAバージョン
            - "torch_cudnn_version" (int, optional): cuDNNバージョン
            - "gpu_model" (str, optional): GPUモデル名
            - "gpu_driver_version" (str, optional): NVIDIAドライバのバージョン
            - "gpu_vram_total_gb" (float, optional): 合計VRAM (GB)
    """
    logger.info(
        "Collecting system information",
        extra={"device": str(device)},
    )
    info = {}

    # OS / Kernel / Platform Info
    logger.info("OS / Kernel / Platform Info")
    info["os_platform"] = platform.system()
    if info["os_platform"] == "Linux":
        info["os_distro"] = distro.name(pretty=True)
    info["kernel_version"] = platform.release()
    info["architecture"] = platform.machine()

    # IP Address
    logger.info("IP Address")
    try:
        hostname = socket.gethostname()
        info["hostname"] = hostname
        info["ip_address"] = socket.gethostbyname(hostname)
    except Exception:
        logger.warning("Failed to get hostname or IP address.")
        info["hostname"] = "unknown"
        info["ip_address"] = "unknown"

    # CPU Info
    logger.info("CPU Info")
    cpu_info = cpuinfo.get_cpu_info()
    info["cpu_model"] = cpu_info.get("brand_raw", "N/A")
    info["cpu_cores_physical"] = psutil.cpu_count(logical=False)
    info["cpu_cores_logical"] = psutil.cpu_count(logical=True)
    info["cpu_max_freq_mhz"] = cpu_info.get("hz_advertised_friendly", "N/A")

    # Memory (RAM) Info
    logger.info("Memory (RAM) Info")
    ram_info = psutil.virtual_memory()
    info["ram_total_gb"] = round(ram_info.total / (1024**3), 2)

    # Python / Library Versions
    logger.info("Python / Library Versions")
    info["python_version"] = platform.python_version()
    info["pytorch_version"] = torch.__version__
    info["torchvision_version"] = torchvision.__version__

    # GPU / CUDA Info
    logger.info("GPU / CUDA Info")
    if device.type == "cuda":
        info["torch_cuda_version"] = torch.version.cuda  # type: ignore
        info["torch_cudnn_version"] = torch.backends.cudnn.version()
        if pynvml:
            logger.info("GPU Info via pynvml")
            handle = None
            try:

                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(device.index)
                info["gpu_model"] = pynvml.nvmlDeviceGetName(handle)
                info["gpu_driver_version"] = pynvml.nvmlSystemGetDriverVersion()
                gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                info["gpu_vram_total_gb"] = round(float(gpu_mem.total) / (1024**3), 2)
            except Exception as e:
                logger.warning(
                    "Failed to get GPU info via pynvml",
                    extra={
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    },
                )
                # pynvmlが失敗した場合でもPyTorchのAPIでFalback
                if "gpu_model" not in info:
                    info["gpu_model"] = torch.cuda.get_device_name(device)
            finally:
                if handle is not None:
                    pynvml.nvmlShutdown()
        else:
            logger.info("GPU Info via PyTorch")
            info["gpu_model"] = torch.cuda.get_device_name(device)  # Fallback

    logger.info(
        "System Information collected.",
        extra=info,
    )
    return info


_TORCHVISION_MODELS = {
    "faster_rcnn": (
        torchvision.models.detection.fasterrcnn_resnet50_fpn_v2,
        torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
    ),
    "ssd300_vgg16": (
        torchvision.models.detection.ssd300_vgg16,
        torchvision.models.detection.SSD300_VGG16_Weights.DEFAULT,
    ),
}


def load_model(model_name: str):
    """
    事前学習済みモデルと transform を返すファクトリ

    Args:
        model_name (str): モデル名。
            例: "faster_rcnn" / "ssd300_vgg16" / "yolov8s" など
    """
    logger.info(f"Creating model: {model_name}")

    if model_name in _TORCHVISION_MODELS:
        builder, default_weight = _TORCHVISION_MODELS[model_name]
        model = builder(weights=default_weight)
        transform = default_weight.transforms()
    elif model_name.startswith("yolov8"):
        model = YOLO(model_name)
        transform = None
    else:
        logger.error(f"Unknown model name:{model_name} ")
        raise ValueError

    logger.info("Model created.")
    return model, transform


def sanitize_for_path(name: str) -> str:
    """
    パス名として安全な文字列に変換するヘルパー関数
    """
    import re

    # モデル名から不要な部分を削除・簡略化
    name = name.replace("(R)", "").replace("(TM)", "").replace("NVIDIA", "").strip()
    # 英数字、ハイフン、アンダースコア以外をハイフンに置換
    sanitized = re.sub(r"[^a-zA-Z0-9_-]+", "-", name)
    # 連続するハイフンや、先頭・末尾のハイフンを整理
    sanitized = re.sub(r"-+", "-", sanitized).strip("-")
    return sanitized
