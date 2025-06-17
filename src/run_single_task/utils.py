import cpuinfo
import distro
import psutil

try:
    import pynvml
except ImportError:
    pynvml = None
import platform
import socket
import traceback

import torch
import torchvision
from ultralytics import YOLO

from common.logger import setup_logger

logger = setup_logger(__name__)


def get_system_info(device: torch.device) -> dict:
    """
    実行環境のハードウェア・ソフトウェア情報を収集する
    """
    logger.info("Collecting system information...")
    info = {}

    # OS / Kernel / Platform Info
    info["os_platform"] = platform.system()
    if info["os_platform"] == "Linux":
        info["os_distro"] = distro.name(pretty=True)
    info["kernel_version"] = platform.release()
    info["architecture"] = platform.machine()

    # IP Address
    try:
        hostname = socket.gethostname()
        info["hostname"] = hostname
        info["ip_address"] = socket.gethostbyname(hostname)
    except Exception:
        logger.warning("Failed to get hostname or IP address.")
        info["hostname"] = "unknown"
        info["ip_address"] = "unknown"

    # CPU Info
    cpu_info = cpuinfo.get_cpu_info()
    info["cpu_model"] = cpu_info.get("brand_raw", "N/A")
    info["cpu_cores_physical"] = psutil.cpu_count(logical=False)
    info["cpu_cores_logical"] = psutil.cpu_count(logical=True)
    info["cpu_max_freq_mhz"] = cpu_info.get("hz_advertised_friendly", "N/A")

    # Memory (RAM) Info
    ram_info = psutil.virtual_memory()
    info["ram_total_gb"] = round(ram_info.total / (1024**3), 2)

    # Python / Library Versions
    info["python_version"] = platform.python_version()
    info["pytorch_version"] = torch.__version__
    info["torchvision_version"] = torchvision.__version__

    # GPU / CUDA Info
    if device.type == "cuda":
        info["torch_cuda_version"] = torch.version.cuda  # type: ignore
        info["torch_cudnn_version"] = torch.backends.cudnn.version()
        if pynvml:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(device.index)
                info["gpu_model"] = pynvml.nvmlDeviceGetName(handle)
                info["gpu_driver_version"] = pynvml.nvmlSystemGetDriverVersion()
                gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                info["gpu_vram_total_gb"] = round(float(gpu_mem.total) / (1024**3), 2)
                pynvml.nvmlShutdown()
            except Exception as e:
                logger.warning(
                    "Failed to get GPU info",
                    extra={"error": str(e), "traceback": traceback.format_exc()},
                )
                info["gpu_model"] = torch.cuda.get_device_name(device)  # Fallback
        else:
            info["gpu_model"] = torch.cuda.get_device_name(device)  # Fallback

    logger.info("System Information collected.", extra={"system_info": info})
    return info


def create_model(model_name: str, device: torch.device) -> tuple:
    """
    モデル名に基づいて、事前学習済みのモデルと対応するtransformを返すファクトリー関数

    Args:
        model_name (str): モデル名
        device (torch.device): 使用するデバイス

    Returns:
        tuple: モデルと対応するtransform
    """
    logger.info(f"Creating model: {model_name}")
    transform = None  # デフォルトはNone

    if model_name == "faster_rcnn":
        weights = (
            torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        )
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)
        transform = weights.transforms()
    elif model_name == "ssd300_vgg16":
        weights = torchvision.models.detection.SSD300_VGG16_Weights.DEFAULT
        model = torchvision.models.detection.ssd300_vgg16(weights=weights)
        transform = weights.transforms()
    elif model_name == "yolov8s":
        model = YOLO("yolov8s.pt")  # .ptファイルをダウンロード・ロード
        # YOLOv8の場合、transformは推論時に自動で適用されるため不要
    else:
        logger.error(f"Unknown model name provided: {model_name}")
        raise ValueError(f"Unknown model name: {model_name}")

    # YOLOv8のモデルオブジェクトは to(device) メソッドを持つ
    model.to(device)

    # eval()モードに設定 (YOLOクラスは内部で適切に処理)
    if hasattr(model, "eval"):
        model.eval()

    logger.info(f"Model '{model_name}' created and moved to {device}.")
    return model, transform
