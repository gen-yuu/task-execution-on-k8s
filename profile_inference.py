from torch.profiler import ProfilerActivity, profile, schedule


def profile_inference(model, sample, warmup=5, iters=20, label="model"):
    """
    Profile `iters` 回の推論を行い
    - Chrome trace: <label>.json
    - 表：CUDA self time 上位 15 行
    """
    sched = schedule(wait=warmup, warmup=0, active=iters, repeat=1)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=sched,
        record_shapes=True,
        profile_memory=True,
        on_trace_ready=lambda p: p.export_chrome_trace(f"{label}.json"),
    ) as prof:

        for _ in range(warmup + iters):
            _ = model(sample)  # forward
            prof.step()  # ★← これを忘れると内部状態が確定しない

    # with ブロックを抜けた時点で profiler.stop() が呼ばれる
    print(f"\n=== {label}  CUDA time top-15 ===")
    print(
        prof.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=15, top_level_events_only=True
        )
    )


def main():
    import torch
    import torchvision.models as tvm

    # ① YOLO-v8 (Ultralytics)
    from ultralytics import YOLO  # type: ignore

    yolo = YOLO("yolov8s.pt").model.cuda().eval()
    dummy = torch.zeros(1, 3, 640, 640, device="cuda")

    # ② SSD300 (torchvision)
    ssd = tvm.detection.ssd300_vgg16(weights="DEFAULT").cuda().eval()
    dummy2 = torch.zeros(1, 3, 300, 300, device="cuda")

    # ③ Faster R-CNN
    faster = tvm.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT").cuda().eval()
    dummy3 = torch.zeros(1, 3, 800, 800, device="cuda")
    profile_inference(yolo, dummy, label="yolov8")
    profile_inference(ssd, dummy2, label="ssd300")
    profile_inference(faster, dummy3, label="fasterrcnn")


if __name__ == "__main__":
    main()
