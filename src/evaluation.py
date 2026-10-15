"""Avaliação comum: detecção por IoU e classificação da imagem com abstinência."""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .dataset import ROOT, audit
from .experiments import SmallCNN, cnn_transforms, device_name, seed_all

sys.path.insert(0, str(ROOT / "vendor/yolov5"))
os.environ.setdefault("YOLOv5_AUTOINSTALL", "false")
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes, xywh2xyxy
from utils.metrics import ap_per_class
from val import process_batch

NAMES = ["gato", "carro"]
CONFIDENCE = 0.25


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def wilson(correct, n):
    """Intervalo de Wilson 95%, apropriado à amostra pequena."""
    z = 1.96
    p = correct / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    radius = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1+z*z/n)
    return [float(center-radius), float(center+radius)]


def classification_summary(truth, predicted):
    n = len(truth)
    hits = sum(int(t == p) for t, p in zip(truth, predicted))
    return {"image_accuracy": float(accuracy_score(truth, predicted)), "correct_images": hits,
            "image_count": n, "error_rate": 1-hits/n,
            "macro_f1": float(f1_score(truth, predicted, labels=[0, 1], average="macro", zero_division=0)),
            "abstentions": sum(p == -1 for p in predicted), "accuracy_ci95_wilson": wilson(hits, n),
            "confusion_labels": ["gato", "carro", "sem_deteccao"],
            "confusion_matrix": confusion_matrix(truth, predicted, labels=[0, 1, -1]).tolist()}


@torch.inference_mode()
def evaluate_yolo(name, weights, split, baseline=False):
    device = torch.device(device_name())
    model = DetectMultiBackend(str(weights), device=device, fp16=False)
    model.eval()
    model.warmup(imgsz=(1, 3, 416, 416))
    manifest = pd.read_csv(ROOT / "data/manifest.csv")
    rows = manifest[manifest.split == split]
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    stats, truth, predicted, timings, details = [], [], [], [], []
    tp_total = fp_total = gt_total = 0
    per_class = {i: {"tp": 0, "fp": 0, "gt": 0} for i in (0, 1)}

    def infer(path):
        # Timer inclui leitura local, pré-processamento, rede e NMS; exclui gravação.
        original = cv2.imread(str(path))
        array = letterbox(original, 416, stride=32, auto=False)[0]
        tensor = torch.from_numpy(np.ascontiguousarray(array.transpose((2, 0, 1))[::-1])).to(device).float() / 255
        tensor = tensor.unsqueeze(0)
        out = model(tensor)
        pred = non_max_suppression(out, conf_thres=0.001, iou_thres=0.45,
                                   classes=[15, 2] if baseline else None, max_det=100)[0]
        if len(pred):
            pred[:, :4] = scale_boxes(tensor.shape[2:], pred[:, :4], original.shape)
            if baseline:
                pred[:, 5] = torch.where(pred[:, 5] == 15, 0, 1)
        return original, pred

    # Aquecimento adicional com uma imagem de validação, inclusive no teste.
    warmup_path = ROOT / manifest[manifest.split == "val"].iloc[0]["file"]
    for _ in range(5):
        infer(warmup_path)
    for _, row in rows.iterrows():
        path = ROOT / row["file"]
        sync()
        t = time.perf_counter()
        original, pred = infer(path)
        sync()
        timings.append((time.perf_counter()-t)*1000)
        # Mais quatro passagens por imagem reduzem a influência de uma só medição.
        for _ in range(4):
            sync()
            t = time.perf_counter()
            infer(path)
            sync()
            timings.append((time.perf_counter()-t)*1000)
        label_path = ROOT / "data/labels" / split / f"{path.stem}.txt"
        labels = torch.tensor(np.loadtxt(label_path, ndmin=2), device=device, dtype=torch.float32)
        boxes = xywh2xyxy(labels[:, 1:5])
        boxes *= torch.tensor([original.shape[1], original.shape[0]]*2, device=device)
        gt = torch.cat((labels[:, 0:1], boxes), 1)
        correct = process_batch(pred, gt, iouv)
        stats.append((correct.cpu().numpy(), pred[:, 4].cpu().numpy(), pred[:, 5].cpu().numpy(), gt[:, 0].cpu().numpy()))
        high = pred[pred[:, 4] >= CONFIDENCE]
        correct_high = process_batch(high, gt, iouv)
        tp = int(correct_high[:, 0].sum())
        tp_total += tp
        fp_total += len(high)-tp
        gt_total += len(gt)
        for cls in (0, 1):
            cls_pred = high[:, 5] == cls
            cls_tp = int(correct_high[cls_pred, 0].sum())
            per_class[cls]["tp"] += cls_tp
            per_class[cls]["fp"] += int(cls_pred.sum())-cls_tp
            per_class[cls]["gt"] += int((gt[:, 0] == cls).sum())
        winner = int(high[high[:, 4].argmax(), 5]) if len(high) else -1
        truth.append(int(row.class_id))
        predicted.append(winner)
        record = {"file": str(path.relative_to(ROOT)), "true_class": int(row.class_id),
                  "image_prediction": winner, "detections": high.cpu().tolist()}
        details.append(record)
        canvas = original.copy()
        for target in gt.cpu().numpy():
            x1, y1, x2, y2 = target[1:].astype(int)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (60, 180, 60), 2)
        for detection in high.cpu().numpy():
            x1, y1, x2, y2 = detection[:4].astype(int)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 130, 255), 2)
            cv2.putText(canvas, f"{NAMES[int(detection[5])]} {detection[4]:.2f}",
                        (x1, max(16, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 255), 2)
        output = ROOT / "results/predictions" / name / split / path.name
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), canvas)
    combined = [np.concatenate([s[i] for s in stats], 0) for i in range(4)]
    ap = np.zeros((2, 10))
    ap_classes = np.array([0, 1])
    if combined[0].any():
        _, _, _, _, _, ap, ap_classes = ap_per_class(*combined, plot=False, names=dict(enumerate(NAMES)))
    for idx, cls in enumerate(ap_classes):
        per_class[int(cls)]["map50"] = float(ap[idx, 0])
        per_class[int(cls)]["map50_95"] = float(ap[idx].mean())
    for counts in per_class.values():
        counts["precision_at025"] = counts["tp"]/max(counts["tp"]+counts["fp"], 1)
        counts["recall_at025"] = counts["tp"]/max(counts["gt"], 1)
    result = {"model": name, "split": split, "task": "detection", "map50": float(ap[:, 0].mean()),
              "map50_95": float(ap.mean()), "precision_at025": tp_total/max(tp_total+fp_total, 1),
              "recall_at025": tp_total/max(gt_total, 1), "tp": tp_total, "fp": fp_total,
              "fn": gt_total-tp_total, "gt_boxes": gt_total, "per_class": per_class,
              "latency_ms_median": float(np.median(timings)), "latency_ms_p95": float(np.percentile(timings, 95)),
              "latency_samples": len(timings), "imgsz": 416, "confidence": CONFIDENCE,
              "ap_confidence_min": 0.001, "nms_iou": 0.45, **classification_summary(truth, predicted)}
    (ROOT / f"results/{name}_{split}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    (ROOT / f"results/{name}_{split}_predictions.json").write_text(json.dumps(details, indent=2))
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    del model
    torch.cuda.empty_cache()
    return result


@torch.inference_mode()
def evaluate_cnn(split):
    device = device_name()
    checkpoint = torch.load(ROOT / "models/cnn_best.pt", map_location=device, weights_only=True)
    model = SmallCNN().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    transform = cnn_transforms()
    idx_to_project = {idx: NAMES.index(name) for name, idx in checkpoint["class_to_idx"].items()}
    data = pd.read_csv(ROOT / "data/manifest.csv")

    def infer(path):
        with Image.open(path) as im:
            tensor = transform(im.convert("RGB")).unsqueeze(0).to(device)
        return model(tensor)

    warmup = ROOT / data[data.split == "val"].iloc[0]["file"]
    for _ in range(5):
        infer(warmup)
    truth, predicted, timings, details = [], [], [], []
    for _, row in data[data.split == split].iterrows():
        path = ROOT / row["file"]
        for _ in range(5):
            sync()
            t = time.perf_counter()
            logits = infer(path)
            sync()
            timings.append((time.perf_counter()-t)*1000)
        prediction = idx_to_project[int(logits.argmax(1).item())]
        truth.append(int(row.class_id))
        predicted.append(prediction)
        details.append({"file": row["file"], "true_class": int(row.class_id), "image_prediction": prediction,
                        "probability": float(logits.softmax(1).max())})
    result = {"model": "cnn_zero", "split": split, "task": "classification", "imgsz": 128,
              "latency_ms_median": float(np.median(timings)), "latency_ms_p95": float(np.percentile(timings, 95)),
              "latency_samples": len(timings), **classification_summary(truth, predicted)}
    (ROOT / f"results/cnn_zero_{split}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    (ROOT / f"results/cnn_zero_{split}_predictions.json").write_text(json.dumps(details, indent=2))
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result


def evaluate_all():
    audit()
    seed_all()
    all_results = []
    variants = [("yolo_padrao", ROOT / "models/yolov5n.pt", True),
                ("yolov5n_30ep", ROOT / "models/yolov5n_30ep_best.pt", False),
                ("yolov5n_60ep", ROOT / "models/yolov5n_60ep_best.pt", False)]
    # Validação primeiro; escolha do modelo antes de acessar o teste.
    for name, weights, baseline in variants:
        all_results.append(evaluate_yolo(name, weights, "val", baseline))
    all_results.append(evaluate_cnn("val"))
    customized = [r for r in all_results if r["model"].startswith("yolov5n_")]
    selected = max(customized, key=lambda r: r["map50_95"])
    (ROOT / "results/model_selection.json").write_text(json.dumps({"selected_model": selected["model"],
        "criterion": "maior mAP50-95 na validação; teste não participa da escolha",
        "validation_map50_95": selected["map50_95"]}, indent=2, ensure_ascii=False))
    for name, weights, baseline in variants:
        all_results.append(evaluate_yolo(name, weights, "test", baseline))
    all_results.append(evaluate_cnn("test"))
    rows = []
    for result in all_results:
        train_file = ROOT / ("results/cnn_train.json" if result["model"] == "cnn_zero"
                            else f"results/{result['model']}_train.json")
        train = json.loads(train_file.read_text()) if train_file.exists() else {"training_seconds": 0, "epochs": 0}
        rows.append({**{k: v for k, v in result.items() if not isinstance(v, (list, dict))},
                     "training_seconds": train["training_seconds"], "epochs": train["epochs"]})
    pd.DataFrame(rows).to_csv(ROOT / "results/comparison.csv", index=False)
    environment = {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform(),
                   "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   "yolov5_commit": "915bbf294bb74c859f0b41f1c23bc395014ea679", "precision": "FP32 para avaliação",
                   "latency_scope": "leitura local + pré-processamento + rede + NMS (YOLO), batch 1, 5 aquecimentos + 5 repetições/imagem"}
    (ROOT / "results/environment.json").write_text(json.dumps(environment, indent=2, ensure_ascii=False))
    create_figures()
    return pd.DataFrame(rows)


def create_figures():
    """Figuras científicas estáticas com todas as oito imagens de teste."""
    plt.rcParams.update({"font.size": 10, "figure.dpi": 130})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for epochs, color in ((30, "#1565a9"), (60, "#d2691e")):
        df = pd.read_csv(ROOT / f"results/yolov5n_{epochs}ep_history.csv")
        axes[0].plot(df.epoch+1, df["metrics/mAP_0.5:0.95"], label=f"{epochs} épocas", color=color)
        axes[1].plot(df.epoch+1, df["train/box_loss"], label=f"{epochs} épocas", color=color)
        axes[2].plot(df.epoch+1, df["val/box_loss"], label=f"{epochs} épocas", color=color)
    for ax, title in zip(axes, ("mAP50–95 • validação", "Loss de caixas • treino", "Loss de caixas • validação")):
        ax.set(title=title, xlabel="Época")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.savefig(ROOT / "results/yolo_learning_curves.png")
    plt.close(fig)
    df = pd.read_csv(ROOT / "results/cnn_history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for split, label in (("train", "Treino"), ("val", "Validação")):
        axes[0].plot(df.epoch, df[f"{split}_loss"], label=label)
        axes[1].plot(df.epoch, df[f"{split}_accuracy"], label=label)
    for ax, title in zip(axes, ("CNN • entropia cruzada", "CNN • acurácia por imagem")):
        ax.set(title=title, xlabel="Época")
        ax.legend()
        ax.grid(alpha=0.2)
    fig.savefig(ROOT / "results/cnn_learning_curves.png")
    plt.close(fig)
    for name in ("yolo_padrao", "yolov5n_30ep", "yolov5n_60ep"):
        files = sorted((ROOT / "results/predictions" / name / "test").glob("*.jpg"))
        fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
        for ax, path in zip(axes.flat, files):
            ax.imshow(Image.open(path))
            ax.set_title(path.stem, fontsize=9)
            ax.axis("off")
        fig.suptitle(f"{name} • todas as imagens de teste | verde: referência, laranja: detecção ≥ 0,25")
        fig.savefig(ROOT / f"results/{name}_test_montage.jpg")
        plt.close(fig)


if __name__ == "__main__":
    print(evaluate_all().to_string(index=False))
