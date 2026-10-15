"""Treinos YOLOv5 independentes e CNN pequena, sem usar o teste na seleção."""
from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .dataset import ROOT, SEED, audit


def device_name():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(2)


def train_yolo(epochs: int, reuse: bool = True) -> dict:
    """Sempre inicia do mesmo checkpoint COCO; reutiliza só execução concluída."""
    audit()
    name = f"yolov5n_{epochs}ep"
    record = ROOT / f"results/{name}_train.json"
    if record.exists() and reuse:
        print(f"Treino concluído preservado: {name}")
        return json.loads(record.read_text())
    command = [sys.executable, "train.py", "--data", str(ROOT / "data/dataset-local.yaml"),
               "--weights", str(ROOT / "models/yolov5n.pt"), "--img", "416", "--batch-size", "8",
               "--epochs", str(epochs), "--seed", str(SEED), "--workers", "0", "--patience", "0",
               "--device", "0" if torch.cuda.is_available() else "cpu",
               "--project", str(ROOT / "results/runs"), "--name", name, "--exist-ok"]
    env = {**os.environ, "WANDB_MODE": "disabled", "COMET_MODE": "DISABLED", "MPLBACKEND": "Agg",
           "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "YOLOv5_AUTOINSTALL": "false"}
    print(f"Treinando {name}; log em results/{name}.log", flush=True)
    start = time.perf_counter()
    with (ROOT / f"results/{name}.log").open("w") as log:
        subprocess.run(command, cwd=ROOT / "vendor/yolov5", env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    elapsed = time.perf_counter() - start
    run = ROOT / "results/runs" / name
    history = pd.read_csv(run / "results.csv")
    history.columns = history.columns.str.strip()
    assert len(history) == epochs
    history.to_csv(ROOT / f"results/{name}_history.csv", index=False)
    # Critério de fitness oficial YOLOv5: 10% mAP50 + 90% mAP50-95.
    fitness = 0.1 * history["metrics/mAP_0.5"] + 0.9 * history["metrics/mAP_0.5:0.95"]
    best_index = int(fitness.idxmax())
    shutil.copy2(run / "weights/best.pt", ROOT / f"models/{name}_best.pt")
    for filename in ("results.png", "confusion_matrix.png", "PR_curve.png", "opt.yaml", "hyp.yaml"):
        if (run / filename).exists():
            shutil.copy2(run / filename, ROOT / f"results/{name}_{filename}")
    result = {"model": name, "epochs": epochs, "training_seconds": elapsed,
              "best_epoch": best_index + 1, "best_fitness": float(fitness.iloc[best_index]),
              "seed": SEED, "imgsz": 416, "batch": 8,
              "initial_weights": "yolov5n.pt — COCO pré-treinado", "selection_split": "val",
              "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    record.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result


class SmallCNN(nn.Module):
    """Três blocos convolucionais, pooling espacial e cabeça de duas classes."""

    def __init__(self):
        super().__init__()
        layers = []
        channels = 3
        for out in (16, 32, 64):
            layers.extend([nn.Conv2d(channels, out, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)])
            channels = out
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
                                        nn.Linear(64 * 4 * 4, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 2))

    def forward(self, x):
        return self.classifier(self.features(x))


def cnn_transforms(training=False):
    ops = [transforms.Resize((128, 128))]
    if training:
        ops.extend([transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.15, 0.15, 0.15)])
    return transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)])


def train_cnn(epochs=60, reuse=True):
    """Inicialização aleatória; melhor época escolhida apenas pela loss de validação."""
    audit()
    record = ROOT / "results/cnn_train.json"
    if record.exists() and reuse:
        return json.loads(record.read_text())
    seed_all()
    device = device_name()
    model = SmallCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_fn = nn.CrossEntropyLoss()
    sets = {split: datasets.ImageFolder(ROOT / "data/classification" / split, cnn_transforms(split == "train"))
            for split in ("train", "val")}
    loaders = {split: DataLoader(ds, batch_size=8, shuffle=split == "train", num_workers=0,
                                generator=torch.Generator().manual_seed(SEED)) for split, ds in sets.items()}
    history = []
    best_loss = float("inf")
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        row = {"epoch": epoch}
        for split in ("train", "val"):
            model.train(split == "train")
            loss_sum = correct = count = 0
            with torch.set_grad_enabled(split == "train"):
                for images, labels in loaders[split]:
                    images, labels = images.to(device), labels.to(device)
                    logits = model(images)
                    loss = loss_fn(logits, labels)
                    if split == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    loss_sum += loss.item() * len(labels)
                    correct += (logits.argmax(1) == labels).sum().item()
                    count += len(labels)
            row[f"{split}_loss"] = loss_sum / count
            row[f"{split}_accuracy"] = correct / count
        history.append(row)
        if row["val_loss"] < best_loss:
            best_loss, best_epoch = row["val_loss"], epoch
            torch.save({"state_dict": model.state_dict(), "class_to_idx": sets["train"].class_to_idx,
                        "epoch": epoch}, ROOT / "models/cnn_best.pt")
        if epoch % 10 == 0:
            print(row, flush=True)
    elapsed = time.perf_counter() - start
    pd.DataFrame(history).to_csv(ROOT / "results/cnn_history.csv", index=False)
    result = {"model": "cnn_zero", "epochs": epochs, "training_seconds": elapsed,
              "best_epoch": best_epoch, "best_val_loss": best_loss, "seed": SEED,
              "parameters": sum(p.numel() for p in model.parameters()),
              "class_to_idx": sets["train"].class_to_idx, "device": device,
              "selection_split": "val", "pretrained": False}
    record.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=["yolo30", "yolo60", "cnn", "all"])
    args = parser.parse_args()
    if args.target in ("yolo30", "all"):
        train_yolo(30)
    if args.target in ("yolo60", "all"):
        train_yolo(60)
    if args.target in ("cnn", "all"):
        train_cnn()
