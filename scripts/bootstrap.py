"""Instala YOLOv5 v7.0 no diretório local e obtém o checkpoint oficial."""
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "915bbf294bb74c859f0b41f1c23bc395014ea679"
target = ROOT / "vendor/yolov5"
if not target.exists():
    target.parent.mkdir(exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", "--branch", "v7.0",
                    "https://github.com/ultralytics/yolov5.git", str(target)], check=True)
actual = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
assert actual == COMMIT, f"Versão YOLOv5 inesperada: {actual}"
weights = ROOT / "models/yolov5n.pt"
weights.parent.mkdir(exist_ok=True)
if not weights.exists():
    urllib.request.urlretrieve("https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt", weights)
print(f"YOLOv5 v7.0 verificado: {actual}; checkpoint: {weights.name}")
