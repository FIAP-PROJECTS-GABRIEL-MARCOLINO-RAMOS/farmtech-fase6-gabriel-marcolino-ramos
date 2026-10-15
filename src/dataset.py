"""Seleção reproduzível COCO, atribuição de fontes e rótulos YOLO/Pascal VOC."""
from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SEED = 565717
CLASSES = {17: (0, "gato"), 3: (1, "carro")}
S3 = "https://s3.amazonaws.com/images.cocodataset.org"


def download(url: str, dest: Path) -> None:
    """Baixa somente arquivos ausentes, sem desativar a verificação TLS."""
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)


def prepare() -> dict:
    """Seleciona 40 imagens por classe e faz a divisão 32/4/4 antes do treino."""
    manifest = ROOT / "data/manifest.csv"
    if manifest.exists():
        return audit()
    archive = ROOT / ".cache/annotations_trainval2017.zip"
    download(f"{S3}/annotations/annotations_trainval2017.zip", archive)
    with zipfile.ZipFile(archive) as z:
        coco = json.loads(z.read("annotations/instances_val2017.json"))
    annotations = defaultdict(list)
    for ann in coco["annotations"]:
        if ann["category_id"] in CLASSES:
            annotations[ann["image_id"]].append(ann)
    licenses = {item["id"]: item for item in coco["licenses"]}
    rng = random.Random(SEED)
    chosen = []
    for category, (class_id, name) in CLASSES.items():
        candidates = []
        for im in sorted(coco["images"], key=lambda item: item["id"]):
            anns = annotations[im["id"]]
            # Exclusividade entre as duas classes torna a classificação unívoca.
            if {a["category_id"] for a in anns} != {category}:
                continue
            # Imagens CC BY, BY-SA, BY-NC ou BY-NC-SA para uso acadêmico.
            if im["license"] not in (1, 2, 4, 5) or any(a["iscrowd"] for a in anns):
                continue
            largest = max(a["bbox"][2] * a["bbox"][3] for a in anns)
            if largest / (im["width"] * im["height"]) < 0.10:
                continue
            candidates.append(im)
        print(f"Candidatas {name}: {len(candidates)}", flush=True)
        if len(candidates) < 40:
            raise ValueError(f"Apenas {len(candidates)} candidatas para {name}")
        rng.shuffle(candidates)
        for i, im in enumerate(candidates[:40]):
            split = "train" if i < 32 else "val" if i < 36 else "test"
            chosen.append((im, class_id, name, split))

    def build(item):
        im, class_id, name, split = item
        filename = im["file_name"]
        relative = Path("data/images") / split / filename
        target = ROOT / relative
        download(f"{S3}/val2017/{filename}", target)
        from PIL import Image
        with Image.open(target) as image:
            assert image.size == (im["width"], im["height"])
            image.verify()
        labels = []
        xml = ET.Element("annotation")
        ET.SubElement(xml, "filename").text = filename
        size = ET.SubElement(xml, "size")
        for key, value in (("width", im["width"]), ("height", im["height"]), ("depth", 3)):
            ET.SubElement(size, key).text = str(value)
        for ann in annotations[im["id"]]:
            x, y, w, h = ann["bbox"]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(im["width"], x + w), min(im["height"], y + h)
            labels.append(f"{class_id} {(x1+x2)/2/im['width']:.8f} {(y1+y2)/2/im['height']:.8f} {(x2-x1)/im['width']:.8f} {(y2-y1)/im['height']:.8f}")
            obj = ET.SubElement(xml, "object")
            ET.SubElement(obj, "name").text = name
            box = ET.SubElement(obj, "bndbox")
            for key, value in zip(("xmin", "ymin", "xmax", "ymax"), (x1, y1, x2, y2)):
                ET.SubElement(box, key).text = str(round(value))
        label_path = ROOT / "data/labels" / split / f"{target.stem}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(labels) + "\n")
        xml_path = ROOT / "data/annotations_voc" / split / f"{target.stem}.xml"
        xml_path.parent.mkdir(parents=True, exist_ok=True)
        ET.ElementTree(xml).write(xml_path, encoding="utf-8", xml_declaration=True)
        lic = licenses[im["license"]]
        return {"image_id": im["id"], "file": str(relative), "class_id": class_id,
                "class_name": name, "split": split, "width": im["width"], "height": im["height"],
                "boxes": len(labels), "coco_url": f"{S3}/val2017/{filename}",
                "flickr_url": im.get("flickr_url", ""), "license": lic["name"],
                "license_url": lic["url"], "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}

    with ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(build, chosen))
    rows.sort(key=lambda r: (r["split"], r["class_id"], r["image_id"]))
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    selected_ids = {item[0]["id"] for item in chosen}
    subset = {"info": coco["info"], "licenses": coco["licenses"],
              "images": [im for im in coco["images"] if im["id"] in selected_ids],
              "categories": [c for c in coco["categories"] if c["id"] in CLASSES],
              "annotations": [{k: v for k, v in a.items() if k != "segmentation"}
                              for a in coco["annotations"] if a["image_id"] in selected_ids and a["category_id"] in CLASSES]}
    (ROOT / "data/coco_subset.json").write_text(json.dumps(subset, indent=2))
    return audit()


def configure() -> None:
    """Gera caminhos absolutos locais e pastas de classificação sem alterar splits."""
    import yaml
    config = {"path": str(ROOT / "data"), "train": "images/train", "val": "images/val",
              "test": "images/test", "nc": 2, "names": ["gato", "carro"]}
    (ROOT / "data/dataset-local.yaml").write_text(yaml.safe_dump(config))
    with (ROOT / "data/manifest.csv").open() as f:
        for row in csv.DictReader(f):
            dst = ROOT / "data/classification" / row["split"] / row["class_name"] / Path(row["file"]).name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(ROOT / row["file"], dst)


def audit() -> dict:
    """Impede IDs/hashes repetidos, verifica arquivos e limites das caixas."""
    with (ROOT / "data/manifest.csv").open() as f:
        rows = list(csv.DictReader(f))
    counts = Counter((r["class_name"], r["split"]) for r in rows)
    assert len(rows) == 80 and len({r["image_id"] for r in rows}) == 80
    assert len({r["sha256"] for r in rows}) == 80
    for name in ("gato", "carro"):
        for split, n in (("train", 32), ("val", 4), ("test", 4)):
            assert counts[name, split] == n
    n_boxes = 0
    visual_hashes = {}
    for row in rows:
        path = ROOT / row["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        from PIL import Image
        import numpy as np
        with Image.open(path) as im:
            pixels = np.asarray(im.convert("L").resize((9, 8)))
        visual_hashes[row["image_id"]] = (pixels[:, 1:] > pixels[:, :-1]).ravel()
        labels = ROOT / "data/labels" / row["split"] / f"{path.stem}.txt"
        for line in labels.read_text().splitlines():
            cls, cx, cy, w, h = map(float, line.split())
            assert cls == int(row["class_id"]) and 0 < w <= 1 and 0 < h <= 1
            assert -1e-7 <= cx - w/2 and cx+w/2 <= 1+1e-7
            assert -1e-7 <= cy - h/2 and cy+h/2 <= 1+1e-7
            n_boxes += 1
    import itertools
    split_by_id = {row["image_id"]: row["split"] for row in rows}
    similar_pairs = [{"image_a": a, "image_b": b, "distance": int((visual_hashes[a] != visual_hashes[b]).sum()),
                      "split_a": split_by_id[a], "split_b": split_by_id[b]}
                     for a, b in itertools.combinations(visual_hashes, 2)
                     if (visual_hashes[a] != visual_hashes[b]).sum() <= 4]
    assert all(pair["split_a"] == pair["split_b"] for pair in similar_pairs), "Revisar cenas semelhantes entre partições"
    configure()
    result = {"images": len(rows), "boxes": n_boxes, "counts": {f"{c}/{s}": n for (c, s), n in counts.items()},
              "seed": SEED, "id_overlap": 0, "sha256_overlap": 0,
              "dhash_threshold": 4, "similar_pairs": similar_pairs, "similar_pairs_across_splits": 0,
              "annotation_source": "COCO 2017; conversão automática para YOLO e Pascal VOC",
              "makesense_review": (json.loads((ROOT / "data/makesense/verificacao.json").read_text()) if (ROOT / "data/makesense/verificacao.json").exists() else "Pendente de importação, revisão e exportação no Make Sense AI")}
    (ROOT / "results/dataset_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2, ensure_ascii=False))
