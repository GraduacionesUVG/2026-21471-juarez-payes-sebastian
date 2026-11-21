#!/usr/bin/env python3
"""
ModeloClasificacion.py
======================
Clasificador binario de imágenes reales frente a imágenes generadas por IA
(ResNet-50 con aprendizaje por transferencia), con mapas de Grad-CAM y de
sensibilidad por oclusión para compararlos con la mirada de los participantes
registrada en el Neurolab de la UVG.

Subcomandos
-----------
  audit    Estadísticas del conjunto de datos y prueba de atajos: qué tan
           separables son las clases usando solo metadatos de archivo.
  split    Partición estratificada y reproducible (train/val/test).
  train    Entrenamiento en dos fases: cabeza con el backbone congelado y
           luego ajuste fino de toda la red.
  eval     Evalúa un checkpoint sobre una partición y guarda métricas y figuras.
  predict  Predice sobre una carpeta de imágenes.
  testext  Evaluación externa pareada sobre los estímulos del experimento
           (real frente a IA por nombre de archivo).
  gradcam  Genera Grad-CAM (y, opcionalmente, oclusión) para una carpeta.

predict, testext y gradcam aceptan --display_size 3840x2160 --display_fit fill
para que el modelo vea los mismos píxeles que vio el participante.

Ejemplo
-------
  python ModeloClasificacion.py audit   --data fakeV2
  python ModeloClasificacion.py split   --data fakeV2 --out runs/exp1
  python ModeloClasificacion.py train   --data fakeV2 --out runs/exp1
  python ModeloClasificacion.py eval    --out runs/exp1 --split test
  python ModeloClasificacion.py gradcam --out runs/exp1 --images estimulos/ --occlusion
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFile, ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    from tqdm import tqdm
except ImportError:  # tqdm es opcional: sin él no hay barras de progreso
    def tqdm(x, **kw):
        return x

# Algunas imágenes del conjunto de Kaggle vienen truncadas.
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
LABELS = {"real": 0, "fake": 1}  # el logit del modelo es la evidencia de "fake"


# =============================================================================
# Configuración
# =============================================================================
@dataclass
class Config:
    """Hiperparámetros del experimento. Se guarda en <out>/config.json."""

    # Datos
    data: str = "fakeV2"
    real_dir: str = "real"
    fake_dir: str = "fake-v2"
    out: str = "runs/exp1"
    val_frac: float = 0.10
    test_frac: float = 0.10
    seed: int = 42

    # Preprocesamiento común a ambas clases (evita que el modelo aprenda el formato)
    resize: int = 256                           # lado corto tras el remuestreo
    crop: int = 224                             # tamaño del recorte de entrada
    jpeg_p: float = 0.6                         # prob. de recompresión JPEG (train)
    jpeg_q: Tuple[int, int] = (30, 95)          # rango de calidad JPEG (train)
    blur_p: float = 0.3                         # prob. de desenfoque gaussiano (train)
    blur_sigma: Tuple[float, float] = (0.0, 2.0)
    eval_jpeg: int = 90                         # calidad JPEG fija en val/test (0 = no recomprimir)

    # Modelo
    backbone: str = "resnet50"
    pretrained: bool = True
    dropout: float = 0.3

    # Entrenamiento
    epochs: int = 12
    head_epochs: int = 2                        # fase 1: solo la cabeza
    batch_size: int = 32
    lr: float = 3e-4                            # tasa de aprendizaje de la cabeza
    backbone_lr_mult: float = 0.1               # el backbone usa lr * 0.1
    weight_decay: float = 0.05
    warmup_epochs: float = 0.5
    label_smoothing: float = 0.02
    grad_clip: float = 1.0
    samples_per_epoch: int = 0                  # 0 = tamaño de la partición de train
    patience: int = 4                           # parada temprana por AUC de validación
    num_workers: int = 4
    amp: bool = True                            # precisión mixta (solo en CUDA)
    device: str = "auto"

    # Mapas de interpretabilidad
    cam_layers: Tuple[str, ...] = ("layer3", "layer4")
    cam_target: str = "predicted"               # predicted | fake | real
    infer_size: int = 224                       # lado corto en inferencia (sin recorte)
    occ_patch: int = 32
    occ_stride: int = 16

    # Estímulos del experimento: reproducir lo que vio el participante
    display_size: str = ""                      # p. ej. "3840x2160"; vacío = imagen tal cual
    display_fit: str = "fill"                   # fill | stretch | fit

    def save(self, path: Path):
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))

    @staticmethod
    def load(path: Path) -> "Config":
        d = json.loads(path.read_text())
        for k in ("jpeg_q", "blur_sigma", "cam_layers"):
            if k in d:
                d[k] = tuple(d[k])
        return Config(**d)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name: str) -> torch.device:
    """CUDA, luego Apple Silicon (MPS), luego CPU."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# Índice del conjunto de datos y particiones
# =============================================================================
def list_images(folder: Path) -> List[Path]:
    return sorted(p for p in folder.iterdir()
                  if p.suffix.lower() in IMG_EXT and not p.name.startswith("."))


def image_meta(path: Path) -> Dict:
    """Metadatos de archivo sin decodificar los píxeles (usado por audit)."""
    try:
        with Image.open(path) as im:
            return {"width": im.size[0], "height": im.size[1], "format": im.format or "",
                    "mode": im.mode, "bytes": path.stat().st_size, "ok": 1}
    except Exception:
        return {"width": 0, "height": 0, "format": "ERR", "mode": "", "bytes": 0, "ok": 0}


def build_index(cfg: Config) -> List[Dict]:
    root = Path(cfg.data)
    rows = []
    for name, sub in (("real", cfg.real_dir), ("fake", cfg.fake_dir)):
        for p in list_images(root / sub):
            rows.append({"path": str(p), "label": LABELS[name], "cls": name})
    if not rows:
        sys.exit(f"No se encontraron imágenes en {root}/{cfg.real_dir} ni {root}/{cfg.fake_dir}")
    return rows


def make_splits(cfg: Config) -> Path:
    """Partición estratificada por clase con semilla fija. Guarda <out>/splits.csv."""
    rows = build_index(cfg)
    rng = random.Random(cfg.seed)
    by_cls: Dict[str, List[Dict]] = {}
    for r in rows:
        by_cls.setdefault(r["cls"], []).append(r)
    for items in by_cls.values():
        rng.shuffle(items)
        n = len(items)
        n_test = int(round(n * cfg.test_frac))
        n_val = int(round(n * cfg.val_frac))
        for i, r in enumerate(items):
            r["split"] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "splits.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "cls", "split"])
        w.writeheader()
        w.writerows(rows)

    c = Counter((r["split"], r["cls"]) for r in rows)
    print("Particiones guardadas en", path)
    for s in ("train", "val", "test"):
        print(f"  {s:5s}  real={c[(s, 'real')]:6d}  fake={c[(s, 'fake')]:6d}")
    return path


def read_splits(cfg: Config, split: str) -> List[Dict]:
    path = Path(cfg.out) / "splits.csv"
    if not path.exists():
        sys.exit(f"No existe {path}. Ejecute primero: "
                 f"python ModeloClasificacion.py split --data {cfg.data} --out {cfg.out}")
    with path.open() as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    for r in rows:
        r["label"] = int(r["label"])
    return rows


# =============================================================================
# Preprocesamiento
# Ambas clases pasan por el mismo proceso (RGB -> remuestreo -> JPEG -> blur)
# para que el modelo no pueda separarlas por formato, tamaño o compresión.
# =============================================================================
def to_rgb(im: Image.Image) -> Image.Image:
    """Convierte cualquier modo a RGB; la transparencia se compone sobre blanco."""
    if im.mode == "RGB":
        return im
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        return bg
    return im.convert("RGB")


def resize_short(im: Image.Image, short: int) -> Image.Image:
    """Remuestrea siempre el lado corto, para que todas las imágenes compartan
    el mismo filtro de interpolación."""
    w, h = im.size
    s = short / min(w, h)
    new = (max(1, round(w * s)), max(1, round(h * s)))
    return im.resize(new, Image.BICUBIC, reducing_gap=3.0)


def jpeg_recompress(im: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=int(quality), subsampling=2)  # 4:2:0
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def to_tensor(im: Image.Image) -> torch.Tensor:
    """PIL -> tensor normalizado con las estadísticas de ImageNet."""
    arr = np.asarray(im, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t - mean) / std


class TrainTransform:
    """Remuestreo -> blur aleatorio -> JPEG aleatorio -> recorte aleatorio suave -> volteo."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def __call__(self, im: Image.Image) -> torch.Tensor:
        cfg = self.cfg
        im = resize_short(to_rgb(im), cfg.resize)
        if random.random() < cfg.blur_p:
            sigma = random.uniform(*cfg.blur_sigma)
            if sigma > 0.05:
                im = im.filter(ImageFilter.GaussianBlur(sigma))
        if random.random() < cfg.jpeg_p:
            im = jpeg_recompress(im, random.randint(*cfg.jpeg_q))

        # Recorte aleatorio conservador: 60-100 % del área, aspecto 0.9-1.1
        w, h = im.size
        area = w * h
        for _ in range(10):
            target = random.uniform(0.6, 1.0) * area
            ratio = math.exp(random.uniform(math.log(0.9), math.log(1.1)))
            cw = int(round(math.sqrt(target * ratio)))
            ch = int(round(math.sqrt(target / ratio)))
            if 0 < cw <= w and 0 < ch <= h:
                x, y = random.randint(0, w - cw), random.randint(0, h - ch)
                im = im.crop((x, y, x + cw, y + ch))
                break
        im = im.resize((cfg.crop, cfg.crop), Image.BICUBIC)
        if random.random() < 0.5:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        return to_tensor(im)


class EvalTransform:
    """Remuestreo -> JPEG de calidad fija -> recorte central."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def __call__(self, im: Image.Image) -> torch.Tensor:
        cfg = self.cfg
        im = resize_short(to_rgb(im), cfg.resize)
        if cfg.eval_jpeg > 0:
            im = jpeg_recompress(im, cfg.eval_jpeg)
        w, h = im.size
        left, top = (w - cfg.crop) // 2, (h - cfg.crop) // 2
        im = im.crop((left, top, left + cfg.crop, top + cfg.crop))
        return to_tensor(im)


def render_displayed(im: Image.Image, size_wh: Tuple[int, int], fit: str) -> Image.Image:
    """Reproduce cómo se mostró el estímulo en pantalla durante el experimento.
    fill: escala hasta cubrir y recorta al centro; stretch: deforma; fit: barras negras."""
    im = to_rgb(im)
    W, H = size_wh
    w, h = im.size
    if fit == "stretch":
        return im.resize((W, H), Image.BICUBIC)
    s = max(W / w, H / h) if fit == "fill" else min(W / w, H / h)
    rw, rh = max(1, round(w * s)), max(1, round(h * s))
    im = im.resize((rw, rh), Image.BICUBIC, reducing_gap=3.0)
    if fit == "fill":
        left, top = (rw - W) // 2, (rh - H) // 2
        return im.crop((left, top, left + W, top + H))
    canvas = Image.new("RGB", (W, H), (0, 0, 0))
    canvas.paste(im, ((W - rw) // 2, (H - rh) // 2))
    return canvas


def parse_size(s: str) -> Optional[Tuple[int, int]]:
    """"3840x2160" -> (3840, 2160)."""
    if not s:
        return None
    w, h = s.lower().replace("×", "x").split("x")
    return int(w), int(h)


class InferTransform:
    """Inferencia sobre imágenes completas, sin recorte (la ResNet es totalmente
    convolucional). Devuelve el tensor y el tamaño original para reproyectar los mapas."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.display = parse_size(cfg.display_size)

    def __call__(self, im: Image.Image) -> Tuple[torch.Tensor, Tuple[int, int]]:
        cfg = self.cfg
        if self.display:
            im = render_displayed(im, self.display, cfg.display_fit)
        orig = im.size
        im = resize_short(to_rgb(im), cfg.infer_size)
        if cfg.eval_jpeg > 0:
            im = jpeg_recompress(im, cfg.eval_jpeg)
        # Dimensiones múltiplo de 32 para que layer4 tenga una cuadrícula exacta
        w, h = im.size
        w32, h32 = max(32, (w // 32) * 32), max(32, (h // 32) * 32)
        if (w32, h32) != (w, h):
            im = im.resize((w32, h32), Image.BICUBIC)
        return to_tensor(im), orig


class ImageListDataset(Dataset):
    def __init__(self, rows: List[Dict], transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        try:
            with Image.open(r["path"]) as im:
                im.load()
                x = self.transform(im)
        except Exception as e:  # imagen corrupta: se reemplaza por un tensor de ceros
            print(f"[aviso] no se pudo leer {r['path']}: {e}", file=sys.stderr)
            x = torch.zeros(3, self.transform.cfg.crop, self.transform.cfg.crop)
        return x, torch.tensor(float(r["label"])), i


def make_loader(cfg: Config, rows: List[Dict], train: bool, device: torch.device) -> DataLoader:
    """En entrenamiento usa un muestreador balanceado por clase (el conjunto es 4.7:1)."""
    ds = ImageListDataset(rows, TrainTransform(cfg) if train else EvalTransform(cfg))
    kw = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers,
              pin_memory=(device.type == "cuda"), persistent_workers=cfg.num_workers > 0)
    if train:
        counts = Counter(r["label"] for r in rows)
        weights = [1.0 / counts[r["label"]] for r in rows]
        n = cfg.samples_per_epoch or len(rows)
        sampler = WeightedRandomSampler(weights, num_samples=n, replacement=True)
        return DataLoader(ds, sampler=sampler, drop_last=True, **kw)
    return DataLoader(ds, shuffle=False, **kw)


# =============================================================================
# Modelo
# =============================================================================
class Classifier(nn.Module):
    """ResNet-50 preentrenada en ImageNet con una cabeza de un solo logit (P(fake))."""

    def __init__(self, cfg: Config):
        super().__init__()
        import torchvision
        if cfg.backbone != "resnet50":
            raise ValueError("Solo se admite backbone='resnet50'.")
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if cfg.pretrained else None
        net = torchvision.models.resnet50(weights=weights)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2 = net.layer1, net.layer2
        self.layer3, self.layer4 = net.layer3, net.layer4
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(cfg.dropout), nn.Linear(2048, 1))

    def features(self, x) -> Dict[str, torch.Tensor]:
        """Activaciones de cada bloque (layer3 y layer4 se usan para Grad-CAM)."""
        f = {}
        x = self.stem(x)
        x = self.layer1(x)
        f["layer1"] = x
        x = self.layer2(x)
        f["layer2"] = x
        x = self.layer3(x)
        f["layer3"] = x
        x = self.layer4(x)
        f["layer4"] = x
        return f

    def forward(self, x):
        return self.head(self.features(x)["layer4"]).squeeze(1)

    def backbone_params(self):
        for m in (self.stem, self.layer1, self.layer2, self.layer3, self.layer4):
            yield from m.parameters()

    def set_backbone_trainable(self, flag: bool):
        """Congela o libera el backbone. Con el backbone congelado, las capas de
        BatchNorm también quedan en modo evaluación para conservar las estadísticas
        de ImageNet."""
        for p in self.backbone_params():
            p.requires_grad_(flag)
        for m in (self.stem, self.layer1, self.layer2, self.layer3, self.layer4):
            for bn in m.modules():
                if isinstance(bn, nn.BatchNorm2d):
                    bn.train(flag)


def load_checkpoint(cfg: Config, device: torch.device, name: str = "best.pt") -> Classifier:
    """Carga pesos y el umbral de decisión calibrado en validación."""
    path = Path(cfg.out) / name
    if not path.exists():
        sys.exit(f"No existe {path}")
    ck = torch.load(path, map_location="cpu")
    model = Classifier(cfg)
    model.load_state_dict(ck["model"])
    model.threshold = float(ck.get("threshold", 0.5))
    return model.to(device).eval()


# =============================================================================
# Métricas
# =============================================================================
def compute_metrics(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    """Métricas binarias con "fake" como clase positiva."""
    y = y.astype(int)
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    m = {"threshold": thr, "acc": (tp + tn) / max(len(y), 1), "balanced_acc": (tpr + tnr) / 2,
         "recall_fake": tpr, "recall_real": tnr, "precision_fake": prec,
         "f1_fake": 2 * prec * tpr / max(prec + tpr, 1e-9),
         "tp": tp, "tn": tn, "fp": fp, "fn": fn}
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        m["auc"] = float(roc_auc_score(y, p))
        m["ap_fake"] = float(average_precision_score(y, p))
    except Exception:
        m["auc"] = auc_manual(y, p)
    return m


def auc_manual(y, p) -> float:
    """AUC-ROC por rangos (Mann-Whitney), usada si scikit-learn no está disponible."""
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    n1 = (y == 1).sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Umbral que maximiza la exactitud balanceada en validación."""
    best, best_t = -1.0, 0.5
    for t in np.linspace(0.02, 0.98, 97):
        b = compute_metrics(y, p, t)["balanced_acc"]
        if b > best:
            best, best_t = b, float(t)
    return best_t


@torch.no_grad()
def run_inference(model: Classifier, loader: DataLoader, device: torch.device,
                  desc="eval") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve etiquetas, probabilidades de "fake" e índices de cada imagen."""
    model.eval()
    ys, ps, idx = [], [], []
    for x, y, i in tqdm(loader, desc=desc, leave=False):
        x = x.to(device, non_blocking=True)
        logits = model(x)
        ps.append(torch.sigmoid(logits).float().cpu().numpy())
        ys.append(y.numpy())
        idx.append(i.numpy())
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(idx)


def save_figures(y, p, thr, out: Path, prefix: str):
    """Curva ROC y matriz de confusión en <out>/<prefix>_roc_cm.png."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve
    except Exception:
        return
    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(fpr, tpr, color="#4C6EF5")
    ax[0].plot([0, 1], [0, 1], "--", color="#999")
    ax[0].set_xlabel("FPR")
    ax[0].set_ylabel("TPR")
    ax[0].set_title(f"ROC (AUC={auc_manual(y.astype(int), p):.3f})")

    m = compute_metrics(y, p, thr)
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    ax[1].imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax[1].text(j, i, str(v), ha="center", va="center", color="black")
    ax[1].set_xticks([0, 1])
    ax[1].set_xticklabels(["pred real", "pred fake"])
    ax[1].set_yticks([0, 1])
    ax[1].set_yticklabels(["real", "fake"])
    ax[1].set_title(f"Matriz de confusión (umbral={thr:.2f})")
    fig.tight_layout()
    fig.savefig(out / f"{prefix}_roc_cm.png", dpi=130)
    plt.close(fig)


# =============================================================================
# Entrenamiento
# =============================================================================
def cosine_lr(step: int, total: int, warmup: int, base: float) -> float:
    """Calentamiento lineal seguido de decaimiento coseno."""
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    prog = (step - warmup) / max(total - warmup, 1)
    return base * 0.5 * (1 + math.cos(math.pi * prog))


def train(cfg: Config):
    """Fase 1 (head_epochs): solo la cabeza. Fase 2: toda la red, con parada
    temprana por AUC de validación. Guarda best.pt, last.pt y history.csv."""
    seed_everything(cfg.seed)
    device = pick_device(cfg.device)
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg.save(out / "config.json")
    use_amp = cfg.amp and device.type == "cuda"
    print(f"device={device}  amp={use_amp}")

    train_rows, val_rows = read_splits(cfg, "train"), read_splits(cfg, "val")
    train_loader = make_loader(cfg, train_rows, True, device)
    val_loader = make_loader(cfg, val_rows, False, device)

    model = Classifier(cfg).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    def loss_fn(logits, y):
        """Entropía cruzada binaria con suavizado de etiquetas (0.01 / 0.99)."""
        y_s = y * (1 - cfg.label_smoothing) + 0.5 * cfg.label_smoothing
        return F.binary_cross_entropy_with_logits(logits, y_s)

    # Tasas de aprendizaje distintas para la cabeza y el backbone
    opt = torch.optim.AdamW([
        {"params": list(model.head.parameters()), "lr": cfg.lr, "name": "head"},
        {"params": list(model.backbone_params()), "lr": cfg.lr * cfg.backbone_lr_mult, "name": "backbone"},
    ], weight_decay=cfg.weight_decay)
    base_lrs = [g["lr"] for g in opt.param_groups]
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):  # torch < 2.3
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    steps_per_epoch = len(train_loader)
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)
    history_path = out / "history.csv"
    with history_path.open("w") as f:
        f.write("epoch,phase,train_loss,val_loss,val_auc,val_bacc,val_thr,lr_head,seconds\n")

    best_auc, best_score, bad_epochs, step = -1.0, (-1.0, -1.0), 0, 0
    for epoch in range(cfg.epochs):
        phase = "head" if epoch < cfg.head_epochs else "full"
        model.set_backbone_trainable(phase == "full")
        model.train()
        if phase == "head":
            model.set_backbone_trainable(False)  # model.train() reactiva BatchNorm; se vuelve a congelar

        # --- Época de entrenamiento
        t0, run_loss, n = time.time(), 0.0, 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{cfg.epochs} [{phase}]", leave=False)
        for x, y, _ in pbar:
            for g, base in zip(opt.param_groups, base_lrs):
                g["lr"] = cosine_lr(step, total_steps, warmup_steps, base)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if device.type == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = model(x)
                loss = loss_fn(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item() * len(y)
            n += len(y)
            step += 1
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix(loss=f"{run_loss / n:.4f}")

        # --- Validación y calibración del umbral
        yv, pv, _ = run_inference(model, val_loader, device, "val")
        with torch.no_grad():
            val_loss = float(F.binary_cross_entropy(torch.tensor(pv).clamp(1e-6, 1 - 1e-6),
                                                    torch.tensor(yv, dtype=torch.float32)))
        thr = best_threshold(yv, pv)
        m = compute_metrics(yv, pv, thr)
        secs = time.time() - t0
        print(f"epoch {epoch + 1:2d} [{phase}] train_loss={run_loss / n:.4f} val_loss={val_loss:.4f} "
              f"val_auc={m['auc']:.4f} val_bacc={m['balanced_acc']:.4f} thr={thr:.2f} ({secs:.0f}s)")
        with history_path.open("a") as f:
            f.write(f"{epoch + 1},{phase},{run_loss / n:.5f},{val_loss:.5f},{m['auc']:.5f},"
                    f"{m['balanced_acc']:.5f},{thr:.3f},{opt.param_groups[0]['lr']:.2e},{secs:.0f}\n")

        # --- Checkpoints: se conserva el mejor por AUC (desempate por exactitud balanceada)
        ck = {"model": model.state_dict(), "epoch": epoch + 1, "threshold": thr,
              "val_metrics": m, "config": asdict(cfg)}
        torch.save(ck, out / "last.pt")
        score = (round(m["auc"], 4), round(m["balanced_acc"], 4))
        if score > best_score:
            best_score, best_auc, bad_epochs = score, m["auc"], 0
            torch.save(ck, out / "best.pt")
            save_figures(yv, pv, thr, out, "val")
        else:
            bad_epochs += 1
            if phase == "full" and bad_epochs >= cfg.patience:
                print(f"Parada temprana: sin mejora en {cfg.patience} épocas")
                break
    print(f"Mejor AUC de validación = {best_auc:.4f}  -> {out / 'best.pt'}")


# =============================================================================
# Evaluación y predicción
# =============================================================================
def evaluate(cfg: Config, split: str, ckpt: str):
    """Métricas con umbral 0.5 y con el umbral de validación; guarda JSON, CSV y figura."""
    device = pick_device(cfg.device)
    model = load_checkpoint(cfg, device, ckpt)
    rows = read_splits(cfg, split)
    loader = make_loader(cfg, rows, False, device)
    y, p, idx = run_inference(model, loader, device, split)

    out = Path(cfg.out)
    res = {"split": split, "checkpoint": ckpt, "n": int(len(y)),
           "at_0.5": compute_metrics(y, p, 0.5),
           "at_val_threshold": compute_metrics(y, p, model.threshold)}
    (out / f"metrics_{split}.json").write_text(json.dumps(res, indent=2))
    with (out / f"predictions_{split}.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "p_fake"])
        for i, yy, pp in zip(idx, y, p):
            w.writerow([rows[int(i)]["path"], int(yy), f"{pp:.5f}"])
    save_figures(y, p, model.threshold, out, split)

    m = res["at_val_threshold"]
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}, indent=2))
    print(f"Guardado en {out}/metrics_{split}.json y predictions_{split}.csv")


def iter_folder_images(folder: Path) -> List[Path]:
    return sorted(p for p in folder.rglob("*")
                  if p.suffix.lower() in IMG_EXT and not p.name.startswith("."))


@torch.no_grad()
def predict_folder(cfg: Config, images: str, ckpt: str, out_csv: Optional[str]):
    """Probabilidad de "fake" y predicción para cada imagen de una carpeta."""
    device = pick_device(cfg.device)
    model = load_checkpoint(cfg, device, ckpt)
    tf = InferTransform(cfg)
    paths = iter_folder_images(Path(images))
    out_csv = out_csv or str(Path(cfg.out) / "predictions_folder.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "p_fake", "pred", "threshold"])
        for p in tqdm(paths, desc="predict"):
            with Image.open(p) as im:
                x, _ = tf(im)
            pf = float(torch.sigmoid(model(x.unsqueeze(0).to(device))).item())
            pred = "fake" if pf >= model.threshold else "real"
            w.writerow([str(p), f"{pf:.5f}", pred, f"{model.threshold:.3f}"])
            print(f"{p.name:40s} p_fake={pf:.3f} -> {pred}")
    print("Guardado en", out_csv)


# =============================================================================
# Evaluación externa pareada (estímulos del experimento)
# =============================================================================
def stimulus_info(path: Path, root: Path) -> Optional[Dict]:
    """Deduce etiqueta, categoría, conjunto y pareja a partir de la ruta.
    Ej.: Estimulos_Reales/Ave_1_Real.jpg  -> real, ave, pareja estimulos:ave_1
         Prompts_ChatGPT/Ave_3_chatgpt.png -> fake, ave, pareja estimulos:ave_3"""
    rel = path.relative_to(root).as_posix().lower()
    stem = path.stem.lower()
    if "_ia" in stem or "chatgpt" in rel or "/imagenes_ia" in rel:
        label = 1
    elif "_real" in stem or "reales" in rel:
        label = 0
    else:
        return None
    category = stem.split("_")[0]
    conjunto = "regenerar" if "regenerar" in rel else "estimulos"
    # El id de pareja quita los marcadores de clase y conserva "re" (imágenes regeneradas)
    pair = re.sub(r"_(ia|real|chatgpt)(?=_|$)", "", stem).rstrip("_")
    return {"path": str(path), "label": label, "cls": "fake" if label else "real",
            "category": category, "set": conjunto, "pair": f"{conjunto}:{pair}"}


def bootstrap_ci(y: np.ndarray, p: np.ndarray, thr: float, n_boot: int = 2000,
                 seed: int = 0) -> Dict[str, Tuple[float, float]]:
    """Intervalos de confianza al 95 % (bootstrap por percentiles) para AUC y exactitud balanceada."""
    rng = np.random.default_rng(seed)
    aucs, baccs = [], []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, pb = y[idx], p[idx]
        if yb.min() == yb.max():  # remuestreo con una sola clase: se descarta
            continue
        m = compute_metrics(yb, pb, thr)
        aucs.append(m["auc"])
        baccs.append(m["balanced_acc"])

    def ci95(a):
        return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

    return {"auc_ci95": ci95(aucs), "balanced_acc_ci95": ci95(baccs)}


@torch.no_grad()
def test_external(cfg: Config, images: str, ckpt: str, out_dir: Optional[str]):
    """Evalúa el checkpoint sobre los estímulos (real frente a IA). Reporta métricas
    globales con IC, elección forzada (2AFC) por pareja y desglose por conjunto y categoría."""
    device = pick_device(cfg.device)
    model = load_checkpoint(cfg, device, ckpt)
    tf = InferTransform(cfg)
    root = Path(images)
    rows = [r for r in (stimulus_info(p, root) for p in iter_folder_images(root)) if r]
    if not rows:
        sys.exit("No se pudieron deducir etiquetas: los nombres deben contener _Real, _IA o "
                 "chatgpt, o estar en carpetas Reales / IA.")
    for r in tqdm(rows, desc="test externo"):
        with Image.open(r["path"]) as im:
            im.load()
            x, _ = tf(im)
        r["p_fake"] = float(torch.sigmoid(model(x.unsqueeze(0).to(device))).item())
        r["pred"] = int(r["p_fake"] >= model.threshold)
        r["correct"] = int(r["pred"] == r["label"])

    out = Path(out_dir or (Path(cfg.out) / "test_externo"))
    out.mkdir(parents=True, exist_ok=True)
    y = np.array([r["label"] for r in rows])
    p = np.array([r["p_fake"] for r in rows])
    report: Dict = {"n": len(rows), "threshold": model.threshold,
                    "display_size": cfg.display_size or None,
                    "display_fit": cfg.display_fit if cfg.display_size else None,
                    "overall": compute_metrics(y, p, model.threshold)}
    report["overall"].update(bootstrap_ci(y, p, model.threshold))

    # 2AFC por pareja: ¿la imagen de IA recibe mayor p_fake que su par real? (no depende del umbral)
    pairs: Dict[str, Dict[str, float]] = {}
    for r in rows:
        pairs.setdefault(r["pair"], {})[r["cls"]] = r["p_fake"]
    complete = {k: v for k, v in pairs.items() if "real" in v and "fake" in v}
    wins = sum(1 for v in complete.values() if v["fake"] > v["real"])
    report["paired_2afc"] = {"pairs": len(complete), "fake_scored_higher": wins,
                             "accuracy": wins / max(len(complete), 1),
                             "incomplete_pairs": sorted(set(pairs) - set(complete))}

    def group(key):
        res = {}
        for g in sorted({r[key] for r in rows}):
            sel = [r for r in rows if r[key] == g]
            yy = np.array([r["label"] for r in sel])
            pp = np.array([r["p_fake"] for r in sel])
            m = compute_metrics(yy, pp, model.threshold)
            res[g] = {k: m[k] for k in ("acc", "balanced_acc", "recall_fake", "recall_real", "auc")} | {"n": len(sel)}
        return res

    report["by_set"] = group("set")
    report["by_category"] = group("category")

    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    with (out / "predictions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "set", "category", "pair", "cls",
                                          "label", "p_fake", "pred", "correct"])
        w.writeheader()
        w.writerows(rows)
    save_figures(y, p, model.threshold, out, "externo")

    o = report["overall"]
    print(f"\nTest externo n={len(rows)}  umbral={model.threshold:.2f}"
          + (f"  (render {cfg.display_size} {cfg.display_fit})" if cfg.display_size else ""))
    print(f"  AUC={o['auc']:.3f} {o['auc_ci95']}  balanced_acc={o['balanced_acc']:.3f} {o['balanced_acc_ci95']}")
    print(f"  recall_fake={o['recall_fake']:.3f}  recall_real={o['recall_real']:.3f}")
    print(f"  2AFC pareado: {wins}/{len(complete)} = {report['paired_2afc']['accuracy']:.3f}")
    for name, grp in (("conjunto", report["by_set"]), ("categoría", report["by_category"])):
        print(f"  por {name}:")
        for g, m in grp.items():
            print(f"    {g:12s} n={m['n']:3d} acc={m['acc']:.2f} bacc={m['balanced_acc']:.2f} auc={m['auc']:.2f}")
    print(f"Guardado en {out}/report.json, predictions.csv y externo_roc_cm.png")


# =============================================================================
# Grad-CAM y sensibilidad por oclusión
# =============================================================================
class GradCAM:
    """Grad-CAM (Selvaraju et al., 2017) sobre varias capas a la vez."""

    def __init__(self, model: Classifier, layers: Sequence[str]):
        self.model, self.layers = model, list(layers)
        self.acts: Dict[str, torch.Tensor] = {}
        self.grads: Dict[str, torch.Tensor] = {}
        self.handles = [getattr(model, name).register_forward_hook(self._fwd(name))
                        for name in self.layers]

    def _fwd(self, name):
        def hook(_m, _i, out):
            if not out.requires_grad:  # p. ej. dentro de torch.no_grad() durante la oclusión
                return
            self.acts[name] = out
            out.register_hook(lambda g: self.grads.__setitem__(name, g))
        return hook

    def __call__(self, x: torch.Tensor, target: str) -> Tuple[float, Dict[str, np.ndarray]]:
        """x: tensor (1, 3, H, W). Devuelve p_fake y {capa: mapa (h, w) en [0, 1]}."""
        self.model.zero_grad(set_to_none=True)
        x = x.requires_grad_(True)
        logit = self.model(x)[0]
        p_fake = float(torch.sigmoid(logit).item())
        # Clase explicada: "fake", "real" o la que predijo el modelo
        if target == "fake":
            score = logit
        elif target == "real":
            score = -logit
        else:
            score = logit if p_fake >= getattr(self.model, "threshold", 0.5) else -logit
        score.backward()

        cams = {}
        for name in self.layers:
            a, g = self.acts[name][0], self.grads[name][0]          # (C, h, w)
            w = g.mean(dim=(1, 2), keepdim=True)                     # peso de cada canal
            cam = F.relu((w * a).sum(0)).detach().float().cpu().numpy()
            cams[name] = normalize01(cam)
        return p_fake, cams

    def close(self):
        for h in self.handles:
            h.remove()


def normalize01(m: np.ndarray) -> np.ndarray:
    m = m - m.min()
    return m / m.max() if m.max() > 0 else m


@torch.no_grad()
def occlusion_map(model: Classifier, x: torch.Tensor, device, patch: int, stride: int,
                  target_sign: float, batch: int = 32) -> np.ndarray:
    """Sensibilidad por oclusión (Zeiler y Fergus, 2014): cuánto cae el puntaje de la
    clase objetivo al cubrir cada parche con gris (0 tras la normalización de ImageNet)."""
    _, _, H, W = x.shape
    base = float(model(x.to(device))[0].item()) * target_sign
    ys = list(range(0, max(H - patch, 0) + 1, stride))
    xs = list(range(0, max(W - patch, 0) + 1, stride))
    heat = np.zeros((len(ys), len(xs)), dtype=np.float32)
    coords = [(i, j) for i in range(len(ys)) for j in range(len(xs))]
    for k in range(0, len(coords), batch):
        chunk = coords[k:k + batch]
        xb = x.repeat(len(chunk), 1, 1, 1)
        for n, (i, j) in enumerate(chunk):
            xb[n, :, ys[i]:ys[i] + patch, xs[j]:xs[j] + patch] = 0.0
        scores = model(xb.to(device)).float().cpu().numpy() * target_sign
        for n, (i, j) in enumerate(chunk):
            heat[i, j] = base - scores[n]
    return normalize01(np.maximum(heat, 0))


def upsample_map(m: np.ndarray, size_wh: Tuple[int, int]) -> np.ndarray:
    """Interpolación bilineal del mapa al tamaño (W, H) del estímulo."""
    t = torch.from_numpy(m)[None, None].float()
    up = F.interpolate(t, size=(size_wh[1], size_wh[0]), mode="bilinear", align_corners=False)[0, 0]
    return normalize01(up.numpy())


def overlay(im: Image.Image, m: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Superpone un mapa en [0, 1] sobre la imagen con una paleta tipo jet."""
    im = to_rgb(im)
    v = np.clip(m, 0, 1)
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    heat = (np.stack([r, g, b], -1) * 255).astype(np.uint8)
    heat = Image.fromarray(heat).resize(im.size, Image.BILINEAR)
    return Image.blend(im, heat, alpha)


def gradcam_folder(cfg: Config, images: str, ckpt: str, out_dir: Optional[str],
                   occlusion: bool, max_side: int):
    """Para cada imagen guarda los mapas como .npy (valores en [0, 1]) y como
    superposición .jpg: layer3, layer4, fused y, si se pide, occlusion."""
    device = pick_device(cfg.device)
    model = load_checkpoint(cfg, device, ckpt)
    tf = InferTransform(cfg)
    cam = GradCAM(model, cfg.cam_layers)
    out = Path(out_dir or (Path(cfg.out) / "gradcam"))
    out.mkdir(parents=True, exist_ok=True)
    paths = iter_folder_images(Path(images))
    summary = []
    for p in tqdm(paths, desc="gradcam"):
        with Image.open(p) as im:
            im.load()
            x, orig = tf(im)
            # Los mapas se exportan con el lado mayor limitado a max_side
            s = min(1.0, max_side / max(orig)) if max_side > 0 else 1.0
            target_wh = (max(1, round(orig[0] * s)), max(1, round(orig[1] * s)))
            p_fake, cams = cam(x.unsqueeze(0).to(device), cfg.cam_target)
            pred = "fake" if p_fake >= model.threshold else "real"
            explains_fake = cfg.cam_target == "fake" or (cfg.cam_target == "predicted" and pred == "fake")
            sign = 1.0 if explains_fake else -1.0
            row = {"image": str(p), "p_fake": round(p_fake, 5), "pred": pred, "cam_target": cfg.cam_target,
                   "orig_w": orig[0], "orig_h": orig[1], "map_w": target_wh[0], "map_h": target_wh[1]}

            small = render_displayed(im, tf.display, cfg.display_fit) if tf.display else to_rgb(im).copy()
            small.thumbnail(target_wh, Image.BICUBIC)
            maps = {name: upsample_map(c, target_wh) for name, c in cams.items()}
            # "fused": media geométrica de layer3 (detalle) y layer4 (contexto)
            if len(maps) >= 2:
                maps["fused"] = normalize01(np.prod([np.sqrt(m + 1e-6) for m in maps.values()], axis=0))
            if occlusion:
                occ = occlusion_map(model, x.unsqueeze(0), device, cfg.occ_patch, cfg.occ_stride, sign)
                maps["occlusion"] = upsample_map(occ, target_wh)
            for name, m in maps.items():
                np.save(out / f"{p.stem}__{name}.npy", m.astype(np.float32))
                overlay(small, m).save(out / f"{p.stem}__{name}.jpg", quality=90)
            summary.append(row)
            print(f"{p.name:40s} p_fake={p_fake:.3f} -> {pred}")
    cam.close()
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()) if summary else ["image"])
        w.writeheader()
        w.writerows(summary)
    print(f"Mapas (.npy y superposiciones .jpg) guardados en {out}")


# =============================================================================
# Auditoría del conjunto de datos
# =============================================================================
def audit(cfg: Config):
    """Resume formato, modo y tamaño por clase y mide qué tan separables son las
    clases usando solo metadatos (regla trivial y regresión logística)."""
    rows = build_index(cfg)
    print(f"Indexando metadatos de {len(rows)} imágenes...")
    for r in tqdm(rows, desc="audit"):
        r.update(image_meta(Path(r["path"])))
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "audit_metadata.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def summarize(cls):
        rs = [r for r in rows if r["cls"] == cls and r["ok"]]
        sizes = Counter((r["width"], r["height"]) for r in rs)
        print(f"\n[{cls}] n={len(rs)}")
        print("  formato:", dict(Counter(r["format"] for r in rs)))
        print("  modo   :", dict(Counter(r["mode"] for r in rs)))
        print("  tamaños más comunes:", sizes.most_common(5))
        print(f"  lado mayor <= 256: {sum(1 for r in rs if max(r['width'], r['height']) <= 256)}")

    summarize("real")
    summarize("fake")

    ok = [r for r in rows if r["ok"]]
    y = np.array([r["label"] for r in ok])
    rule = np.array([0 if max(r["width"], r["height"]) <= 256 else 1 for r in ok])
    m_rule = compute_metrics(y, rule.astype(float), 0.5)
    print(f"\nRegla trivial 'lado <= 256 => real': balanced_acc={m_rule['balanced_acc']:.3f}  acc={m_rule['acc']:.3f}")
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        fmts = sorted({r["format"] for r in ok})
        modes = sorted({r["mode"] for r in ok})
        X = np.array([[math.log1p(r["width"]), math.log1p(r["height"]), math.log1p(r["bytes"]),
                       r["width"] / max(r["height"], 1)]
                      + [r["format"] == f for f in fmts] + [r["mode"] == mo for mo in modes]
                      for r in ok], dtype=np.float32)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        p = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1]
        m = compute_metrics(y, p, 0.5)
        print(f"Regresión logística solo con metadatos (5 particiones): "
              f"AUC={m['auc']:.3f}  balanced_acc={m['balanced_acc']:.3f}")
        m_rule["metadata_logreg"] = m
    except ImportError:
        print("(instale scikit-learn para la prueba con regresión logística)")
    (out / "audit_shortcut_probe.json").write_text(json.dumps(m_rule, indent=2))
    print(f"\nAuditoría guardada en {out}/audit_metadata.csv y audit_shortcut_probe.json")


# =============================================================================
# Interfaz de línea de comandos
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--data", default=None)
        p.add_argument("--out", default="runs/exp1")
        p.add_argument("--device", default=None)
        p.add_argument("--num_workers", type=int, default=None)
        p.add_argument("--batch_size", type=int, default=None)

    def display_opts(p):
        p.add_argument("--display_size", default=None,
                       help='p. ej. "3840x2160": reproduce el estímulo como se mostró en pantalla')
        p.add_argument("--display_fit", choices=["fill", "stretch", "fit"], default=None)
        p.add_argument("--infer_size", type=int, default=None)

    p = sub.add_parser("audit", help="auditoría de metadatos del conjunto de datos")
    common(p)

    p = sub.add_parser("split", help="partición estratificada train/val/test")
    common(p)
    p.add_argument("--val_frac", type=float, default=None)
    p.add_argument("--test_frac", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)

    p = sub.add_parser("train", help="entrenamiento en dos fases")
    common(p)
    for k in ("epochs", "head_epochs", "patience", "samples_per_epoch", "resize", "crop", "eval_jpeg", "seed"):
        p.add_argument(f"--{k}", type=int, default=None)
    for k in ("lr", "backbone_lr_mult", "weight_decay", "dropout", "jpeg_p", "blur_p", "label_smoothing"):
        p.add_argument(f"--{k}", type=float, default=None)
    p.add_argument("--no_pretrained", action="store_true")
    p.add_argument("--no_amp", action="store_true")

    p = sub.add_parser("eval", help="evaluación sobre una partición")
    common(p)
    p.add_argument("--split", default="test")
    p.add_argument("--ckpt", default="best.pt")

    p = sub.add_parser("predict", help="predicción sobre una carpeta de imágenes")
    common(p)
    display_opts(p)
    p.add_argument("--images", required=True)
    p.add_argument("--ckpt", default="best.pt")
    p.add_argument("--csv", default=None)

    p = sub.add_parser("testext", help="evaluación externa pareada sobre los estímulos")
    common(p)
    display_opts(p)
    p.add_argument("--images", required=True)
    p.add_argument("--ckpt", default="best.pt")
    p.add_argument("--out_dir", default=None)

    p = sub.add_parser("gradcam", help="mapas de Grad-CAM y oclusión")
    common(p)
    display_opts(p)
    p.add_argument("--images", required=True)
    p.add_argument("--ckpt", default="best.pt")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--occlusion", action="store_true")
    p.add_argument("--cam_target", choices=["predicted", "fake", "real"], default=None)
    p.add_argument("--occ_patch", type=int, default=None)
    p.add_argument("--occ_stride", type=int, default=None)
    p.add_argument("--max_side", type=int, default=1024,
                   help="lado mayor de los mapas exportados (0 = tamaño original)")
    return ap


def config_from_args(args) -> Config:
    """Parte de <out>/config.json si existe y aplica encima los argumentos dados."""
    out = Path(args.out)
    cfg = Config.load(out / "config.json") if (out / "config.json").exists() else Config()
    cfg.out = args.out
    for k, v in vars(args).items():
        if k in ("cmd", "no_pretrained", "no_amp") or v is None or not hasattr(cfg, k):
            continue
        setattr(cfg, k, v)
    if getattr(args, "no_pretrained", False):
        cfg.pretrained = False
    if getattr(args, "no_amp", False):
        cfg.amp = False
    return cfg


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    if args.cmd == "audit":
        audit(cfg)
    elif args.cmd == "split":
        make_splits(cfg)
    elif args.cmd == "train":
        train(cfg)
    elif args.cmd == "eval":
        evaluate(cfg, args.split, args.ckpt)
    elif args.cmd == "predict":
        predict_folder(cfg, args.images, args.ckpt, args.csv)
    elif args.cmd == "testext":
        test_external(cfg, args.images, args.ckpt, args.out_dir)
    elif args.cmd == "gradcam":
        gradcam_folder(cfg, args.images, args.ckpt, args.out_dir, args.occlusion, args.max_side)


if __name__ == "__main__":
    main()
