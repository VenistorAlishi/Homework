"""
Полный цикл: split → обучение YOLO → инференс (TTA) → sample_submission.csv.
Seed 993. Параметры через переменные окружения (см. main).
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

RNG_SEED = 993


def find_repo_root() -> Path:
    here = Path(__file__).resolve().parent.parent
    for p in [here, *here.parents]:
        if (p / "2026-cv-competition" / "sample_submission.csv").is_file():
            return p
    raise FileNotFoundError("Нет 2026-cv-competition/sample_submission.csv рядом с репозиторием.")


def set_seeds() -> None:
    os.environ["PYTHONHASHSEED"] = str(RNG_SEED)
    random.seed(RNG_SEED)

    import numpy as np

    np.random.seed(RNG_SEED)

    import torch

    torch.manual_seed(RNG_SEED)
    torch.cuda.manual_seed_all(RNG_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_split(root: Path) -> Path:
    import yaml
    from sklearn.model_selection import train_test_split

    comp = root / "2026-cv-competition"
    train_img = comp / "train" / "train" / "images"
    all_images = sorted(train_img.glob("*.jpg"))
    train_paths, val_paths = train_test_split(
        all_images,
        test_size=0.15,
        random_state=RNG_SEED,
        shuffle=True,
    )

    runs = root / "runs"
    split_dir = runs / "yolo_split"
    split_dir.mkdir(parents=True, exist_ok=True)
    train_txt = split_dir / "train.txt"
    val_txt = split_dir / "val.txt"
    train_txt.write_text("\n".join(str(p.resolve()) for p in train_paths), encoding="utf-8")
    val_txt.write_text("\n".join(str(p.resolve()) for p in val_paths), encoding="utf-8")

    names = [str(i) for i in range(52)]
    yaml_path = split_dir / "dataset.yaml"
    data = {
        "train": str(train_txt.resolve()),
        "val": str(val_txt.resolve()),
        "nc": 52,
        "names": names,
    }
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print("split:", len(train_paths), "train,", len(val_paths), "val ->", yaml_path)
    return yaml_path


def train_model(root: Path, yaml_path: Path) -> Path:
    import torch
    from ultralytics import YOLO

    epochs = int(os.environ.get("EPOCHS", "50"))
    imgsz = int(os.environ.get("IMGSZ", "640"))
    batch = int(os.environ.get("BATCH", "16" if torch.cuda.is_available() else "4"))
    base = os.environ.get("BASE", "yolov8s.pt")
    run_name = os.environ.get("RUN_NAME", "yolov8s_52cls")

    model = YOLO(base)
    model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        seed=RNG_SEED,
        deterministic=True,
        project=str(root / "runs" / "detect"),
        name=run_name,
        exist_ok=True,
        workers=int(os.environ.get("WORKERS", "0")),
        patience=int(os.environ.get("PATIENCE", "25")),
        cos_lr=os.environ.get("COS_LR", "1") == "1",
    )
    best = root / "runs" / "detect" / run_name / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(best)
    print("checkpoint:", best)
    return best


def predict_and_write(root: Path, best_ckpt: Path) -> Path:
    import numpy as np
    import pandas as pd
    from ultralytics import YOLO

    comp = root / "2026-cv-competition"
    test_img = comp / "test" / "test" / "images"
    sample_sub = comp / "sample_submission.csv"

    imgsz_train = int(os.environ.get("IMGSZ", "640"))
    imgsz_pred = int(os.environ.get("IMGSZ_PRED", str(imgsz_train)))
    conf_infer = float(os.environ.get("CONF_INFER", "0.02"))
    conf_thr = float(os.environ.get("CONF_THR", "0.12"))
    use_augment = os.environ.get("AUGMENT", "1") == "1"

    model = YOLO(str(best_ckpt))
    results = list(
        model.predict(
            source=str(test_img),
            imgsz=imgsz_pred,
            conf=conf_infer,
            iou=0.65,
            augment=use_augment,
            save=False,
            verbose=False,
            stream=True,
        )
    )
    print("inference images:", len(results), "augment:", use_augment, "imgsz:", imgsz_pred)

    def boxes_to_prediction_string(result, thr: float) -> str:
        im_h, im_w = result.orig_shape
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return ""
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        keep = conf >= thr
        if not np.any(keep):
            return ""
        xyxy, cls, conf = xyxy[keep], cls[keep], conf[keep]
        order = np.argsort(-conf)
        xyxy, cls = xyxy[order], cls[order]
        xyxy[:, [0, 2]] /= im_w
        xyxy[:, [1, 3]] /= im_h
        xyxy = np.clip(xyxy, 0.0, 1.0)
        parts: list[str] = []
        for i in range(len(cls)):
            parts.append(str(int(cls[i])))
            parts += [f"{v:.6f}" for v in xyxy[i]]
        return " ".join(parts)

    rows = []
    for r in results:
        path = Path(r.path)
        rows.append({"image_id": path.stem, "PredictionString": boxes_to_prediction_string(r, conf_thr)})

    pred_df = pd.DataFrame(rows)
    template = pd.read_csv(sample_sub)
    out = template[["image_id"]].merge(pred_df, on="image_id", how="left")
    out["PredictionString"] = out["PredictionString"].fillna("")

    out_path = root / "sample_submission.csv"
    out.to_csv(out_path, index=False)
    nonempty = (out["PredictionString"].str.len() > 0).sum()
    print("saved:", out_path, "rows:", len(out), "with boxes:", int(nonempty))
    return out_path


def main() -> int:
    os.environ.setdefault("MPLBACKEND", "Agg")
    set_seeds()
    root = find_repo_root()
    print("ROOT:", root)
    yaml_path = prepare_split(root)
    best = train_model(root, yaml_path)
    predict_and_write(root, best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
