from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision.ops import nms
from ultralytics import YOLO

SEED = 993


def set_seed() -> None:
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def repo_root() -> Path:
    here = Path(__file__).resolve().parent.parent
    for p in [here, *here.parents]:
        if (p / "2026-cv-competition" / "sample_submission.csv").is_file():
            return p
    raise FileNotFoundError("Could not find 2026-cv-competition/sample_submission.csv")


def model_predict(model: YOLO, test_dir: Path, imgsz: int, conf: float, iou: float, augment: bool):
    return list(
        model.predict(
            source=str(test_dir),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            augment=augment,
            save=False,
            verbose=False,
            stream=True,
        )
    )


def merge_image_predictions(
    preds: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    img_w: float,
    img_h: float,
    score_thr: float,
    nms_iou: float,
    max_det: int,
) -> list[str]:
    boxes_all = []
    scores_all = []
    cls_all = []
    for boxes, scores, cls_ids in preds:
        if boxes.size == 0:
            continue
        keep = scores >= score_thr
        if not np.any(keep):
            continue
        boxes_all.append(boxes[keep])
        scores_all.append(scores[keep])
        cls_all.append(cls_ids[keep])

    if not boxes_all:
        return ["0", "0.000000", "0", "0", "0", "0"]

    boxes = np.concatenate(boxes_all, axis=0)
    scores = np.concatenate(scores_all, axis=0)
    cls_ids = np.concatenate(cls_all, axis=0)

    out_boxes = []
    out_scores = []
    out_cls = []
    for c in np.unique(cls_ids):
        idx = np.where(cls_ids == c)[0]
        b = torch.from_numpy(boxes[idx]).float()
        s = torch.from_numpy(scores[idx]).float()
        keep_idx = nms(b, s, nms_iou).cpu().numpy()
        out_boxes.append(boxes[idx][keep_idx])
        out_scores.append(scores[idx][keep_idx])
        out_cls.append(np.full(len(keep_idx), int(c), dtype=np.int32))

    boxes = np.concatenate(out_boxes, axis=0)
    scores = np.concatenate(out_scores, axis=0)
    cls_ids = np.concatenate(out_cls, axis=0)

    order = np.argsort(-scores)[:max_det]
    boxes = boxes[order]
    scores = scores[order]
    cls_ids = cls_ids[order]

    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0.0, img_w - 1.0)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0.0, img_h - 1.0)

    parts: list[str] = []
    for i in range(len(cls_ids)):
        x1, y1, x2, y2 = boxes[i]
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        parts.extend(
            [
                str(int(cls_ids[i])),
                f"{float(scores[i]):.6f}",
                f"{x1:.2f}",
                f"{y1:.2f}",
                f"{x2:.2f}",
                f"{y2:.2f}",
            ]
        )
    return parts if parts else ["0", "0.000000", "0", "0", "0", "0"]


def main() -> None:
    set_seed()
    root = repo_root()
    comp = root / "2026-cv-competition"
    test_dir = comp / "test" / "test" / "images"
    template = pd.read_csv(comp / "sample_submission.csv")[["image_id"]]

    ckpt_s = root / "runs" / "detect" / "yolov8s_submit" / "weights" / "best.pt"
    ckpt_n = root / "runs" / "detect" / "yolov8n_52cls" / "weights" / "best.pt"
    assert ckpt_s.is_file(), ckpt_s
    assert ckpt_n.is_file(), ckpt_n

    score_thr = float(os.environ.get("ENS_SCORE_THR", "0.07"))
    nms_iou = float(os.environ.get("ENS_NMS_IOU", "0.55"))
    max_det = int(os.environ.get("ENS_MAX_DET", "120"))
    w_s = float(os.environ.get("ENS_W_S", "1.0"))
    w_n = float(os.environ.get("ENS_W_N", "0.85"))

    m_s = YOLO(str(ckpt_s))
    m_n = YOLO(str(ckpt_n))

    rs = model_predict(m_s, test_dir, imgsz=960, conf=0.005, iou=0.65, augment=True)
    rn = model_predict(m_n, test_dir, imgsz=832, conf=0.005, iou=0.65, augment=True)
    assert len(rs) == len(rn), "Model outputs differ by image count"

    rows = []
    for a, b in zip(rs, rn):
        p_a = Path(a.path)
        p_b = Path(b.path)
        assert p_a.stem == p_b.stem, f"Mismatched images: {p_a.name} vs {p_b.name}"
        h, w = a.orig_shape
        pa = (
            a.boxes.xyxy.cpu().numpy() if a.boxes is not None and len(a.boxes) else np.zeros((0, 4), dtype=np.float32),
            (a.boxes.conf.cpu().numpy() * w_s) if a.boxes is not None and len(a.boxes) else np.zeros((0,), dtype=np.float32),
            a.boxes.cls.cpu().numpy().astype(np.int32) if a.boxes is not None and len(a.boxes) else np.zeros((0,), dtype=np.int32),
        )
        pb = (
            b.boxes.xyxy.cpu().numpy() if b.boxes is not None and len(b.boxes) else np.zeros((0, 4), dtype=np.float32),
            (b.boxes.conf.cpu().numpy() * w_n) if b.boxes is not None and len(b.boxes) else np.zeros((0,), dtype=np.float32),
            b.boxes.cls.cpu().numpy().astype(np.int32) if b.boxes is not None and len(b.boxes) else np.zeros((0,), dtype=np.int32),
        )
        pred = " ".join(
            merge_image_predictions(
                preds=[pa, pb],
                img_w=float(w),
                img_h=float(h),
                score_thr=score_thr,
                nms_iou=nms_iou,
                max_det=max_det,
            )
        )
        rows.append({"image_id": p_a.stem, "PredictionString": pred})

    sub = template.merge(pd.DataFrame(rows), on="image_id", how="left")
    sub["PredictionString"] = sub["PredictionString"].fillna("0 0.000000 0 0 0 0")

    out_main = root / "sample_submission.csv"
    out_aux = root / "sample_submission_ensemble.csv"
    sub.to_csv(out_main, index=False)
    sub.to_csv(out_aux, index=False)

    tok = sub["PredictionString"].astype(str).str.split().map(len)
    boxes = tok // 6
    print(
        "saved",
        out_main,
        "and",
        out_aux,
        "rows",
        len(sub),
        "nonempty",
        int((boxes > 0).sum()),
        "avg_boxes",
        round(float(boxes.mean()), 3),
    )


if __name__ == "__main__":
    main()
