# Tune the YOLO + PCU traffic-level baseline without touching the test set.
# Run from the outer project folder:  python tune_yolo.py
import os, json
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
from traffic_core import *

DATA = "traffic_project"
IMG = f"{DATA}/images"
OUT = f"{DATA}/models"
RES = f"{DATA}/results"
SEED = 42

df = pd.read_csv(f"{DATA}/labels.csv")
df["level"] = df["level"].str.strip().str.lower()
df = df[df["filename"].apply(lambda f: os.path.exists(f"{IMG}/{f}"))].reset_index(drop=True)
df["y"] = df["level"].map({c: i for i, c in enumerate(CLASSES)})
train_df, tmp_df = train_test_split(df, test_size=0.30, stratify=df["y"], random_state=SEED)
val_df, test_df = train_test_split(tmp_df, test_size=0.50, stratify=tmp_df["y"], random_state=SEED)
print("train/val/test:", len(train_df), len(val_df), len(test_df))

def boxes_for(yolo, path, conf=0.10, imgsz=640):
    # Run detection once; keep vehicle boxes (COCO classes) with their confidence.
    r = yolo(path, conf=conf, imgsz=imgsz, verbose=False)[0]
    boxes = r.boxes
    out = []
    if boxes is not None:
        clses = boxes.cls.tolist()
        scores = boxes.conf.tolist()
        for c, s in zip(clses, scores):
            ci = int(c)
            if ci in COCO:
                out.append((ci, float(s)))
    return out

def pcu_for(boxlist, conf):
    # PCU = weighted vehicle count; weights: car 1, motorcycle 0.5, bus 3, truck 3.
    return float(sum(PCU[COCO[ci]] for ci, s in boxlist if s >= conf))

def yolo_pcu(yolo, filenames, conf=0.10, imgsz=640):
    return {f: pcu_for(boxes_for(yolo, f"{IMG}/{f}", conf, imgsz), conf) for f in filenames}

results = []
for mname in ["yolov8n.pt", "yolov8s.pt"]:
    for imgsz in [640, 960]:
        yolo = YOLO(mname)
        bl = {f: boxes_for(yolo, f"{IMG}/{f}", conf=0.10, imgsz=imgsz) for f in df["filename"]}
        for conf in [0.15, 0.25, 0.35]:
            tr_p = np.array([pcu_for(bl[f], conf) for f in train_df["filename"]])
            tr_y = train_df["y"].values
            best = (0, None)
            for lo in range(1, 30):
                for hi in range(lo + 2, 80, 2):
                    acc = np.mean(np.array([pcu_level(p, lo, hi) for p in tr_p]) == tr_y)
                    if acc > best[0]:
                        best = (acc, (lo, hi))
            LOW, HIGH = best[1]
            va_pred = np.array([pcu_level(pcu_for(bl[f], conf), LOW, HIGH) for f in val_df["filename"]])
            va_y = val_df["y"].values
            va_acc = accuracy_score(va_y, va_pred)
            va_f1 = f1_score(va_y, va_pred, average="macro")
            results.append({"model": mname, "imgsz": imgsz, "conf": conf,
                            "low_thr": int(LOW), "high_thr": int(HIGH),
                            "val_acc": float(va_acc), "val_f1": float(va_f1)})

winner = None
best_vf = -1.0
for r in results:
    if r["val_f1"] > best_vf:
        best_vf = r["val_f1"]; winner = r

yolo = YOLO(winner["model"])
bl_te = {f: boxes_for(yolo, f"{IMG}/{f}", conf=winner["conf"], imgsz=winner["imgsz"]) for f in test_df["filename"]}
te_pred = np.array([pcu_level(pcu_for(bl_te[f], winner["conf"]), winner["low_thr"], winner["high_thr"]) for f in test_df["filename"]])
te_y = test_df["y"].values
yolo_acc = float(accuracy_score(te_y, te_pred))
yolo_f1 = float(f1_score(te_y, te_pred, average="macro"))

tuning = pd.DataFrame(results, columns=["model", "imgsz", "conf", "low_thr", "high_thr", "val_acc", "val_f1"])
tuning.to_csv(f"{RES}/yolo_tuning.csv", index=False)

cfg = json.load(open(f"{OUT}/config.json"))
cfg.update({
    "yolo_model": winner["model"],
    "yolo_conf": winner["conf"],
    "yolo_imgsz": winner["imgsz"],
    "low_thr": winner["low_thr"],
    "high_thr": winner["high_thr"],
    "yolo_test_acc": yolo_acc,
    "yolo_test_f1": yolo_f1,
})
json.dump(cfg, open(f"{OUT}/config.json", "w"), indent=2)

print("YOLO_TUNING_RESULTS")
print(tuning.to_string())
print("WINNER", winner)
print(f"WINNER TEST acc {yolo_acc:.4f} macro-F1 {yolo_f1:.4f}")
print("Saved", f"{RES}/yolo_tuning.csv")
print("Updated", f"{OUT}/config.json")
