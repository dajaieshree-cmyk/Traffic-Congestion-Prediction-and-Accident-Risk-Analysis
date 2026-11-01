"""Retrain ResNet18 on the video-sequence split and save group results."""
import os, json, glob
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

IMG_DIR = r"C:\Users\DELL\traffic_project\traffic_project\images"
OUT_DIR = r"C:\Users\DELL\traffic_project\traffic_project\results"
CLASSES = ["low", "medium", "high"]
SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
tf_train = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
                               transforms.ColorJitter(0.2, 0.2), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
tf_eval = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

TRAIN_FRAME = None
VAL_FRAME = None

class TrafficImages(Dataset):
    def __init__(self, frame, tf): self.f = frame["filename"].tolist(); self.y = frame["y"].tolist(); self.tf = tf
    def __len__(self): return len(self.f)
    def __getitem__(self, i):
        img = Image.open(os.path.join(IMG_DIR, self.f[i])).convert("RGB")
        return self.tf(img), self.y[i]

def loaders(bs):
    return (DataLoader(TrafficImages(TRAIN_FRAME, tf_train), batch_size=bs, shuffle=True),
            DataLoader(TrafficImages(VAL_FRAME, tf_eval), batch_size=32))

def make_resnet(dropout=0.3):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for p in m.parameters(): p.requires_grad = False
    m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, 3))
    return m.to(device)

def run_epoch(model, loader, opt=None):
    train = opt is not None
    model.train(train)
    loss_fn = nn.CrossEntropyLoss()
    tot = correct = n = 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x); loss = loss_fn(out, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(y); correct += (out.argmax(1) == y).sum().item(); n += len(y)
    return tot / n, correct / n

def fit_resnet(lr, bs, ep_head=2, ep_ft=3, dropout=0.3):
    torch.manual_seed(SEED)
    trl, vl = loaders(bs)
    m = make_resnet(dropout)
    h = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc, best_state = -1, None
    def log(a, b):
        h["train_loss"].append(a[0]); h["train_acc"].append(a[1]); h["val_loss"].append(b[0]); h["val_acc"].append(b[1])
    opt = torch.optim.Adam(m.fc.parameters(), lr=lr)
    for _ in range(ep_head):
        a = run_epoch(m, trl, opt); b = run_epoch(m, vl); log(a, b)
        if b[1] > best_acc: best_acc, best_state = b[1], {k: v.clone() for k, v in m.state_dict().items()}
    for p in m.layer4.parameters(): p.requires_grad = True
    opt = torch.optim.Adam([{"params": m.layer4.parameters(), "lr": lr / 10}, {"params": m.fc.parameters(), "lr": lr}])
    for _ in range(ep_ft):
        a = run_epoch(m, trl, opt); b = run_epoch(m, vl); log(a, b)
        if b[1] > best_acc: best_acc, best_state = b[1], {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_state)
    return m, h, best_acc

def acc_f1(y_true, y_pred):
    acc = float(np.mean(y_pred == y_true))
    f1s = []
    for c in range(3):
        tp = float(np.sum((y_true == c) & (y_pred == c)))
        fp = float(np.sum((y_pred == c) & (y_true != c)))
        fn = float(np.sum((y_true == c) & (y_pred != c)))
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0)
    return acc, float(np.mean(f1s))

# ---- load sequence split ----
split_df = pd.read_csv(os.path.join(OUT_DIR, "split_group.csv"))
train_df = split_df[split_df["subset"] == "train"].reset_index(drop=True)
val_df = split_df[split_df["subset"] == "val"].reset_index(drop=True)
test_df = split_df[split_df["subset"] == "test"].reset_index(drop=True)

def frame_from_subset(df):
    return pd.DataFrame({
        "filename": df["filename"].tolist(),
        "y": df["level"].map({"low": 0, "medium": 1, "high": 2}).astype(int).tolist(),
    })

TRAIN_FRAME = frame_from_subset(train_df)
VAL_FRAME = frame_from_subset(val_df)
TEST_FRAME = frame_from_subset(test_df)

# ---- train: lr sweep @ bs=16, then bs sweep at best lr ----
runs = []
for lr in [0.01, 0.001, 0.0001]:
    m, h, acc = fit_resnet(lr=lr, bs=16)
    runs.append({"lr": lr, "bs": 16, "val_acc": acc, "hist": h, "model": m})
best_lr = max(runs, key=lambda r: r["val_acc"])["lr"]
for bs in [8, 32]:
    m, h, acc = fit_resnet(lr=best_lr, bs=bs)
    runs.append({"lr": best_lr, "bs": bs, "val_acc": acc, "hist": h, "model": m})
best_run = max(runs, key=lambda r: r["val_acc"])
best_model = best_run["model"]; best_hist = best_run["hist"]; best_val_acc = best_run["val_acc"]
best_lr_used = best_run["lr"]; best_bs = best_run["bs"]

# ---- test on best run ----
TEST_LOADER = DataLoader(TrafficImages(TEST_FRAME, tf_eval), batch_size=32)
best_model.eval()
preds, trues = [], []
with torch.no_grad():
    for x, y in TEST_LOADER:
        p = torch.softmax(best_model(x.to(device)), 1).cpu()
        preds += p.argmax(1).tolist(); trues += y.tolist()
res_pred = np.array(preds); res_true = np.array(trues)
cm = confusion_matrix(res_true, res_pred, labels=[0, 1, 2])
res_acc = float(np.trace(cm) / cm.sum())
f1s = []
for i in range(3):
    tp = float(cm[i, i]); fp = float(cm[:, i].sum() - tp); fn = float(cm[i].sum() - tp)
    f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0)
res_f1 = float(np.mean(f1s))

ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(cmap="Blues")
plt.title("ResNet18 group (test)"); plt.savefig(os.path.join(OUT_DIR, "fig_cm_resnet_group.png"), dpi=150); plt.close()

plt.figure(figsize=(6, 4))
plt.plot(best_hist["train_loss"], label="train"); plt.plot(best_hist["val_loss"], label="val")
plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.title("ResNet18 group curves"); plt.savefig(os.path.join(OUT_DIR, "fig_resnet_curves_group.png"), dpi=150); plt.close()

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].plot([r["val_acc"] for r in runs], marker="o"); axs[0].set_title("Val acc per run"); axs[0].set_xlabel("run"); axs[0].set_ylabel("val acc")
axs[1].plot([r["val_loss"] for r in runs], marker="o"); axs[1].set_title("Val loss per run"); axs[1].set_xlabel("run"); axs[1].set_ylabel("val loss")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "fig_hparam_group.png"), dpi=150); plt.close()

wrong = [i for i, (p, t) in enumerate(zip(res_pred, res_true)) if p != t][:6]
if len(wrong) > 0:
    fig, axs = plt.subplots(1, len(wrong), figsize=(3 * len(wrong), 3.2), squeeze=False)
    for ax, i in zip(axs[0], wrong): ax.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "fig_errors_group.png"), dpi=150); plt.close()

# YOLO + PCU on same test images
det = pd.read_csv(os.path.join(OUT_DIR, "detections.csv"))
det["seq"] = det["filename"].str.split("_img").str[0]
test_det = det[det["seq"].isin(sorted(set(test_df["seq"].tolist())))]
train_det = det[det["seq"].isin(sorted(set(train_df["seq"].tolist())))]

def predict_levels(pcu, LOW, HIGH):
    return np.where(pcu < LOW, 0, np.where(pcu < HIGH, 1, 2))

def acc_f1(y_true, y_pred):
    acc = float(np.mean(y_pred == y_true))
    f1s = []
    for c in range(3):
        tp = float(np.sum((y_true == c) & (y_pred == c)))
        fp = float(np.sum((y_pred == c) & (y_true != c)))
        fn = float(np.sum((y_true == c) & (y_pred != c)))
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0)
    return acc, float(np.mean(f1s))

best_fit = (-1.0, None, None)
for LOW in range(1, 30):
    for HIGH in range(LOW + 2, 80, 2):
        a, _ = acc_f1(train_det["y"].to_numpy(), predict_levels(train_det["pcu"].to_numpy(), LOW, HIGH))
        if a > best_fit[0]:
            best_fit = (a, LOW, HIGH)
pcu_train_acc, LOW, HIGH = best_fit
pred_test = predict_levels(test_det["pcu"].to_numpy(), LOW, HIGH)
pcu_test_acc, pcu_test_f1 = acc_f1(test_det["y"].to_numpy(), pred_test)

results_group = {
    "resnet_group": {
        "test_acc": float(res_acc),
        "test_macro_f1": float(res_f1),
        "confusion_matrix": cm.tolist(),
        "runs": [{"lr": float(r["lr"]), "bs": int(r["bs"]), "val_acc": float(r["val_acc"])} for r in runs],
        "best_val_acc": float(best_val_acc),
        "best_config": {"lr": float(best_lr_used), "bs": int(best_bs)},
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "test_sequences": sorted(set(test_df["seq"].tolist())),
        "class_counts": {
            sub: {c: int((split_df[split_df["subset"] == sub]["level"] == c).sum()) for c in ["low", "medium", "high"]}
            for sub in ["train", "val", "test"]
        },
    },
    "yolo_pcu": {
        "low_thr": int(LOW),
        "high_thr": int(HIGH),
        "test_acc": float(pcu_test_acc),
        "test_macro_f1": float(pcu_test_f1),
        "test_sequences": sorted(set(test_det["seq"].tolist())),
    },
}
with open(os.path.join(OUT_DIR, "results_group.json"), "w") as f:
    json.dump(results_group, f, indent=2)

print("RESULTS_GROUP_WRITTEN")
print(json.dumps(results_group, indent=2))
print("ResNet test acc:", res_acc, "macro_f1:", res_f1)
print("YOLO+PCU test acc:", pcu_test_acc, "macro_f1:", pcu_test_f1)
print("runs:", runs)
print("best val acc:", best_val_acc, "best config:", best_lr_used, best_bs)
print("n train/val/test:", len(train_df), len(val_df), len(test_df))
print("test sequences:", sorted(set(test_df["seq"].tolist())))
print("class counts:", {sub: {c: int(((split_df[split_df["subset"] == sub])["level"] == c).sum()) for c in ["low", "medium", "high"]} for sub in ["train", "val", "test"]})
