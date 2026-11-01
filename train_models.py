# Run once from the outer project folder:  python train_models.py
# Trains the final ResNet18 and the accident network and saves them to traffic_project/models/
import os, json, copy, random
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from ultralytics import YOLO
from traffic_core import *

DATA = "traffic_project"
IMG = f"{DATA}/images"
OUT = f"{DATA}/models"
os.makedirs(OUT, exist_ok=True)
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# ---------------- 1. labels and split (70/15/15, stratified, same rule as the notebook) ----------------
df = pd.read_csv(f"{DATA}/labels.csv")
df["level"] = df["level"].str.strip().str.lower()
df = df[df["filename"].apply(lambda f: os.path.exists(f"{IMG}/{f}"))].reset_index(drop=True)
df["y"] = df["level"].map({c: i for i, c in enumerate(CLASSES)})
train_df, tmp_df = train_test_split(df, test_size=0.30, stratify=df["y"], random_state=SEED)
val_df, test_df = train_test_split(tmp_df, test_size=0.50, stratify=tmp_df["y"], random_state=SEED)
print("train/val/test:", len(train_df), len(val_df), len(test_df))

# ---------------- 2. YOLO + PCU thresholds (chosen on TRAIN images only) ----------------
yolo = YOLO("yolov8n.pt")
pcu = {}
for f in df["filename"]:
    pcu[f] = detect(yolo, f"{IMG}/{f}")[1]
tr_p = np.array([pcu[f] for f in train_df["filename"]]); tr_y = train_df["y"].values
best = (0, None)
for lo in range(1, 30):
    for hi in range(lo + 2, 80, 2):
        acc = np.mean(np.array([pcu_level(p, lo, hi) for p in tr_p]) == tr_y)
        if acc > best[0]:
            best = (acc, (lo, hi))
LOW, HIGH = best[1]
te_pred = [pcu_level(pcu[f], LOW, HIGH) for f in test_df["filename"]]
yolo_acc = accuracy_score(test_df["y"], te_pred)
yolo_f1 = f1_score(test_df["y"], te_pred, average="macro")
print(f"PCU thresholds (put these in the paper): LOW={LOW}, HIGH={HIGH}")
print(f"YOLO+PCU test: acc {yolo_acc:.3f}, macro-F1 {yolo_f1:.3f}")

# ---------------- 3. ResNet18 two-stage fine-tuning ----------------
class Frames(Dataset):
    def __init__(self, frame, tf):
        self.f, self.y, self.tf = frame["filename"].tolist(), frame["y"].tolist(), tf
    def __len__(self): return len(self.f)
    def __getitem__(self, i):
        return self.tf(Image.open(f"{IMG}/{self.f[i]}").convert("RGB")), self.y[i]

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

LR, BS = 1e-3, 16
trl = DataLoader(Frames(train_df, tf_train), batch_size=BS, shuffle=True)
vl = DataLoader(Frames(val_df, tf_eval), batch_size=32)
tel = DataLoader(Frames(test_df, tf_eval), batch_size=32)
m = make_resnet(0.3).to(device)
for n_, p in m.named_parameters():
    p.requires_grad = n_.startswith("fc")
best_acc, best_state = -1, None
opt = torch.optim.Adam(m.fc.parameters(), lr=LR)
for ep in range(2):
    run_epoch(m, trl, opt); b = run_epoch(m, vl)
    print("stage 1 epoch", ep + 1, "val acc", round(b[1], 3))
    if b[1] > best_acc: best_acc, best_state = b[1], copy.deepcopy(m.state_dict())
for p in m.layer4.parameters(): p.requires_grad = True
opt = torch.optim.Adam([{"params": m.layer4.parameters(), "lr": LR / 10}, {"params": m.fc.parameters(), "lr": LR}])
for ep in range(3):
    run_epoch(m, trl, opt); b = run_epoch(m, vl)
    print("stage 2 epoch", ep + 1, "val acc", round(b[1], 3))
    if b[1] > best_acc: best_acc, best_state = b[1], copy.deepcopy(m.state_dict())
m.load_state_dict(best_state)
m.eval(); preds, trues = [], []
with torch.no_grad():
    for x, y in tel:
        preds += m(x.to(device)).argmax(1).cpu().tolist(); trues += y.tolist()
res_acc = accuracy_score(trues, preds); res_f1 = f1_score(trues, preds, average="macro")
print(f"ResNet18 test: acc {res_acc:.3f}, macro-F1 {res_f1:.3f}")
torch.save({k: v.cpu() for k, v in m.state_dict().items()}, f"{OUT}/resnet_traffic.pt")
json.dump({"low_thr": LOW, "high_thr": HIGH, "classes": CLASSES,
           "resnet_test_acc": res_acc, "resnet_test_f1": res_f1,
           "yolo_test_acc": yolo_acc, "yolo_test_f1": yolo_f1}, open(f"{OUT}/config.json", "w"), indent=2)

# ---------------- 4. accident severity network (dropout 0.3) ----------------
acc_df = pd.read_csv(f"{DATA}/accidents.csv")
if len(acc_df) > 100000:
    acc_df = acc_df.sample(100000, random_state=SEED)
acc_df = acc_df.reset_index(drop=True)
t = pd.to_datetime(acc_df["Start_Time"], format="mixed", errors="coerce")
X = pd.DataFrame({"hour": t.dt.hour, "dayofweek": t.dt.dayofweek, "month": t.dt.month})
for c in ["Temperature(F)", "Humidity(%)", "Pressure(in)", "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)"]:
    if c in acc_df: X[c] = acc_df[c]
for c in ["Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit", "Railway", "Roundabout",
          "Station", "Stop", "Traffic_Calming", "Traffic_Signal"]:
    if c in acc_df: X[c] = acc_df[c].astype(int)
if "Sunrise_Sunset" in acc_df: X["is_night"] = (acc_df["Sunrise_Sunset"] == "Night").astype(int)
top = list(acc_df["Weather_Condition"].value_counts().index[:10])
wx = acc_df["Weather_Condition"].where(acc_df["Weather_Condition"].isin(top), "Other")
X = pd.concat([X, pd.get_dummies(wx, prefix="wx").astype(int)], axis=1)
y = (acc_df["Severity"] >= 3).astype(int).values

Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.50, stratify=ytmp, random_state=SEED)
med = Xtr.median(); Xtr, Xva, Xte = Xtr.fillna(med), Xva.fillna(med), Xte.fillna(med)
mu, sd = Xtr.mean(), Xtr.std().replace(0, 1)
to_t = lambda a: torch.tensor(((a - mu) / sd).values, dtype=torch.float32)
Xtr_t, Xva_t, Xte_t = to_t(Xtr), to_t(Xva), to_t(Xte)
ytr_t, yva_t = torch.tensor(ytr, dtype=torch.float32), torch.tensor(yva, dtype=torch.float32)
pos_w = torch.tensor([(1 - ytr.mean()) / ytr.mean()], dtype=torch.float32).to(device)

torch.manual_seed(SEED)
mlp = make_mlp(Xtr_t.shape[1], False, 0.3).to(device)
opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
dl = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=256, shuffle=True)
best_vl, best_state = 1e9, None
for ep in range(30):
    mlp.train()
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        loss = loss_fn(mlp(xb).squeeze(1), yb)
        opt.zero_grad(); loss.backward(); opt.step()
    mlp.eval()
    with torch.no_grad():
        v = loss_fn(mlp(Xva_t.to(device)).squeeze(1), yva_t.to(device)).item()
    if v < best_vl: best_vl, best_state = v, copy.deepcopy(mlp.state_dict())
mlp.load_state_dict(best_state); mlp.eval()
with torch.no_grad():
    pr = torch.sigmoid(mlp(Xte_t.to(device)).squeeze(1)).cpu().numpy()
print(f"Accident network test: acc {accuracy_score(yte, pr > 0.5):.3f}, F1 {f1_score(yte, pr > 0.5):.3f}, AUC {roc_auc_score(yte, pr):.3f}")
print("severe share in sample:", round(float(y.mean()), 3), "(put this in the paper)")
torch.save({k: v.cpu() for k, v in mlp.state_dict().items()}, f"{OUT}/accident_mlp.pt")
json.dump({"cols": list(X.columns), "wx_options": top + ["Other"], "median": med.to_dict(),
           "mean": mu.to_dict(), "std": sd.to_dict(), "severe_share": float(y.mean())},
          open(f"{OUT}/acc_meta.json", "w"), indent=2)
print("Saved everything to", OUT)
