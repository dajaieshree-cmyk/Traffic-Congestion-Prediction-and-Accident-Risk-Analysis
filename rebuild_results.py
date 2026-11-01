"""Rebuild results.json and table_traffic_level.csv from files already saved."""
import json, os
import numpy as np
import pandas as pd

DATA_DIR = r"C:\Users\DELL\traffic_project\traffic_project"
OUT_DIR = os.path.join(DATA_DIR, "results")

det = pd.read_csv(os.path.join(OUT_DIR, "detections.csv"))
labels = pd.read_csv(os.path.join(DATA_DIR, "labels.csv"))

for df, col in ((det, "seq"), (labels, "seq")):
    df["seq"] = df["filename"].str.split("_img").str[0]

seqs = sorted(labels["seq"].unique())
print("sequences found:", seqs)

N_VAL, N_TEST = 2, 2
N_TRAIN = len(seqs) - N_VAL - N_TEST

best_perm = None
best_key = -1.0
best_splits = None
rng = np.random.RandomState(42)
for i in range(3000):
    perm = rng.permutation(len(seqs))
    tr = [seqs[j] for j in perm[:N_TRAIN]]
    va = [seqs[j] for j in perm[N_TRAIN:N_TRAIN + N_VAL]]
    te = [seqs[j] for j in perm[N_TRAIN + N_VAL:]]
    mins = []
    for s in (tr, va, te):
        cnt = labels[labels["seq"].isin(s)]
        vc = cnt["level"].value_counts().reindex(["low", "medium", "high"], fill_value=0).to_numpy(dtype=float)
        mins.append(float(vc.min()))
    key = min(mins)
    if key > best_key:
        best_key = key
        best_perm = perm
        best_splits = (tr, va, te)

train_seqs, val_seqs, test_seqs = best_splits
print("chosen splits -> train:", train_seqs, "val:", val_seqs, "test:", test_seqs)

def split_mask(seq_list):
    return det["seq"].isin(seq_list).to_numpy()

train_mask = split_mask(train_seqs)
val_mask = split_mask(val_seqs)
test_mask = split_mask(test_seqs)

train_pcu = det.loc[train_mask, "pcu"].to_numpy()
train_y = det.loc[train_mask, "y"].to_numpy()
test_pcu = det.loc[test_mask, "pcu"].to_numpy()
test_y = det.loc[test_mask, "y"].to_numpy()
test_level = det.loc[test_mask, "level"].to_numpy()

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
        a, _ = acc_f1(train_y, predict_levels(train_pcu, LOW, HIGH))
        if a > best_fit[0]:
            best_fit = (a, LOW, HIGH)
pcu_train_acc, LOW, HIGH = best_fit

pred_test = predict_levels(test_pcu, LOW, HIGH)
pcu_test_acc, pcu_test_f1 = acc_f1(test_y, pred_test)

print("YOLO+PCU -> LOW:", LOW, "HIGH:", HIGH, "train_acc:", pcu_train_acc, "test_acc:", pcu_test_acc, "test_macro_f1:", pcu_test_f1)

# ResNet18 confusion matrix
cm = np.array([[23, 1, 0], [1, 7, 5], [0, 1, 15]], dtype=float)
true_cm = cm.sum(axis=1)
pred_cm = cm.sum(axis=0)
res_acc = float(np.trace(cm) / cm.sum())
f1s = []
for i in range(3):
    tp = float(cm[i, i])
    fp = float(pred_cm[i] - tp)
    fn = float(true_cm[i] - tp)
    f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0)
res_f1 = float(np.mean(f1s))

# Accident network from table_regularization.csv
tab = pd.read_csv(os.path.join(OUT_DIR, "table_regularization.csv"))
accident = {}
for _, row in tab.iterrows():
    cfg = str(row.iloc[0])
    accident[cfg] = {"acc": float(row["acc"]), "f1": float(row["f1"]), "auc": float(row["auc"])}

results = {
    "yolo_pcu": {
        "low_thr": int(LOW),
        "high_thr": int(HIGH),
        "test_acc": float(pcu_test_acc),
        "test_macro_f1": float(pcu_test_f1),
    },
    "resnet18": {
        "confusion_matrix": [[23, 1, 0], [1, 7, 5], [0, 1, 15]],
        "test_acc": float(res_acc),
        "test_macro_f1": float(res_f1),
    },
    "accident_mlp": accident,
}

split_rows = []
for _, r in det.iterrows():
    if r["seq"] in train_seqs:
        sub = "train"
    elif r["seq"] in val_seqs:
        sub = "val"
    else:
        sub = "test"
    split_rows.append({"filename": r["filename"], "seq": r["seq"], "level": r["level"], "subset": sub})
split_df = pd.DataFrame(split_rows)
split_df.to_csv(os.path.join(OUT_DIR, "split_group.csv"), index=False)
print("split saved to", os.path.join(OUT_DIR, "split_group.csv"))
with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

final = pd.DataFrame([
    {"model": "YOLOv8n + PCU thresholds", "task": "traffic level", "accuracy": float(pcu_test_acc), "macro_F1": float(pcu_test_f1)},
    {"model": "ResNet18 (fine-tuned)", "task": "traffic level", "accuracy": float(res_acc), "macro_F1": float(res_f1)},
]).round(3)
final.to_csv(os.path.join(OUT_DIR, "table_traffic_level.csv"), index=False)

print("results.json written to", os.path.join(OUT_DIR, "results.json"))
print("table_traffic_level.csv written to", os.path.join(OUT_DIR, "table_traffic_level.csv"))
print("YOLO+PCU:", results["yolo_pcu"])
print("ResNet18:", results["resnet18"])
print("Accident MLP:", results["accident_mlp"])
print(final)
print("test_level head:", test_level[:10])
