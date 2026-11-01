import numpy as np
import torch.nn as nn
from torchvision import models, transforms

CLASSES = ["low", "medium", "high"]
COCO = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
PCU = {"car": 1.0, "motorcycle": 0.5, "bus": 3.0, "truck": 3.0}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

tf_eval = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
tf_train = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
                               transforms.ColorJitter(0.2, 0.2), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def make_resnet(dropout=0.3, pretrained=True):
    w = models.ResNet18_Weights.DEFAULT if pretrained else None
    m = models.resnet18(weights=w)
    m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, 3))
    return m


def make_mlp(d, use_bn, p):
    layers, prev = [], d
    for w in [128, 64]:
        layers.append(nn.Linear(prev, w))
        if use_bn:
            layers.append(nn.BatchNorm1d(w))
        layers.append(nn.ReLU())
        if p > 0:
            layers.append(nn.Dropout(p))
        prev = w
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def detect(yolo, img, conf=0.25):
    # img: PIL image or file path. Returns counts per vehicle type, PCU, annotated RGB image.
    r = yolo(img, conf=conf, verbose=False)[0]
    counts = {n: 0 for n in COCO.values()}
    for c in r.boxes.cls.tolist():
        if int(c) in COCO:
            counts[COCO[int(c)]] += 1
    pcu = sum(counts[k] * PCU[k] for k in counts)
    annotated = r.plot()[:, :, ::-1]   # BGR to RGB
    return counts, pcu, np.ascontiguousarray(annotated)


def pcu_level(p, lo, hi):
    return 0 if p < lo else (1 if p < hi else 2)
