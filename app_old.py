# Run from the outer project folder:  streamlit run app.py
import os, json
import numpy as np, pandas as pd, streamlit as st, torch
from PIL import Image
from ultralytics import YOLO
from traffic_core import *

DATA = "traffic_project"
IMG = f"{DATA}/images"
MODELS = f"{DATA}/models"
RES = f"{DATA}/results"

st.set_page_config(page_title="Traffic Congestion and Accident Risk", page_icon="🚦", layout="wide")


@st.cache_resource
def load_yolo():
    return YOLO(load_config()["yolo_model"])


@st.cache_resource
def load_resnet():
    m = make_resnet(pretrained=False)
    m.load_state_dict(torch.load(f"{MODELS}/resnet_traffic.pt", map_location="cpu"))
    m.eval()
    return m


@st.cache_resource
def load_config():
    return json.load(open(f"{MODELS}/config.json"))


@st.cache_resource
def load_accident():
    meta = json.load(open(f"{MODELS}/acc_meta.json"))
    m = make_mlp(len(meta["cols"]), False, 0.3)
    m.load_state_dict(torch.load(f"{MODELS}/accident_mlp.pt", map_location="cpu"))
    m.eval()
    return m, meta


st.title("🚦 Traffic Congestion Prediction and Accident Risk Analysis")
st.caption("ADA23CT301 Deep Learning project. YOLOv8n + PCU, fine-tuned ResNet18, and a fully connected accident severity network.")

if not os.path.exists(f"{MODELS}/config.json"):
    st.error("Models not found. Run  python train_models.py  once from this folder, then restart the app.")
    st.stop()

cfg = load_config()
tab1, tab2, tab3 = st.tabs(["Live traffic analysis", "Accident severity risk", "Project results"])

# ------------------------------------------------------------------ tab 1
with tab1:
    st.subheader("Upload a traffic photo")
    up = st.file_uploader("JPG or PNG", type=["jpg", "jpeg", "png"])
    samples = sorted(f for f in os.listdir(IMG) if f.lower().endswith((".jpg", ".jpeg", ".png"))) if os.path.isdir(IMG) else []
    pick = st.selectbox("...or pick a sample frame from your dataset", ["(none)"] + samples)
    img = None
    if up is not None:
        img = Image.open(up).convert("RGB")
    elif pick != "(none)":
        img = Image.open(f"{IMG}/{pick}").convert("RGB")

    if img is not None:
        with st.spinner("Running YOLO and ResNet18..."):
            counts, pcu, annotated = detect(load_yolo(), img, conf=cfg["yolo_conf"], imgsz=cfg["yolo_imgsz"])
            rule = CLASSES[pcu_level(pcu, cfg["low_thr"], cfg["high_thr"])]
            with torch.no_grad():
                probs = torch.softmax(load_resnet()(tf_eval(img).unsqueeze(0)), 1)[0].numpy()
            cnn = CLASSES[int(probs.argmax())]
        left, right = st.columns([3, 2])
        with left:
            st.image(annotated, caption="YOLOv8n detections", use_container_width=True)
        with right:
            st.markdown("**Vehicles detected**")
            c = st.columns(4)
            for col, k in zip(c, counts):
                col.metric(k, counts[k])
            st.metric("PCU score", f"{pcu:.1f}")
            st.markdown(f"**Density by PCU rule:** {rule.upper()}  (cutoffs {cfg['low_thr']} and {cfg['high_thr']})")
            st.markdown(f"**ResNet18 prediction:** {cnn.upper()}  ({probs.max() * 100:.0f}% confidence)")
            st.bar_chart(pd.DataFrame({"probability": probs}, index=CLASSES))
            if rule == cnn:
                st.success("Both methods agree.")
            else:
                st.warning("The two methods disagree. Medium traffic is the hardest class for both.")
        st.caption("Both models were trained on a small set of hand-labelled UA-DETRAC frames, so results on other roads may differ.")
    else:
        st.info("Upload a photo or choose a sample frame to see the detection, PCU score and traffic level.")

# ------------------------------------------------------------------ tab 2
with tab2:
    st.subheader("Accident severity risk")
    st.caption("Given the conditions, how likely is an accident to be severe (Severity 3 or 4)? Units follow the US Accidents dataset.")
    mlp, meta = load_accident()
    a, b, c3 = st.columns(3)
    hour = a.slider("Hour of day", 0, 23, 18)
    dow = a.selectbox("Day of week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], index=4)
    month = a.slider("Month", 1, 12, 10)
    temp = b.number_input("Temperature (°F)", -20.0, 130.0, 75.0)
    hum = b.number_input("Humidity (%)", 0.0, 100.0, 60.0)
    vis = b.number_input("Visibility (miles)", 0.0, 15.0, 10.0)
    wind = c3.number_input("Wind speed (mph)", 0.0, 80.0, 5.0)
    prec = c3.number_input("Precipitation (inches)", 0.0, 5.0, 0.0)
    wx = c3.selectbox("Weather", meta["wx_options"])
    d, e, f = st.columns(3)
    junction = d.checkbox("Junction")
    crossing = e.checkbox("Crossing")
    signal = f.checkbox("Traffic signal")
    night = st.checkbox("Night time", value=hour < 6 or hour >= 19)

    vals = {"hour": hour, "dayofweek": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(dow),
            "month": month, "Temperature(F)": temp, "Humidity(%)": hum, "Visibility(mi)": vis,
            "Wind_Speed(mph)": wind, "Precipitation(in)": prec, "Junction": int(junction),
            "Crossing": int(crossing), "Traffic_Signal": int(signal), "is_night": int(night)}
    row = []
    for col in meta["cols"]:
        if col.startswith("wx_"):
            v = 1.0 if col == f"wx_{wx}" else 0.0
        else:
            v = float(vals.get(col, meta["median"][col]))     # features not in the form use the training median
        row.append((v - meta["mean"][col]) / (meta["std"][col] or 1.0))
    with torch.no_grad():
        score = torch.sigmoid(mlp(torch.tensor([row], dtype=torch.float32))).item()
    st.progress(min(max(score, 0.0), 1.0))
    band = "HIGHER risk" if score >= 0.5 else "LOWER risk"
    st.metric("Relative severity score", f"{score * 100:.0f} / 100", band)
    st.caption("The score is a relative ranking, not a true probability, because the network was trained with extra weight on severe accidents. The model is a baseline (AUC about 0.71).")

# ------------------------------------------------------------------ tab 3
with tab3:
    st.subheader("Project results")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ResNet18 accuracy", f"{cfg['resnet_test_acc']:.3f}")
    m2.metric("ResNet18 macro-F1", f"{cfg['resnet_test_f1']:.3f}")
    m3.metric("YOLO + PCU accuracy", f"{cfg['yolo_test_acc']:.3f}")
    m4.metric("YOLO + PCU macro-F1", f"{cfg['yolo_test_f1']:.3f}")
    figs = [("fig_cm_resnet.png", "ResNet18 confusion matrix"), ("fig_cm_yolo.png", "YOLO + PCU confusion matrix"),
            ("fig_hparam.png", "Learning rate and batch size study"), ("fig_regularization.png", "Regularization comparison")]
    cols = st.columns(2)
    for i, (fn, cap) in enumerate(figs):
        p = f"{RES}/{fn}"
        if os.path.exists(p):
            cols[i % 2].image(p, caption=cap, use_container_width=True)
    p = f"{RES}/table_regularization.csv"
    if os.path.exists(p):
        st.markdown("**Accident network: regularization comparison**")
        st.dataframe(pd.read_csv(p, index_col=0), use_container_width=True)
