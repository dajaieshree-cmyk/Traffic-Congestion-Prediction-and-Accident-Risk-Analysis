# Run from the outer project folder:  streamlit run app.py
import os, io, json, urllib.request
from datetime import datetime
import numpy as np, pandas as pd, streamlit as st, torch
from PIL import Image
from ultralytics import YOLO
from traffic_core import *

DATA = "traffic_project"
IMG = f"{DATA}/images"
MODELS = f"{DATA}/models"
RES = f"{DATA}/results"
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
LEVEL_COLOR = {"low": "#1f9d55", "medium": "#e08a00", "high": "#d64545"}

st.set_page_config(page_title="Traffic Congestion and Accident Risk", page_icon="🚦", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:1.6rem;max-width:1250px}
.hero{padding:18px 22px;border-radius:16px;background:linear-gradient(120deg,#16325c 0%,#2b6cb0 100%);color:#fff;margin-bottom:14px}
.hero h1{margin:0;font-size:1.7rem;color:#fff}.hero p{margin:4px 0 0;opacity:.9}
.card{border:1px solid rgba(128,128,128,.28);border-radius:14px;padding:16px 18px;background:rgba(128,128,128,.06);height:100%}
.card h4{margin:0 0 8px;font-size:.95rem;opacity:.75;font-weight:600;letter-spacing:.02em}
.badge{display:inline-block;padding:6px 16px;border-radius:999px;color:#fff;font-weight:700;font-size:1.15rem;letter-spacing:.04em}
.big{font-size:2.1rem;font-weight:700;line-height:1.1}
.chip{display:inline-block;margin:3px 6px 3px 0;padding:4px 11px;border-radius:999px;background:rgba(43,108,176,.14);font-size:.9rem}
.bar{height:16px;border-radius:999px;background:rgba(128,128,128,.25);overflow:hidden;margin:8px 0 6px}
.bar>div{height:100%;border-radius:999px}
.kpi{border:1px solid rgba(128,128,128,.28);border-radius:14px;padding:14px 16px;background:rgba(128,128,128,.06)}
.kpi .l{font-size:.85rem;opacity:.7}.kpi .v{font-size:1.9rem;font-weight:700}
.small{font-size:.85rem;opacity:.75}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_config():
    return json.load(open(f"{MODELS}/config.json"))


@st.cache_resource
def load_yolo():
    return YOLO(load_config().get("yolo_model", "yolov8n.pt"))


@st.cache_resource
def load_resnet():
    m = make_resnet(pretrained=False)
    m.load_state_dict(torch.load(f"{MODELS}/resnet_traffic.pt", map_location="cpu"))
    m.eval()
    return m


@st.cache_resource
def load_accident():
    meta = json.load(open(f"{MODELS}/acc_meta.json"))
    m = make_mlp(len(meta["cols"]), False, 0.3)
    m.load_state_dict(torch.load(f"{MODELS}/accident_mlp.pt", map_location="cpu"))
    m.eval()
    return m, meta


def badge(level):
    return f"<span class='badge' style='background:{LEVEL_COLOR[level]}'>{level.upper()}</span>"


if not os.path.exists(f"{MODELS}/config.json"):
    st.error("Models not found. Run  python train_models.py  once from this folder, then restart the app.")
    st.stop()

cfg = load_config()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## 🚦 Traffic AI")
    st.caption(" Deep Learning")
    st.divider()
    st.markdown("**Test results**")
    st.markdown(f"ResNet18: **{cfg['resnet_test_acc']:.3f}** accuracy")
    st.markdown(f"YOLO + PCU: **{cfg['yolo_test_acc']:.3f}** accuracy")
    st.caption("Small hand-labelled UA-DETRAC set. Results on other roads may differ.")

st.markdown("<div class='hero'><h1>Traffic Congestion Prediction and Accident Risk Analysis</h1>"
            "<p>Upload a road photo to get vehicle counts, a PCU score and a traffic level. "
            "Then check how risky the conditions are for a severe accident.</p></div>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📷  Live traffic analysis", "⚠️  Accident severity risk", "📊  Project results"])

# ------------------------------------------------------------------ tab 1
with tab1:
    c_up, c_pick = st.columns([3, 2])
    up = c_up.file_uploader("Upload a traffic photo (JPG or PNG)", type=["jpg", "jpeg", "png"])
    samples = sorted(f for f in os.listdir(IMG) if f.lower().endswith((".jpg", ".jpeg", ".png"))) if os.path.isdir(IMG) else []
    pick = c_pick.selectbox("...or use a sample frame from the dataset", ["(none)"] + samples)
    img = None
    if up is not None:
        img = Image.open(up).convert("RGB")
    elif pick != "(none)":
        img = Image.open(f"{IMG}/{pick}").convert("RGB")

    if img is None:
        st.info("Upload a photo or choose a sample frame to start.")
    else:
        with st.spinner("Running YOLO and ResNet18..."):
            counts, pcu, annotated = detect(load_yolo(), img, conf=cfg.get("yolo_conf", 0.25))
            rule = CLASSES[pcu_level(pcu, cfg["low_thr"], cfg["high_thr"])]
            with torch.no_grad():
                probs = torch.softmax(load_resnet()(tf_eval(img).unsqueeze(0)), 1)[0].numpy()
            cnn = CLASSES[int(probs.argmax())]

        left, right = st.columns([3, 2], gap="large")
        with left:
            st.image(annotated, caption="YOLOv8n detections", use_container_width=True)
            buf = io.BytesIO(); Image.fromarray(annotated).save(buf, format="PNG")
            st.download_button("⬇️ Download annotated image", buf.getvalue(), "annotated.png", "image/png")
        with right:
            chips = "".join(f"<span class='chip'><b>{v}</b> {k}</span>" for k, v in counts.items())
            st.markdown(f"<div class='card'><h4>VEHICLES DETECTED</h4>{chips}"
                        f"<div class='small' style='margin-top:10px'>PCU score</div><div class='big'>{pcu:.1f}</div></div>",
                        unsafe_allow_html=True)

        st.write("")
        k1, k2 = st.columns(2, gap="large")
        with k1:
            st.markdown(f"<div class='card'><h4>METHOD 1 · YOLO + PCU RULE</h4>{badge(rule)}"
                        f"<div class='small' style='margin-top:10px'>Cutoffs: low below {cfg['low_thr']}, "
                        f"high from {cfg['high_thr']} PCU</div></div>", unsafe_allow_html=True)
        with k2:
            st.markdown(f"<div class='card'><h4>METHOD 2 · RESNET18 CLASSIFIER</h4>{badge(cnn)}"
                        f"<div class='small' style='margin-top:10px'>Confidence {probs.max() * 100:.0f}%</div></div>",
                        unsafe_allow_html=True)
            st.bar_chart(pd.DataFrame({"probability": probs}, index=CLASSES), height=180)

        if rule == cnn:
            st.success(f"Both methods agree: {cnn.upper()} traffic.")
        else:
            st.warning("The two methods disagree. Medium traffic is the hardest class for both, "
                       "and the ResNet18 was more accurate on the test set.")

# ------------------------------------------------------------------ tab 2
with tab2:
    mlp, meta = load_accident()
    opts = meta["wx_options"]

    def pick_wx(*words):
        for o in opts:
            if any(w in o.lower() for w in words):
                return o
        return opts[0]

    def clock_defaults():
        n = datetime.now()
        return dict(a_hour=n.hour, a_dow=DAYS[n.weekday()], a_month=n.month, a_temp=82.0, a_hum=65.0, a_vis=6.0,
                    a_wind=8.0, a_prec=0.0, a_wx=pick_wx("fair", "clear"), a_junction=False, a_crossing=False,
                    a_signal=False, a_night=(n.hour < 6 or n.hour >= 19))

    for k, v in clock_defaults().items():
        st.session_state.setdefault(k, v)

    def set_vals(**kw):
        for k, v in kw.items():
            st.session_state[k] = v

    def sc_rush():
        set_vals(a_hour=18, a_temp=78.0, a_hum=50.0, a_vis=10.0, a_wind=6.0, a_prec=0.0, a_wx=pick_wx("fair", "clear"),
                 a_junction=False, a_crossing=False, a_signal=True, a_night=False)

    def sc_rain():
        set_vals(a_hour=23, a_temp=68.0, a_hum=92.0, a_vis=2.0, a_wind=15.0, a_prec=0.3, a_wx=pick_wx("heavy rain", "rain"),
                 a_junction=True, a_crossing=False, a_signal=False, a_night=True)

    def sc_fog():
        set_vals(a_hour=7, a_temp=55.0, a_hum=97.0, a_vis=0.5, a_wind=3.0, a_prec=0.0, a_wx=pick_wx("fog", "mist", "haze"),
                 a_junction=False, a_crossing=True, a_signal=False, a_night=False)

    def sc_reset():
        set_vals(**clock_defaults())

    def get_weather():
        url = ("https://api.open-meteo.com/v1/forecast?latitude=13.08&longitude=80.27"
               "&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,visibility"
               "&temperature_unit=fahrenheit&wind_speed_unit=mph&precipitation_unit=inch")
        try:
            with urllib.request.urlopen(url, timeout=6) as r:
                d = json.load(r)["current"]
            vals = dict(a_temp=float(np.clip(d["temperature_2m"], -20, 130)),
                        a_hum=float(np.clip(d["relative_humidity_2m"], 0, 100)),
                        a_wind=float(np.clip(d["wind_speed_10m"], 0, 80)),
                        a_prec=float(np.clip(d["precipitation"], 0, 5)))
            if d.get("visibility") is not None:
                vals["a_vis"] = float(np.clip(d["visibility"] / 1609.34, 0, 15))
            vals["a_wx"] = pick_wx("rain") if vals["a_prec"] > 0 else pick_wx("fair", "clear")
            set_vals(**vals)
            st.session_state["wx_msg"] = ("ok", "Live weather for Chennai loaded.")
        except Exception:
            st.session_state["wx_msg"] = ("bad", "Could not get live weather (no internet?). You can still enter values by hand.")

    MODES = ["🌐 Live now (Chennai)", "🌤️ Clear day, rush hour", "🌧️ Night, heavy rain", "🌫️ Foggy morning"]

    def apply_mode():
        m = st.session_state.get("a_mode", MODES[0])
        if m == MODES[0]:
            sc_reset()
            get_weather()
        elif m == MODES[1]:
            sc_rush()
        elif m == MODES[2]:
            sc_rain()
        else:
            sc_fog()

    def hour_changed():
        h = st.session_state["a_hour"]
        st.session_state["a_night"] = bool(h < 6 or h >= 19)

    if "a_init" not in st.session_state:          # first load: fill time from the clock and weather from the internet
        st.session_state["a_init"] = True
        st.session_state["a_mode"] = MODES[0]
        apply_mode()

    st.markdown("#### 1. Choose the conditions")
    st.radio("Conditions", MODES, key="a_mode", horizontal=True, on_change=apply_mode, label_visibility="collapsed")
    rb1, rb2, _ = st.columns([2, 1, 3])
    rb1.button("🌐 Fetch live weather (Chennai)", on_click=get_weather, use_container_width=True)
    rb2.button("↩️ Reset", on_click=sc_reset, use_container_width=True)
    msg = st.session_state.pop("wx_msg", None)
    if msg:
        (st.success if msg[0] == "ok" else st.warning)(msg[1])

    S = st.session_state
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("#### 2. Road features")
        d, e, f = st.columns(3)
        d.checkbox("Junction", key="a_junction")
        e.checkbox("Crossing", key="a_crossing")
        f.checkbox("Traffic signal", key="a_signal")
        st.caption(f"Using: {S.a_hour:02d}:00, {S.a_dow}  |  {S.a_temp:.0f} °F, humidity {S.a_hum:.0f}%, visibility {S.a_vis:.1f} mi, "
                   f"wind {S.a_wind:.0f} mph, rain {S.a_prec:.2f} in, {S.a_wx}")
        with st.expander("Fine-tune every input (optional)"):
            a, b, c = st.columns(3)
            a.slider("Hour of day", 0, 23, key="a_hour", on_change=hour_changed)
            a.selectbox("Day of week", DAYS, key="a_dow")
            a.slider("Month", 1, 12, key="a_month")
            b.number_input("Temperature (°F)", -20.0, 130.0, key="a_temp")
            b.number_input("Humidity (%)", 0.0, 100.0, key="a_hum")
            b.number_input("Visibility (miles)", 0.0, 15.0, key="a_vis")
            c.number_input("Wind speed (mph)", 0.0, 80.0, key="a_wind")
            c.number_input("Precipitation (inches)", 0.0, 5.0, key="a_prec")
            c.selectbox("Weather", opts, key="a_wx")
            st.checkbox("Night time", key="a_night")
            st.caption("Units follow the US Accidents dataset (°F, miles, mph, inches).")

    S = st.session_state
    vals = {"hour": S.a_hour, "dayofweek": DAYS.index(S.a_dow), "month": S.a_month, "Temperature(F)": S.a_temp,
            "Humidity(%)": S.a_hum, "Visibility(mi)": S.a_vis, "Wind_Speed(mph)": S.a_wind, "Precipitation(in)": S.a_prec,
            "Junction": int(S.a_junction), "Crossing": int(S.a_crossing), "Traffic_Signal": int(S.a_signal),
            "is_night": int(S.a_night)}
    row = []
    for col in meta["cols"]:
        if col.startswith("wx_"):
            v = 1.0 if col == f"wx_{S.a_wx}" else 0.0
        else:
            v = float(vals.get(col, meta["median"][col]))     # features not in the form use the training median
        row.append((v - meta["mean"][col]) / (meta["std"][col] or 1.0))
    with torch.no_grad():
        score = torch.sigmoid(mlp(torch.tensor([row], dtype=torch.float32))).item()
    pct = int(round(min(max(score, 0.0), 1.0) * 100))
    band, color = (("LOWER risk", "#1f9d55") if pct < 40 else ("MODERATE risk", "#e08a00") if pct <= 60 else ("HIGHER risk", "#d64545"))
    with right:
        st.markdown(f"<div class='card'><h4>RELATIVE SEVERITY SCORE</h4><div class='big'>{pct} <span class='small'>/ 100</span></div>"
                    f"<div class='bar'><div style='width:{pct}%;background:{color}'></div></div>"
                    f"<span class='badge' style='background:{color}'>{band}</span>"
                    f"<div class='small' style='margin-top:10px'>This is a relative ranking, not a true probability, because the "
                    f"network was trained with extra weight on severe accidents. It is a baseline model (AUC about 0.71).</div></div>",
                    unsafe_allow_html=True)

# ------------------------------------------------------------------ tab 3
with tab3:
    m = st.columns(4)
    for col, (lab, val) in zip(m, [("ResNet18 accuracy", cfg["resnet_test_acc"]), ("ResNet18 macro-F1", cfg["resnet_test_f1"]),
                                   ("YOLO + PCU accuracy", cfg["yolo_test_acc"]), ("YOLO + PCU macro-F1", cfg["yolo_test_f1"])]):
        col.markdown(f"<div class='kpi'><div class='l'>{lab}</div><div class='v'>{val:.3f}</div></div>", unsafe_allow_html=True)
    st.write("")
    st.markdown("**What the results show**")
    st.markdown("- The fine-tuned ResNet18 beats the vehicle-count rule, because it sees the whole scene.\n"
                "- Medium traffic is the hardest class for both methods.\n"
                "- Dropout lowered the validation loss of the accident network, but the test scores of all four settings were almost equal.")
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
