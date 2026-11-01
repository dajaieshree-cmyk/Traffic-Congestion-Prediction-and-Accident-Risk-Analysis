# Run from the project folder:  streamlit run app.py
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

st.set_page_config(page_title="Traffic Congestion and Accident Risk", layout="wide")


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


CSS = """
<style>
.block-container{padding-top:1.4rem;max-width:1300px}
.title{font-size:2.05rem;font-weight:700;margin:0;letter-spacing:-.01em}
.title span{color:#2dd4bf}
.sub{color:#93a4c0;margin:2px 0 6px;font-size:1rem}
.card{background:#16223a;border:1px solid #243553;border-radius:14px;padding:16px 18px;height:100%}
.lab{font-size:.72rem;letter-spacing:.09em;color:#93a4c0;text-transform:uppercase;margin-bottom:8px;font-weight:600}
.kpi .v{font-size:2rem;font-weight:700;color:#e6edf7;line-height:1.15}
.kpi .n{font-size:.8rem;color:#93a4c0}
.pill{display:inline-block;padding:7px 18px;border-radius:999px;font-weight:700;color:#0b1220;font-size:1.05rem;letter-spacing:.05em}
.small{font-size:.82rem;color:#93a4c0;margin-top:10px}
.pr{display:flex;align-items:center;gap:10px;margin:7px 0;font-size:.85rem;color:#c7d3e6}
.pr .nm{width:62px}.pr .tr{flex:1;height:10px;border-radius:999px;background:#0e1624;overflow:hidden}
.pr .tr div{height:100%;border-radius:999px}.pr .pc{width:44px;text-align:right}
.gauge{height:14px;border-radius:999px;background:linear-gradient(90deg,#22c55e 0%,#f59e0b 50%,#ef4444 100%);position:relative;margin:18px 0 8px}
.gauge i{position:absolute;top:-6px;width:5px;height:26px;background:#fff;border-radius:3px;box-shadow:0 0 0 2px #0e1624}
.gl{display:flex;justify-content:space-between;font-size:.75rem;color:#93a4c0}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

LEVEL_COLOR = {"low": "#22c55e", "medium": "#f59e0b", "high": "#ef4444"}


def pill(level, text=None):
    return f"<span class='pill' style='background:{LEVEL_COLOR[level]}'>{text or level.upper()}</span>"


def kpi(label, value, note=""):
    return f"<div class='card kpi'><div class='lab'>{label}</div><div class='v'>{value}</div><div class='n'>{note}</div></div>"


def prob_bars(probs):
    out = ""
    for name, p in zip(CLASSES, probs):
        out += (f"<div class='pr'><span class='nm'>{name}</span><span class='tr'><div style='width:{p * 100:.0f}%;"
                f"background:{LEVEL_COLOR[name]}'></div></span><span class='pc'>{p * 100:.0f}%</span></div>")
    return out


if not os.path.exists(f"{MODELS}/config.json"):
    st.error("Models not found. Run  python train_models.py  once, then restart the app.")
    st.stop()

cfg = load_config()

# ------------------------------------------------------------------ sidebar
st.sidebar.title("About the project")
st.sidebar.write("Deep Learning  mini project")
st.sidebar.subheader("How it works")
st.sidebar.write("1. YOLOv8n detects vehicles in the photo.")
st.sidebar.write("2. The vehicle counts are converted to a PCU score and a traffic level.")
st.sidebar.write("3. A fine-tuned ResNet18 predicts the traffic level directly from the photo.")
st.sidebar.write("4. A neural network estimates how severe an accident could be from time, weather and road features.")
st.sidebar.subheader("Test accuracy")
st.sidebar.write(f"ResNet18: {cfg['resnet_test_acc']:.3f}")
st.sidebar.write(f"YOLO + PCU: {cfg['yolo_test_acc']:.3f}")

st.markdown("<div class='title'>Traffic Congestion Prediction <span>and Accident Risk Analysis</span></div>"
            "<div class='sub'>Vehicle detection, traffic level classification and accident severity estimation</div>",
            unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Traffic analysis", "Accident risk", "Results"])

# ------------------------------------------------------------------ tab 1
with tab1:
    st.header("Traffic analysis")
    st.write("Upload a road photo, or choose one from the dataset.")
    up = st.file_uploader("Upload photo (JPG or PNG)", type=["jpg", "jpeg", "png"])
    samples = sorted(f for f in os.listdir(IMG) if f.lower().endswith((".jpg", ".jpeg", ".png"))) if os.path.isdir(IMG) else []
    pick = st.selectbox("Or choose a sample image", ["(none)"] + samples)
    img = None
    if up is not None:
        img = Image.open(up).convert("RGB")
    elif pick != "(none)":
        img = Image.open(f"{IMG}/{pick}").convert("RGB")

    if img is not None:
        counts, pcu, annotated = detect(load_yolo(), img, conf=cfg.get("yolo_conf", 0.25))
        rule = CLASSES[pcu_level(pcu, cfg["low_thr"], cfg["high_thr"])]
        with torch.no_grad():
            probs = torch.softmax(load_resnet()(tf_eval(img).unsqueeze(0)), 1)[0].numpy()
        cnn = CLASSES[int(probs.argmax())]

        c1, c2 = st.columns(2)
        c1.image(img, caption="Original photo", use_container_width=True)
        c2.image(annotated, caption="Detected vehicles", use_container_width=True)
        buf = io.BytesIO()
        Image.fromarray(annotated).save(buf, format="PNG")
        st.download_button("Download annotated image", buf.getvalue(), "annotated.png", "image/png")

        st.write("")
        k = st.columns(5)
        for col, (name, n) in zip(k[:4], counts.items()):
            col.markdown(kpi(name, n), unsafe_allow_html=True)
        k[4].markdown(kpi("PCU score", f"{pcu:.1f}", "passenger car units"), unsafe_allow_html=True)

        st.write("")
        r1, r2 = st.columns(2)
        r1.markdown(f"<div class='card'><div class='lab'>Method 1 | YOLO + PCU rule</div>{pill(rule)}"
                    f"<div class='small'>Low if PCU is below {cfg['low_thr']}, high if it is {cfg['high_thr']} or more.</div></div>",
                    unsafe_allow_html=True)
        r2.markdown(f"<div class='card'><div class='lab'>Method 2 | ResNet18 classifier</div>{pill(cnn)}"
                    f"<div class='small'>Confidence {probs.max() * 100:.0f}%</div>{prob_bars(probs)}</div>",
                    unsafe_allow_html=True)
        st.write("")
        if rule == cnn:
            st.success("Both methods give the same result.")
        else:
            st.warning("The two methods give different results. Medium traffic is the hardest class for both.")

# ------------------------------------------------------------------ tab 2
with tab2:
    st.header("Accident severity risk")
    st.write("This model estimates how likely an accident is to be severe under the chosen conditions.")
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
            st.session_state["wx_msg"] = ("bad", "Could not get live weather (no internet?). Enter the values by hand.")

    MODES = ["Live now (Chennai)", "Clear day, rush hour", "Night, heavy rain", "Foggy morning"]

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

    if "a_init" not in st.session_state:
        st.session_state["a_init"] = True
        st.session_state["a_mode"] = MODES[0]
        apply_mode()

    st.subheader("Step 1: choose the conditions")
    st.radio("Conditions", MODES, key="a_mode", horizontal=True, on_change=apply_mode)
    b1, b2, _ = st.columns([2, 1, 3])
    b1.button("Fetch live weather (Chennai)", on_click=get_weather, use_container_width=True)
    b2.button("Reset", on_click=sc_reset, use_container_width=True)
    msg = st.session_state.pop("wx_msg", None)
    if msg:
        (st.success if msg[0] == "ok" else st.warning)(msg[1])

    S = st.session_state
    st.subheader("Step 2: road features")
    d, e, f = st.columns(3)
    d.checkbox("Junction", key="a_junction")
    e.checkbox("Crossing", key="a_crossing")
    f.checkbox("Traffic signal", key="a_signal")
    st.caption(f"Current inputs: {S.a_hour:02d}:00 on {S.a_dow}, {S.a_temp:.0f} F, humidity {S.a_hum:.0f}%, "
               f"visibility {S.a_vis:.1f} mi, wind {S.a_wind:.0f} mph, rain {S.a_prec:.2f} in, weather: {S.a_wx}")
    with st.expander("Change all inputs (optional)"):
        a, b, c = st.columns(3)
        a.slider("Hour of day", 0, 23, key="a_hour", on_change=hour_changed)
        a.selectbox("Day of week", DAYS, key="a_dow")
        a.slider("Month", 1, 12, key="a_month")
        b.number_input("Temperature (F)", -20.0, 130.0, key="a_temp")
        b.number_input("Humidity (%)", 0.0, 100.0, key="a_hum")
        b.number_input("Visibility (miles)", 0.0, 15.0, key="a_vis")
        c.number_input("Wind speed (mph)", 0.0, 80.0, key="a_wind")
        c.number_input("Precipitation (inches)", 0.0, 5.0, key="a_prec")
        c.selectbox("Weather", opts, key="a_wx")
        st.checkbox("Night time", key="a_night")
        st.caption("Units are the same as in the US Accidents dataset.")

    vals = {"hour": S.a_hour, "dayofweek": DAYS.index(S.a_dow), "month": S.a_month, "Temperature(F)": S.a_temp,
            "Humidity(%)": S.a_hum, "Visibility(mi)": S.a_vis, "Wind_Speed(mph)": S.a_wind, "Precipitation(in)": S.a_prec,
            "Junction": int(S.a_junction), "Crossing": int(S.a_crossing), "Traffic_Signal": int(S.a_signal),
            "is_night": int(S.a_night)}
    row = []
    for col in meta["cols"]:
        if col.startswith("wx_"):
            v = 1.0 if col == f"wx_{S.a_wx}" else 0.0
        else:
            v = float(vals.get(col, meta["median"][col]))     # inputs not in the form use the training median
        row.append((v - meta["mean"][col]) / (meta["std"][col] or 1.0))
    with torch.no_grad():
        score = torch.sigmoid(mlp(torch.tensor([row], dtype=torch.float32))).item()
    pct = int(round(min(max(score, 0.0), 1.0) * 100))

    st.subheader("Result")
    band, color = (("LOWER risk", "#22c55e") if pct < 40 else ("MODERATE risk", "#f59e0b") if pct <= 60 else ("HIGHER risk", "#ef4444"))
    st.markdown(f"<div class='card'><div class='lab'>Severity risk score</div>"
                f"<div style='display:flex;align-items:center;gap:18px'><div class='kpi'><div class='v' style='font-size:2.6rem'>{pct}"
                f"<span style='font-size:1rem;color:#93a4c0'> / 100</span></div></div>"
                f"<span class='pill' style='background:{color}'>{band}</span></div>"
                f"<div class='gauge'><i style='left:calc({pct}% - 2px)'></i></div>"
                f"<div class='gl'><span>Lower</span><span>Moderate</span><span>Higher</span></div>"
                f"<div class='small'>This is a relative score, not a true probability, because the network was trained with extra "
                f"weight on severe accidents. The model is a baseline (AUC about 0.71).</div></div>", unsafe_allow_html=True)

# ------------------------------------------------------------------ tab 3
with tab3:
    st.header("Project results")
    m = st.columns(4)
    m[0].markdown(kpi("ResNet18 accuracy", f"{cfg['resnet_test_acc']:.3f}"), unsafe_allow_html=True)
    m[1].markdown(kpi("ResNet18 macro-F1", f"{cfg['resnet_test_f1']:.3f}"), unsafe_allow_html=True)
    m[2].markdown(kpi("YOLO + PCU accuracy", f"{cfg['yolo_test_acc']:.3f}"), unsafe_allow_html=True)
    m[3].markdown(kpi("YOLO + PCU macro-F1", f"{cfg['yolo_test_f1']:.3f}"), unsafe_allow_html=True)
    st.write("")
    st.write("The fine-tuned ResNet18 does better than the vehicle-count rule. Medium traffic is the hardest class for both "
             "methods. In the accident network, dropout lowered the validation loss, but the test scores of all four "
             "settings were almost the same.")
    figs = [("fig_cm_resnet.png", "ResNet18 confusion matrix"), ("fig_cm_yolo.png", "YOLO + PCU confusion matrix"),
            ("fig_hparam.png", "Learning rate and batch size study"), ("fig_regularization.png", "Regularization comparison")]
    cols = st.columns(2)
    for i, (fn, cap) in enumerate(figs):
        p = f"{RES}/{fn}"
        if os.path.exists(p):
            cols[i % 2].image(p, caption=cap, use_container_width=True)
    p = f"{RES}/table_regularization.csv"
    if os.path.exists(p):
        st.subheader("Accident network: regularization comparison")
        st.dataframe(pd.read_csv(p, index_col=0), use_container_width=True)