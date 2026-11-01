import os, csv, tkinter as tk
from PIL import Image, ImageTk

IMG_DIR = os.path.join("traffic_project", "images")
CSV_PATH = os.path.join("traffic_project", "labels.csv")
LEVELS = {"1": "low", "2": "medium", "3": "high"}

if not os.path.isdir(IMG_DIR):
    raise SystemExit("Folder not found: " + os.path.abspath(IMG_DIR))
images = sorted(f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")))
if not images:
    raise SystemExit("No images found in " + os.path.abspath(IMG_DIR))

labels = {}
if os.path.exists(CSV_PATH):
    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            labels[row["filename"]] = row["level"]

def save():
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "level"])
        for fn in images:
            if fn in labels:
                w.writerow([fn, labels[fn]])

root = tk.Tk()
root.title("Traffic labelling")
panel = tk.Label(root)
panel.pack()
status = tk.Label(root, font=("Arial", 12))
status.pack(pady=6)

state = {"i": 0, "photo": None}
while state["i"] < len(images) and images[state["i"]] in labels:
    state["i"] += 1

def show():
    i = state["i"]
    if i >= len(images):
        save()
        panel.config(image="")
        status.config(text="No more images. Press q to close.")
        return
    fn = images[i]
    img = Image.open(os.path.join(IMG_DIR, fn))
    img.thumbnail((950, 600))
    state["photo"] = ImageTk.PhotoImage(img)
    panel.config(image=state["photo"])
    status.config(text=f"{fn}   current: {labels.get(fn, '-')}   labelled {len(labels)}/{len(images)}\n"
                       "1 = low    2 = medium    3 = high    s = skip    b = back    q = quit")

def advance():
    i = state["i"] + 1
    while i < len(images) and images[i] in labels:
        i += 1
    state["i"] = i
    show()

def on_key(event):
    k = event.char.lower()
    i = state["i"]
    if k in LEVELS and i < len(images):
        labels[images[i]] = LEVELS[k]
        save()
        advance()
    elif k == "s" and i < len(images):
        advance()
    elif k == "b":
        state["i"] = max(0, i - 1)
        show()
    elif k == "q":
        save()
        root.destroy()

root.bind("<Key>", on_key)
root.focus_force()
show()
root.mainloop()