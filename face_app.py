import os
import sys
import time
import pickle
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

DETECTOR_PATH = "models/face_detection_yunet_2023mar.onnx"
RECOGNIZER_PATH = "models/face_recognition_sface_2021dec.onnx"
LANDMARK_PATH = "models/face_landmarker.task"
DB_PATH = "faces.pkl"

NUM_CAPTURES = 10
CAPTURE_GAP = 0.6
MATCH_THRESHOLD = 0.363

WINDOW = "Face Recognition"
BTN_ADD = (10, 10, 140, 40)
BTN_REMOVE = (160, 10, 170, 40)

# Animation timing (seconds after a face first appears)
T_TRACE = 0.8           # outline draws itself + scan line starts
T_ANALYZE_END = 2.8     # analysis overlay stays until here
T_FADE = 0.6            # analysis overlay fades out

# Colors (BGR)
SCAN = (255, 220, 0)
KNOWN = (0, 255, 170)
UNKNOWN = (60, 60, 255)

# Special name that gets a compliment panel (compared in lowercase)
COMPLIMENT_NAME = "ravi"
COMPLIMENT_ROWS = [
    (0.9, "EYES", "Striking"),
    (1.2, "BROWS", "Perfectly shaped"),
    (1.5, "NOSE", "Refined"),
    (1.8, "SMILE", "Warm & charming"),
    (2.1, "JAW", "Excellent"),
    (2.4, "LOOKS", "Great overall!"),
]

# MediaPipe face mesh landmark indices
OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
        379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
        234, 127, 162, 21, 54, 103, 67, 109]
LEFT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386,
            387, 388, 466]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159,
             160, 161, 246]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269,
        267, 0, 37, 39, 40, 185]
R_BROW = [70, 63, 105, 66, 107]
L_BROW = [300, 293, 334, 296, 336]
NOSE_BRIDGE = [168, 6, 197, 195, 5, 4]
NOSE_BASE = [129, 98, 97, 2, 326, 327, 358]

_last_ts = 0


# ---------------------------------------------------------------- models
def load_models(frame_size):
    detector = cv2.FaceDetectorYN.create(
        DETECTOR_PATH, "", frame_size, score_threshold=0.8
    )
    recognizer = cv2.FaceRecognizerSF.create(RECOGNIZER_PATH, "")
    return detector, recognizer


def create_landmarker():
    with open(LANDMARK_PATH, "rb") as f:
        model_bytes = f.read()
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_buffer=model_bytes),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=4,
    )
    return vision.FaceLandmarker.create_from_options(options)


def landmarks_for(landmarker, frame):
    """Returns a list of (478, 2) pixel-coordinate arrays, one per face."""
    global _last_ts
    ts = int(time.monotonic() * 1000)
    if ts <= _last_ts:
        ts = _last_ts + 1
    _last_ts = ts
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, ts)
    h, w = frame.shape[:2]
    out = []
    for lm in result.face_landmarks:
        out.append(np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32))
    return out


# -------------------------------------------------------------- database
def load_db():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def save_db(db):
    with open(DB_PATH, "wb") as f:
        pickle.dump(db, f)


def open_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open the camera.")
        sys.exit(1)
    return cap


# --------------------------------------------------------------- helpers
def largest_face(faces):
    if faces is None or len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def match_yunet(faces, center):
    """Find the YuNet detection that belongs to a MediaPipe face."""
    if faces is None or len(faces) == 0:
        return None
    best, bd = None, 1e9
    for f in faces:
        x, y, w, h = f[:4]
        d = float(np.hypot(x + w / 2 - center[0], y + h / 2 - center[1]))
        if d < bd:
            bd, best = d, f
    if best is not None and bd < best[2]:
        return best
    return None


def identify(recognizer, db, frame, face):
    aligned = recognizer.alignCrop(frame, face)
    feat = recognizer.feature(aligned)
    best_name, best_score = "Unknown", 0.0
    for name, feats in db.items():
        for known in feats:
            score = recognizer.match(
                feat, known.reshape(1, -1), cv2.FaceRecognizerSF_FR_COSINE
            )
            if score > best_score:
                best_score, best_name = score, name
    if best_score < MATCH_THRESHOLD:
        best_name = "Unknown"
    return best_name


def inside(pt, rect):
    if pt is None:
        return False
    x, y, w, h = rect
    return x <= pt[0] <= x + w and y <= pt[1] <= y + h


def draw_button(img, rect, text, color=(70, 70, 70)):
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 1)
    cv2.putText(img, text, (x + 10, y + 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def on_mouse(event, x, y, flags, state):
    if event == cv2.EVENT_LBUTTONDOWN:
        state["click"] = (x, y)


def put_text_bg(img, text, org, scale=0.6, color=(255, 255, 255),
                bg=(20, 20, 20), pad=6, thickness=1):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale,
                                     thickness)
    x, y = org
    cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + base + pad),
                  bg, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA)


# --------------------------------------------------------------- drawing
def poly(img, pts, idx, color, thickness=1, closed=False, frac=1.0):
    p = pts[idx].astype(np.int32)
    n = max(2, int(len(p) * frac))
    cv2.polylines(img, [p[:n].reshape(-1, 1, 2)],
                  closed and frac >= 1.0, color, thickness, cv2.LINE_AA)


def center_of(pts, idx):
    return pts[idx].mean(axis=0)


def line(img, a, b, color, thickness=1):
    cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
             color, thickness, cv2.LINE_AA)


def draw_face_ui(frame, pts, name, born, now):
    age = now - born
    h, w = frame.shape[:2]
    xmin, xmax = float(pts[:, 0].min()), float(pts[:, 0].max())
    ymin, ymax = float(pts[:, 1].min()), float(pts[:, 1].max())

    known = name != "Unknown"
    base = KNOWN if known else UNKNOWN
    compliment = name.strip().lower() == COMPLIMENT_NAME

    # 1) Face outline
    frac = min(1.0, age / T_TRACE)
    oval_color = SCAN if age < T_ANALYZE_END else base
    poly(frame, pts, OVAL, oval_color, 2, closed=True, frac=frac)

    # 2) Scan line
    if age < T_ANALYZE_END:
        ph = (age / 1.4) % 1.0
        tri = 1 - abs(2 * ph - 1)
        y = ymin + (ymax - ymin) * tri
        line(frame, (xmin - 10, y), (xmax + 10, y), SCAN, 2)

    # 3) Analysis layer fade
    if age < T_TRACE:
        a = 0.0
    elif age < T_ANALYZE_END:
        a = min(1.0, (age - T_TRACE) / 0.3)
    elif age < T_ANALYZE_END + T_FADE:
        a = 1.0 - (age - T_ANALYZE_END) / T_FADE
    else:
        a = 0.0

    if a > 0.01:
        ov = frame.copy()
        for idx in (LEFT_EYE, RIGHT_EYE, LIPS):
            poly(ov, pts, idx, (255, 255, 255), 1, closed=True)
        for idx in (R_BROW, L_BROW, NOSE_BRIDGE, NOSE_BASE):
            poly(ov, pts, idx, (255, 255, 255), 1)

        le, re = center_of(pts, LEFT_EYE), center_of(pts, RIGHT_EYE)
        nose, mouth = pts[4], center_of(pts, [13, 14])
        line(ov, pts[234], pts[454], (200, 200, 200))
        line(ov, le, nose, (200, 200, 200))
        line(ov, re, nose, (200, 200, 200))
        line(ov, nose, mouth, (200, 200, 200))
        line(ov, pts[10], pts[152], (150, 150, 150))

        for x, yy in pts.astype(int):
            cv2.circle(ov, (int(x), int(yy)), 1, (255, 255, 255), -1)

        # Readout panel rows
        if compliment:
            rows = COMPLIMENT_ROWS
            value_color = (120, 255, 120)
            pw = 300
        else:
            eye_span = int(np.linalg.norm(pts[33] - pts[263]))
            mouth_w = int(np.linalg.norm(pts[61] - pts[291]))
            nose_len = int(np.linalg.norm(pts[168] - pts[2]))
            rows = [
                (0.9, "EYES", f"{eye_span} px"),
                (1.2, "BROWS", "OK"),
                (1.5, "NOSE", f"{nose_len} px"),
                (1.8, "MOUTH", f"{mouth_w} px"),
                (2.1, "JAW", "OK"),
                (2.4, "ID", name),
            ]
            value_color = (255, 255, 255)
            pw = 230

        row_h = 28
        ph_ = 14 + row_h * len(rows)
        px = int(xmax) + 16
        if px + pw > w - 5:
            px = max(5, int(xmin) - pw - 16)
        py = int(np.clip(ymin, 5, h - ph_ - 5))

        cv2.rectangle(ov, (px, py), (px + pw, py + ph_), (25, 25, 25), -1)
        cv2.rectangle(ov, (px, py), (px + pw, py + ph_), (230, 230, 230), 1)
        for i, (start, label, value) in enumerate(rows):
            if age < start:
                continue
            ry = py + 28 + i * row_h - 4
            cv2.putText(ov, label, (px + 12, ry), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (180, 180, 180), 1, cv2.LINE_AA)
            col = value_color
            if label == "ID":
                col = KNOWN if known else (110, 110, 255)
            cv2.putText(ov, value, (px + 95, ry), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62, col, 2, cv2.LINE_AA)

        # Progress bar under the chin
        prog = float(np.clip((age - T_TRACE) / (T_ANALYZE_END - T_TRACE), 0, 1))
        bx, by = int(xmin), int(min(h - 15, ymax + 14))
        bw = int(xmax - xmin)
        cv2.rectangle(ov, (bx, by), (bx + bw, by + 8), (25, 25, 25), -1)
        cv2.rectangle(ov, (bx, by), (bx + int(bw * prog), by + 8),
                      (255, 255, 255), -1)
        cv2.rectangle(ov, (bx, by), (bx + bw, by + 8), (230, 230, 230), 1)

        cv2.addWeighted(ov, a, frame, 1 - a, 0, frame)

    # 4) Floating name
    if age >= T_ANALYZE_END - 0.3:
        bob = 4 * np.sin(now * 2.5)
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cx = int(pts[10][0])
        ty = int(max(th + 10, ymin - 28 + bob))
        tx = int(np.clip(cx - tw / 2, 5, w - tw - 5))
        cv2.putText(frame, name, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(frame, name, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    base, 2, cv2.LINE_AA)


# ----------------------------------------------------------- registration
def ask_name_gui(cap, db):
    """Type the name inside the camera window. Returns None if cancelled."""
    text, msg = "", ""
    while True:
        ok, frame = cap.read()
        if not ok:
            return None
        H, W = frame.shape[:2]
        bw, bh = min(W - 40, 480), 50
        bx, by = (W - bw) // 2, H // 2 - bh // 2

        put_text_bg(frame, "Enter the name to register", (bx, by - 20),
                    0.8, (255, 255, 255), thickness=2)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (25, 25, 25), -1)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 255, 255), 2)
        cursor = "|" if int(time.time() * 2) % 2 == 0 else ""
        cv2.putText(frame, text + cursor, (bx + 12, by + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
                    cv2.LINE_AA)
        put_text_bg(frame, "Enter = OK     Esc = Cancel", (bx, by + bh + 40),
                    0.6)
        if msg:
            put_text_bg(frame, msg, (bx, by + bh + 80), 0.65,
                        (110, 110, 255), thickness=2)
        cv2.imshow("Register", frame)

        key = cv2.waitKey(30) & 0xFF
        if key == 27:                       # Esc
            cv2.destroyWindow("Register")
            return None
        elif key in (13, 10):               # Enter
            name = text.strip()
            if not name:
                msg = "Name cannot be empty"
            elif name in db:
                msg = f"{name} already exists"
            else:
                cv2.destroyWindow("Register")
                return name
        elif key in (8, 127):               # Backspace
            text, msg = text[:-1], ""
        elif 32 <= key <= 126 and len(text) < 24:
            text, msg = text + chr(key), ""


def register(cap, db, landmarker):
    name = ask_name_gui(cap, db)
    if name is None:
        print("Registration cancelled.")
        return
    ok, frame = cap.read()
    H, W = frame.shape[:2]
    detector, recognizer = load_models((W, H))

    cx0, cy0 = W // 2, H // 2
    radius = int(min(W, H) * 0.09)

    samples = []
    last = 0
    print(f"Registering {name}. Center your face on the crosshair and turn "
          "your head slightly between captures.")
    print("Press Q in the camera window to cancel.")
    while len(samples) < NUM_CAPTURES:
        ok, frame = cap.read()
        if not ok:
            break
        _, faces = detector.detect(frame)
        face = largest_face(faces)
        display = frame.copy()

        for pts in landmarks_for(landmarker, frame):
            poly(display, pts, OVAL, (255, 255, 255), 1, closed=True)

        # Alignment check: face center near the crosshair and a good size
        aligned = False
        msg = "No face detected"
        if face is not None:
            x, y, fw, fh = face[:4]
            dist = float(np.hypot(x + fw / 2 - cx0, y + fh / 2 - cy0))
            ratio = fw / W
            if ratio < 0.18:
                msg = "Move closer"
            elif ratio > 0.45:
                msg = "Move back a little"
            elif dist > radius:
                msg = "Center your face on the crosshair"
            else:
                aligned = True
                msg = "Hold still, capturing..."

        # Black crosshair with a halo (white normally, green when aligned)
        halo = (0, 220, 0) if aligned else (255, 255, 255)
        arm = radius + 35
        for col, th in ((halo, 6), ((0, 0, 0), 2)):
            cv2.circle(display, (cx0, cy0), radius, col, th, cv2.LINE_AA)
            cv2.line(display, (cx0 - arm, cy0), (cx0 + arm, cy0), col, th,
                     cv2.LINE_AA)
            cv2.line(display, (cx0, cy0 - arm), (cx0, cy0 + arm), col, th,
                     cv2.LINE_AA)

        if aligned and time.time() - last > CAPTURE_GAP:
            crop = recognizer.alignCrop(frame, face)
            samples.append(recognizer.feature(crop))
            last = time.time()

        put_text_bg(display, f"Captured {len(samples)}/{NUM_CAPTURES}",
                    (15, 35), 0.8, (255, 255, 255), thickness=2)
        put_text_bg(display, msg, (15, H - 20), 0.7,
                    (120, 255, 120) if aligned else (255, 255, 255),
                    thickness=2)
        cv2.imshow("Register", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyWindow("Register")

    if len(samples) == NUM_CAPTURES:
        db[name] = np.vstack(samples)
        save_db(db)
        print(f"Registered {name} with {NUM_CAPTURES} captures.")
    else:
        print("Registration cancelled.")


# ------------------------------------------------------------ recognition
def recognize(cap, db, landmarker):
    ok, frame = cap.read()
    detector, recognizer = load_models((frame.shape[1], frame.shape[0]))

    state = {"click": None}
    mode = "main"  # "main" or "remove"
    tracks = {}
    next_id = 0
    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse, state)
    print("Use the buttons in the window. Keys: Q = quit, A = add, D = remove.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()

        click = state["click"]
        state["click"] = None

        names = list(db.keys())
        name_rects = [(10, 60 + i * 45, 250, 38) for i in range(len(names))]
        cancel_rect = (10, 60 + len(names) * 45, 250, 38)

        do_add = False
        if mode == "main":
            if inside(click, BTN_ADD):
                do_add = True
            elif inside(click, BTN_REMOVE):
                mode = "remove"
        else:
            if inside(click, cancel_rect):
                mode = "main"
            else:
                for n, r in zip(names, name_rects):
                    if inside(click, r):
                        del db[n]
                        save_db(db)
                        print(f"Removed {n}.")
                        mode = "main"
                        break

        if do_add:
            register(cap, db, landmarker)
            continue

        # Detect with YuNet (for identity) and MediaPipe (for the outline)
        _, faces = detector.detect(frame)
        lm_faces = landmarks_for(landmarker, frame)

        used = set()
        for pts in lm_faces:
            center = pts.mean(axis=0)
            width = float(pts[:, 0].max() - pts[:, 0].min())

            tid, bd = None, max(80.0, width * 0.8)
            for k, t in tracks.items():
                if k in used:
                    continue
                d = float(np.hypot(*(t["center"] - center)))
                if d < bd:
                    bd, tid = d, k
            if tid is None:
                tid = next_id
                next_id += 1
                tracks[tid] = {"born": now, "name": "Unknown"}
            used.add(tid)
            t = tracks[tid]
            t["center"] = center
            t["seen"] = now

            yface = match_yunet(faces, center)
            if yface is not None:
                t["name"] = identify(recognizer, db, frame, yface)

            draw_face_ui(frame, pts, t["name"], t["born"], now)

        # Forget faces that left, so the scan animation replays on return
        for k in [k for k, t in tracks.items() if now - t["seen"] > 1.0]:
            del tracks[k]

        # Buttons / remove list
        if mode == "main":
            draw_button(frame, BTN_ADD, "Add Face", (0, 120, 0))
            draw_button(frame, BTN_REMOVE, "Remove Face", (0, 0, 150))
        else:
            cv2.putText(frame, "Click a name to remove:", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            for n, r in zip(names, name_rects):
                draw_button(frame, r, n, (0, 0, 150))
            draw_button(frame, cancel_rect, "Cancel", (90, 90, 90))

        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("a"):
            register(cap, db, landmarker)
        if key == ord("d"):
            mode = "remove"

    cv2.destroyAllWindows()


def main():
    cap = open_camera()
    landmarker = create_landmarker()
    db = load_db()
    if not db:
        print("No faces registered yet. Let's register the first person.")
        register(cap, db, landmarker)
        if not db:
            cap.release()
            return
    recognize(cap, db, landmarker)
    cap.release()


if __name__ == "__main__":
    main()