import os
import sys
import time
import pickle
import cv2
import numpy as np

DETECTOR_PATH = "models/face_detection_yunet_2023mar.onnx"
RECOGNIZER_PATH = "models/face_recognition_sface_2021dec.onnx"
DB_PATH = "faces.pkl"

NUM_CAPTURES = 10          # samples taken per person
CAPTURE_GAP = 0.6          # seconds between samples
MATCH_THRESHOLD = 0.363    # cosine similarity; higher = stricter


def load_models(frame_size):
    detector = cv2.FaceDetectorYN.create(
        DETECTOR_PATH, "", frame_size, score_threshold=0.8
    )
    recognizer = cv2.FaceRecognizerSF.create(RECOGNIZER_PATH, "")
    return detector, recognizer


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


def largest_face(faces):
    if faces is None or len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def register(cap, db):
    name = input("Enter the name to register: ").strip()
    if not name:
        print("Name cannot be empty.")
        return
    ok, frame = cap.read()
    detector, recognizer = load_models((frame.shape[1], frame.shape[0]))

    samples = []
    last = 0
    print("Look at the camera. Slowly turn your head a little between captures.")
    print("Press Q to cancel.")
    while len(samples) < NUM_CAPTURES:
        ok, frame = cap.read()
        if not ok:
            break
        _, faces = detector.detect(frame)
        face = largest_face(faces)
        display = frame.copy()
        if face is not None:
            x, y, w, h = face[:4].astype(int)
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 255), 2)
            if time.time() - last > CAPTURE_GAP:
                aligned = recognizer.alignCrop(frame, face)
                samples.append(recognizer.feature(aligned))
                last = time.time()
        cv2.putText(display, f"Captured {len(samples)}/{NUM_CAPTURES}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
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


def recognize(cap, db):
    ok, frame = cap.read()
    detector, recognizer = load_models((frame.shape[1], frame.shape[0]))
    print("Recognizing. Press Q to quit, R to register another person.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        _, faces = detector.detect(frame)
        if faces is not None:
            for face in faces:
                x, y, w, h = face[:4].astype(int)
                aligned = recognizer.alignCrop(frame, face)
                feat = recognizer.feature(aligned)

                best_name, best_score = "Unknown", 0.0
                for name, feats in db.items():
                    for known in feats:
                        score = recognizer.match(
                            feat, known.reshape(1, -1),
                            cv2.FaceRecognizerSF_FR_COSINE
                        )
                        if score > best_score:
                            best_score, best_name = score, name
                if best_score < MATCH_THRESHOLD:
                    best_name = "Unknown"

                color = (0, 255, 0) if best_name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                label = f"{best_name} ({best_score:.2f})"
                cv2.putText(frame, label, (x, max(y - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("Face Recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            cv2.destroyWindow("Face Recognition")
            register(cap, db)
    cv2.destroyAllWindows()


def main():
    cap = open_camera()
    db = load_db()
    if not db:
        print("No faces registered yet. Let's register the first person.")
        register(cap, db)
        if not db:
            cap.release()
            return
    recognize(cap, db)
    cap.release()


if __name__ == "__main__":
    main()