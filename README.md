# Facial Recognition

A real-time face recognition app for your laptop camera, built with Python, OpenCV and MediaPipe.

Instead of a plain bounding box, each face gets a MediaPipe landmark outline. When a face first appears, the outline traces itself, a scan line sweeps over it, and an analysis overlay (eyes, brows, nose, mouth, support lines and a readout panel) plays for about three seconds. Then the details fade away, leaving only the face outline and the person's name floating above their head.

## Features

- **First-run registration:** if no faces are saved, the app asks for a name and captures 10 samples of your face.
- **Crosshair-guided capture:** a black crosshair in the Register window tells you where to center your face. Samples are only taken when you are aligned and at a good distance, and on-screen hints tell you what to fix.
- **Live recognition:** known faces get a green outline and name, unknown faces get a red outline and "Unknown".
- **Scan animation:** outline trace, scan line, landmark analysis and readout panel, then fade to outline + name.
- **In-window buttons:** **Add Face** and **Remove Face** buttons inside the camera window.
- **Multiple faces:** tracks up to 4 faces at once, each with its own animation.
- **No compiling needed:** uses ONNX models via OpenCV, so there is no dlib or CMake.

## How it works

| Task | Tool |
|------|------|
| Face detection | OpenCV YuNet (`face_detection_yunet_2023mar.onnx`) |
| Face recognition (embeddings) | OpenCV SFace (`face_recognition_sface_2021dec.onnx`) |
| Face outline and landmarks | MediaPipe Face Landmarker (`face_landmarker.task`) |
| Camera and drawing | OpenCV |

Each registered person is stored as 10 face embeddings. A live face is compared to every stored embedding using cosine similarity, and the best match above the threshold wins.

## Requirements

- Windows, macOS or Linux with a webcam
- Python 3.10 or newer (developed on Python 3.14)
- Packages: `opencv-python`, `numpy`, `mediapipe`

## Setup

### 1. Clone the repository

```
git clone https://github.com/SohamG185/Facial-Recognition.git
cd Facial-Recognition
```

### 2. Create and activate a virtual environment

Windows (PowerShell):

```
python -m venv venv
.\venv\Scripts\Activate.ps1
```

macOS / Linux:

```
python3 -m venv venv
source venv/bin/activate
```

### 3. Install the dependencies

```
pip install -r requirements.txt
```

### 4. Download the models

The model files are not stored in the repository (they are in `.gitignore`), so download them into a `models` folder.

Windows (PowerShell):

```
mkdir models
Invoke-WebRequest -Uri "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" -OutFile "models\face_detection_yunet_2023mar.onnx"
Invoke-WebRequest -Uri "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx" -OutFile "models\face_recognition_sface_2021dec.onnx"
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" -OutFile "models\face_landmarker.task"
```

macOS / Linux:

```
mkdir -p models
curl -L -o models/face_detection_yunet_2023mar.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
curl -L -o models/face_recognition_sface_2021dec.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
curl -L -o models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

Expected sizes: about 230 KB, 38 MB and 3.7 MB. If a file is only a few hundred bytes, the download failed.

## Usage

```
python face_app.py
```

1. **First run:** the terminal asks for a name. Type it, press Enter, then center your face on the black crosshair. Slowly turn your head a little between captures. The app takes 10 samples automatically.
2. **Recognition:** the live camera window opens and recognizes registered faces.
3. **Add or remove people** with the buttons in the window, or the keyboard shortcuts below.

### Controls

| Action | Mouse | Keyboard |
|--------|-------|----------|
| Add a new face | Click **Add Face** | `A` |
| Remove a face | Click **Remove Face**, then click a name | `D` |
| Cancel registration | n/a | `Q` in the Register window |
| Quit | n/a | `Q` |

When adding a face, the camera window pauses and the terminal asks for the name. Click the terminal, type the name and press Enter.

## Configuration

Settings are at the top of `face_app.py`:

| Setting | Default | Meaning |
|---------|---------|---------|
| `NUM_CAPTURES` | 10 | Samples taken per person |
| `CAPTURE_GAP` | 0.6 | Seconds between samples |
| `MATCH_THRESHOLD` | 0.363 | Cosine similarity needed for a match. Raise it to be stricter, lower it to be more lenient |
| `T_TRACE`, `T_ANALYZE_END`, `T_FADE` | 0.8, 2.8, 0.6 | Animation timing in seconds |
| `COMPLIMENT_NAME` / `COMPLIMENT_ROWS` | `ravi` | A name that shows a custom message panel instead of measurements. Edit or remove it freely |

## Project structure

```
Facial-Recognition/
├── face_app.py          # the whole application
├── requirements.txt     # Python dependencies
├── README.md
├── .gitignore
├── models/              # downloaded models (not in git)
└── faces.pkl            # saved face data, created on first run (not in git)
```

## Privacy

`faces.pkl` contains biometric data (face embeddings) and is excluded from Git on purpose. Do not commit it or share it. Only register people who have agreed to it. To delete someone, use the **Remove Face** button, or delete `faces.pkl` to remove everyone.

## Troubleshooting

- **`FileNotFoundError: models/...`:** a model is missing. Repeat the download step.
- **"Could not open the camera":** close other apps using the webcam (Zoom, Teams, browser tabs) and try again.
- **PowerShell blocks the activate script:** run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`, then activate again.
- **Wrong person recognized, or known people shown as Unknown:** adjust `MATCH_THRESHOLD`, and re-register in good, even lighting.
- **`pip install mediapipe` fails:** MediaPipe can lag behind brand-new Python versions. Use a slightly older Python (3.11 or 3.12) in a fresh virtual environment.

## Acknowledgements

- [OpenCV Zoo](https://github.com/opencv/opencv_zoo) for the YuNet and SFace models
- [MediaPipe](https://ai.google.dev/edge/mediapipe) for the Face Landmarker
