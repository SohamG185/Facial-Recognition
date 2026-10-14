# Facial Recognition

![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-5.0-5C3EE8)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Landmarker-orange)
![Status](https://img.shields.io/badge/status-active-brightgreen)

**A real-time face registration and recognition system built entirely from open source software.**

Open Source Technologies (OST) course project. The application uses a laptop webcam to detect faces, recognize registered people, and draw an animated MediaPipe landmark outline with the person's name floating above their head. No proprietary SDKs, cloud APIs or paid services are used. Everything runs locally and offline.

---

## Table of contents

1. [Project overview](#project-overview)
2. [Why open source](#why-open-source)
3. [Open source stack and licenses](#open-source-stack-and-licenses)
4. [Features](#features)
5. [How it works](#how-it-works)
6. [Getting started](#getting-started)
7. [Usage](#usage)
8. [Configuration](#configuration)
9. [Project structure](#project-structure)
10. [Contributing](#contributing)
11. [Roadmap](#roadmap)
12. [Privacy and ethics](#privacy-and-ethics)
13. [Troubleshooting](#troubleshooting)
14. [License](#license)
15. [Acknowledgements](#acknowledgements)

---

## Project overview

Most facial recognition tutorials depend on `dlib` (which needs a C++ compiler and CMake) or on paid cloud APIs. This project shows that a complete, good-looking system can be built from freely available, permissively licensed components that install with a single `pip` command.

**What it does**

- If no faces are registered, it asks for a name and captures face samples, guided by an on-screen crosshair.
- It then watches the camera, recognizes known people, and labels unknown ones.
- Each face is shown with a MediaPipe landmark outline, a short scan and analysis animation, and then just the outline and a floating name.
- People can be added or removed at any time with buttons inside the camera window.

## Why open source

This project is both *made with* and *released as* open source:

- **Built on open components.** Every library and model used is open source and can be inspected, modified and redistributed.
- **Released under the MIT License.** Anyone can use, study, change and share the code.
- **Developed in the open.** The source lives on GitHub, uses Git for version control, and welcomes issues and pull requests.
- **Reproducible.** A `requirements.txt`, documented model downloads and step-by-step setup let anyone rebuild the project on their own machine.
- **Local and transparent.** No data leaves the machine, and no black-box service is involved in recognition.

## Open source stack and licenses

| Component | Role in the project | License |
|-----------|--------------------|---------|
| [Python](https://www.python.org/) | Programming language | PSF License |
| [OpenCV](https://opencv.org/) | Camera capture, image processing, drawing, DNN inference | Apache 2.0 |
| [OpenCV Zoo: YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | Face detection model | MIT |
| [OpenCV Zoo: SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) | Face recognition (embedding) model | Apache 2.0 |
| [MediaPipe](https://ai.google.dev/edge/mediapipe) | Face Landmarker for the outline and facial features | Apache 2.0 |
| [NumPy](https://numpy.org/) | Numerical arrays and vector math | BSD 3-Clause |
| [Git](https://git-scm.com/) | Version control | GPL v2 |
| [GitHub](https://github.com/) | Public hosting, issues, pull requests | Hosting platform (the project code is MIT) |

> Licenses are listed as published by each project. Check each repository for the current terms before redistributing the model files.

## Features

- **First-run registration.** If no faces are saved, the app asks for a name and takes 10 samples.
- **Crosshair-guided capture.** A black crosshair shows where to center your face. Samples are taken only when you are aligned and at a good distance, with live hints ("Move closer", "Center your face on the crosshair").
- **Live recognition.** Known faces get a green outline and name, unknown faces get a red outline and the label "Unknown".
- **Landmark outline.** MediaPipe's 478-point face mesh replaces the plain bounding box.
- **Scan animation.** The outline traces itself, a scan line sweeps the face, and eye, brow, nose and mouth outlines with support lines appear alongside a readability-focused readout panel, then fade to leave only the outline and name.
- **Custom message panel.** A specific registered name can show a custom set of messages instead of measurements (configurable).
- **In-window buttons.** Add Face and Remove Face buttons inside the camera window, plus keyboard shortcuts.
- **Multi-face tracking.** Up to 4 faces at once, each with its own animation.
- **No compiler needed.** Pure `pip` install on Windows, macOS and Linux.

## How it works

```
 Webcam frame
      |
      +--> YuNet (OpenCV DNN) ------> face box + 5 key points
      |                                      |
      |                                      v
      |                           SFace alignCrop + feature
      |                                      |
      |                                      v
      |                     cosine similarity vs. saved embeddings
      |                                      |
      +--> MediaPipe Face Landmarker         v
           (478 landmarks) --------> name  (or "Unknown")
                    |                        |
                    +-----------+------------+
                                v
              Animated outline + floating name drawn with OpenCV
```

**Recognition pipeline**

1. **Detect:** YuNet finds faces and their key points.
2. **Embed:** SFace aligns each face and converts it to a 128-dimensional vector (an embedding).
3. **Match:** the live embedding is compared to every saved embedding using cosine similarity. The best score above `MATCH_THRESHOLD` gives the name.
4. **Draw:** MediaPipe landmarks drive the outline and animation, and each landmark face is paired with its YuNet detection to attach the identity.

**Storage:** registered people are saved locally in `faces.pkl` as a dictionary of `name -> array of embeddings`. No photos are stored.

## Getting started

### Requirements

- A computer with a webcam (Windows, macOS or Linux)
- Python 3.10 or newer (developed and tested on Python 3.14)
- Git

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

The model files are large, so they are not stored in the repository. Download them into a `models` folder.

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

Expected sizes: about 230 KB, 38 MB and 3.7 MB. A file of only a few hundred bytes means the download failed.

## Usage

```
python face_app.py
```

1. **First run:** the terminal asks for a name. Type it and press Enter, then center your face on the black crosshair and slowly turn your head slightly between captures. Ten samples are taken automatically.
2. **Recognition:** the live window opens and recognizes registered faces.
3. **Manage people** with the on-screen buttons or the keys below.

| Action | Mouse | Keyboard |
|--------|-------|----------|
| Add a new face | Click **Add Face** | `A` |
| Remove a face | Click **Remove Face**, then click a name | `D` |
| Cancel a registration | n/a | `Q` in the Register window |
| Quit | n/a | `Q` |

When adding a face, the camera window pauses and the terminal asks for the name. Click the terminal, type the name and press Enter.

## Configuration

Settings are constants at the top of `face_app.py`:

| Setting | Default | Meaning |
|---------|---------|---------|
| `NUM_CAPTURES` | 10 | Samples taken per person |
| `CAPTURE_GAP` | 0.6 | Seconds between samples |
| `MATCH_THRESHOLD` | 0.363 | Cosine similarity needed for a match. Higher is stricter |
| `T_TRACE`, `T_ANALYZE_END`, `T_FADE` | 0.8, 2.8, 0.6 | Animation timing in seconds |
| `COMPLIMENT_NAME`, `COMPLIMENT_ROWS` | `ravi` | Name that gets a custom message panel, and its text. Edit or remove freely |

## Project structure

```
Facial-Recognition/
├── face_app.py          # the application (registration, recognition, UI)
├── requirements.txt     # Python dependencies
├── README.md            # this file
├── LICENSE              # MIT License
├── .gitignore           # keeps venv, models and face data out of Git
├── models/              # downloaded model files (not tracked)
└── faces.pkl            # saved face embeddings, created on first run (not tracked)
```

## Contributing

Contributions are welcome. This is how to take part:

1. **Fork** the repository on GitHub.
2. **Create a branch** for your change:
   ```
   git switch -c feature/short-description
   ```
3. **Make your changes** and test them with a real camera.
4. **Commit** with a clear message:
   ```
   git commit -m "Add: short description of the change"
   ```
5. **Push** your branch and open a **pull request** describing what changed and why.

You can also open an **issue** to report a bug, suggest a feature, or ask a question. Please include your operating system, Python version and the full error text when reporting a bug.

**Guidelines**

- Keep code readable and commented where the logic is not obvious.
- Do not commit `venv/`, `models/` or `faces.pkl`.
- Do not add dependencies with licenses incompatible with MIT.
- Never commit anyone's face data.

## Roadmap

Ideas for future contributors:

- [ ] Command-line options (camera index, threshold, database path)
- [ ] Script to download models automatically (`download_models.py`)
- [ ] Liveness check to resist photo spoofing
- [ ] Export and import of the face database
- [ ] Unit tests for the matching logic
- [ ] Configurable themes and animation styles
- [ ] Packaging as an installable module

## Privacy and ethics

Face data is biometric data and deserves care.

- Everything runs **locally**. Nothing is uploaded.
- Only face **embeddings** are stored, not photos, in `faces.pkl`, which is excluded from Git.
- Only register people who have **given their consent**.
- Remove a person with the **Remove Face** button, or delete `faces.pkl` to erase everyone.
- This is an educational project and is **not** suitable for security, access control or any decision that affects people.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `FileNotFoundError: models/...` | A model is missing. Repeat the download step |
| "Could not open the camera" | Close other apps using the webcam (Zoom, Teams, browser tabs) |
| PowerShell blocks the activate script | Run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`, then activate again |
| People misidentified or shown as Unknown | Adjust `MATCH_THRESHOLD` and re-register in even lighting |
| `pip install mediapipe` fails | Use Python 3.11 or 3.12 in a fresh virtual environment |

## License

This project is released under the **MIT License**. See the [LICENSE](LICENSE) file for the full text.

Third-party libraries and models keep their own licenses, listed in the [stack table](#open-source-stack-and-licenses).

## Acknowledgements

- [OpenCV](https://opencv.org/) and the [OpenCV Zoo](https://github.com/opencv/opencv_zoo) for the detection and recognition models
- [MediaPipe](https://ai.google.dev/edge/mediapipe) for the Face Landmarker
- [NumPy](https://numpy.org/) and the [Python](https://www.python.org/) community
- Created as a project for the **Open Source Technologies** course
