# 🎭 Face Detection & Recognition System

A complete Python-based face detection and recognition system using **OpenCV** and the **face_recognition** library. Supports real-time webcam feeds, static images, and video files. Easily extensible to add new people without retraining from scratch.

---

## 📁 Project Structure

```
face-recognition-system/
│
├── face_recognition_main.py    # Main script — detection + recognition
├── encode_faces.py             # Build encoding database from known_faces/
├── add_person.py               # Register new people (webcam or image)
├── detect_faces_opencv.py      # Lightweight OpenCV-only detector (no dlib)
│
├── known_faces/                # Add sub-folders per person here
│   ├── Alice/
│   │   ├── alice_01.jpg
│   │   └── alice_02.jpg
│   └── Bob/
│       └── bob_01.jpg
│
├── unknown_faces/              # Auto-saved crops of unrecognised faces
├── models/                     # encodings.pkl saved here after encoding step
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/face-recognition-system.git
cd face-recognition-system
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note (Windows / macOS):** `face-recognition` requires `dlib`. If installation fails, install CMake first:
> ```bash
> pip install cmake
> pip install dlib
> pip install face-recognition
> ```

---

## 🚀 Quick Start

### Step 1 — Add known faces

**Option A — Copy photos manually:**
```
known_faces/
    Alice/
        photo1.jpg
        photo2.jpg
    Bob/
        photo1.jpg
```

**Option B — Capture from webcam:**
```bash
python add_person.py capture --name "Alice" --count 10
```

**Option C — Add a single image:**
```bash
python add_person.py add --name "Bob" --image /path/to/bob.jpg
```

---

### Step 2 — Encode faces
```bash
python encode_faces.py
```
This scans `known_faces/` and saves `models/encodings.pkl`.

---

### Step 3 — Run the recognition system

**Webcam (real-time):**
```bash
python face_recognition_main.py --mode webcam
```

**Static image:**
```bash
python face_recognition_main.py --mode image --input photo.jpg --output result.jpg
```

**Video file:**
```bash
python face_recognition_main.py --mode video --input clip.mp4 --output result.mp4
```

---

## 🔍 OpenCV-Only Detection (no face_recognition required)

For a lighter installation using only OpenCV (no recognition, detection only):

**Haar cascades:**
```bash
python detect_faces_opencv.py --mode webcam --detector haar
```

**DNN (more accurate — requires model files):**
```bash
python detect_faces_opencv.py --mode webcam --detector dnn
```

> Download DNN model files:
> - [`deploy.prototxt`](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt)
> - [`res10_300x300_ssd_iter_140000.caffemodel`](https://github.com/opencv/opencv_3rdparty/tree/dnn_samples_face_detector_20170830)
>
> Place both in the `models/` folder.

---

## 🎮 Controls

| Key | Action |
|-----|--------|
| `Q` | Quit the application |
| `S` | Save a snapshot (webcam mode) |
| `SPACE` | Start auto-capture (add_person.py) |

---

## ⚡ Configuration

Edit constants at the top of `face_recognition_main.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `TOLERANCE` | `0.5` | Match threshold — lower = stricter |
| `MODEL` | `"hog"` | Detection model: `hog` (CPU) / `cnn` (GPU) |
| `SCALE_FACTOR` | `0.5` | Frame resize ratio for speed |

---

## 📦 Dependencies

| Library | Purpose |
|---------|---------|
| `opencv-python` | Frame capture, drawing, video I/O |
| `face-recognition` | Face encoding & matching (built on dlib) |
| `numpy` | Array operations |

---

## 🧠 How It Works

1. **Encoding phase** — Each known face is passed through a deep ResNet model (via `face_recognition`) that produces a 128-dimensional embedding vector. These are stored in `models/encodings.pkl`.

2. **Detection phase** — Incoming frames are resized and scanned for face bounding boxes using HOG (fast, CPU) or CNN (accurate, GPU).

3. **Recognition phase** — Each detected face is encoded and compared against all stored embeddings using Euclidean distance. The closest match below `TOLERANCE` wins.

4. **Unknown faces** — Faces with no match below the tolerance are labelled "Unknown" and their cropped images are auto-saved to `unknown_faces/` for later review.

---

## 🤝 Contributing

Pull requests are welcome! To add a feature:

1. Fork the repo
2. Create a branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "Add my feature"`
4. Push: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
