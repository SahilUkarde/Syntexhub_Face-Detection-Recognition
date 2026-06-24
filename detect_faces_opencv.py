"""
detect_faces_opencv.py
──────────────────────
Lightweight face *detection* using OpenCV's built-in Haar cascades or DNN.
No face_recognition library required — useful as a quick standalone demo
or on systems where dlib cannot be installed.

Usage:
  python detect_faces_opencv.py --mode webcam
  python detect_faces_opencv.py --mode image  --input photo.jpg --output out.jpg
  python detect_faces_opencv.py --mode video  --input clip.mp4  --output result.mp4
  python detect_faces_opencv.py --mode webcam --detector dnn
"""

import cv2
import argparse
import os

FONT = cv2.FONT_HERSHEY_SIMPLEX

# DNN model paths (download separately — see README)
DNN_PROTO  = "models/deploy.prototxt"
DNN_MODEL  = "models/res10_300x300_ssd_iter_140000.caffemodel"
DNN_CONF   = 0.5   # minimum confidence threshold


# ──────────────────────────────────────────────
# Detectors
# ──────────────────────────────────────────────

def get_haar_cascade():
    xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(xml)
    if cascade.empty():
        raise RuntimeError("Haar cascade XML not found — check OpenCV installation.")
    return cascade


def get_dnn_net():
    if not os.path.exists(DNN_PROTO) or not os.path.exists(DNN_MODEL):
        raise FileNotFoundError(
            "DNN model files not found.\n"
            f"  Expected:\n    {DNN_PROTO}\n    {DNN_MODEL}\n"
            "  Download from: https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector"
        )
    return cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL)


# ──────────────────────────────────────────────
# Detection functions
# ──────────────────────────────────────────────

def detect_haar(frame, cascade, scale=1.1, min_neighbors=5):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray  = cv2.equalizeHist(gray)
    faces = cascade.detectMultiScale(gray, scaleFactor=scale,
                                     minNeighbors=min_neighbors,
                                     minSize=(30, 30))
    boxes = []
    for (x, y, w, h) in faces:
        boxes.append((y, x + w, y + h, x))   # top, right, bottom, left
    return boxes


def detect_dnn(frame, net):
    h, w = frame.shape[:2]
    blob  = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                  1.0, (300, 300),
                                  (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > DNN_CONF:
            box   = detections[0, 0, i, 3:7] * [w, h, w, h]
            x1, y1, x2, y2 = box.astype(int)
            boxes.append((y1, x2, y2, x1))   # top, right, bottom, left
    return boxes


def draw_boxes(frame, boxes, detector_name):
    for (top, right, bottom, left) in boxes:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, "Face", (left, top - 8), FONT, 0.6, (0, 255, 0), 2)
    count_text = f"{detector_name} | Faces: {len(boxes)}"
    cv2.putText(frame, count_text, (10, 30), FONT, 0.7, (255, 255, 0), 2)
    return frame


# ──────────────────────────────────────────────
# Run modes
# ──────────────────────────────────────────────

def run(mode, detector, input_path=None, output_path=None):
    if detector == "haar":
        cascade = get_haar_cascade()
        detect  = lambda f: detect_haar(f, cascade)
        label   = "Haar"
    else:
        net    = get_dnn_net()
        detect = lambda f: detect_dnn(f, net)
        label  = "DNN"

    # ── Webcam ──────────────────────────────
    if mode == "webcam":
        cap = cv2.VideoCapture(0)
        print("[INFO] Webcam started. Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            boxes = detect(frame)
            frame = draw_boxes(frame, boxes, label)
            cv2.imshow("Face Detection — press q", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()

    # ── Image ───────────────────────────────
    elif mode == "image":
        frame = cv2.imread(input_path)
        if frame is None:
            print(f"[ERROR] Cannot read: {input_path}")
            return
        boxes = detect(frame)
        frame = draw_boxes(frame, boxes, label)
        print(f"[INFO] Detected {len(boxes)} face(s).")
        if output_path:
            cv2.imwrite(output_path, frame)
            print(f"[INFO] Saved to {output_path}")
        else:
            cv2.imshow("Result — any key to close", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # ── Video ────────────────────────────────
    elif mode == "video":
        cap    = cv2.VideoCapture(input_path)
        fps    = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        fc = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            boxes = detect(frame)
            frame = draw_boxes(frame, boxes, label)
            if writer:
                writer.write(frame)
            cv2.imshow("Video — q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            fc += 1

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print(f"[INFO] Processed {fc} frame(s).")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="OpenCV-only face detection (Haar / DNN).")
    p.add_argument("--mode",     choices=["webcam", "image", "video"], default="webcam")
    p.add_argument("--detector", choices=["haar", "dnn"],              default="haar")
    p.add_argument("--input",    type=str, help="Input image or video path.")
    p.add_argument("--output",   type=str, help="Output path to save annotated result.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.mode, args.detector, args.input, args.output)
