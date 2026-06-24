"""
Face Detection & Recognition System
Uses OpenCV + face_recognition library for real-time face detection and recognition.
"""

import cv2
import face_recognition
import numpy as np
import os
import pickle
import argparse
from datetime import datetime


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
KNOWN_FACES_DIR = "known_faces"
ENCODINGS_FILE  = "models/encodings.pkl"
UNKNOWN_DIR     = "unknown_faces"
TOLERANCE       = 0.5          # Lower = stricter match
MODEL           = "hog"        # "hog" (CPU) or "cnn" (GPU, more accurate)
SCALE_FACTOR    = 0.5          # Resize frame for faster processing
BOX_COLOR       = (0, 255, 0)  # Green bounding box for known faces
UNKNOWN_COLOR   = (0, 0, 255)  # Red bounding box for unknown faces
FONT            = cv2.FONT_HERSHEY_SIMPLEX


# ──────────────────────────────────────────────
# Encoding helpers
# ──────────────────────────────────────────────

def load_encodings(encodings_file: str = ENCODINGS_FILE):
    """Load pre-computed face encodings from disk."""
    if os.path.exists(encodings_file):
        with open(encodings_file, "rb") as f:
            data = pickle.load(f)
        print(f"[INFO] Loaded {len(data['names'])} encoding(s) from {encodings_file}")
        return data["encodings"], data["names"]
    print("[WARN] No encodings file found. Run encode_faces.py first.")
    return [], []


def save_encodings(encodings, names, encodings_file: str = ENCODINGS_FILE):
    """Save face encodings to disk."""
    os.makedirs(os.path.dirname(encodings_file), exist_ok=True)
    with open(encodings_file, "wb") as f:
        pickle.dump({"encodings": encodings, "names": names}, f)
    print(f"[INFO] Saved {len(names)} encoding(s) to {encodings_file}")


# ──────────────────────────────────────────────
# Core recognition logic
# ──────────────────────────────────────────────

def recognize_faces_in_frame(frame, known_encodings, known_names):
    """
    Detect and recognise faces in a single BGR frame.
    Returns a list of (top, right, bottom, left, name) tuples.
    """
    # Resize for speed, convert BGR→RGB
    small = cv2.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
    rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    locations = face_recognition.face_locations(rgb, model=MODEL)
    encodings = face_recognition.face_encodings(rgb, locations)

    results = []
    for encoding, location in zip(encodings, locations):
        name = "Unknown"

        if known_encodings:
            matches    = face_recognition.compare_faces(known_encodings, encoding, tolerance=TOLERANCE)
            distances  = face_recognition.face_distance(known_encodings, encoding)
            best_idx   = np.argmin(distances)
            if matches[best_idx]:
                name = known_names[best_idx]

        # Scale locations back to original frame size
        top, right, bottom, left = [int(v / SCALE_FACTOR) for v in location]
        results.append((top, right, bottom, left, name))

    return results


def draw_results(frame, results):
    """Draw bounding boxes and name labels on the frame."""
    for (top, right, bottom, left, name) in results:
        color = BOX_COLOR if name != "Unknown" else UNKNOWN_COLOR
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Label background
        label_bg_top = bottom if bottom + 30 < frame.shape[0] else top - 30
        cv2.rectangle(frame, (left, label_bg_top), (right, label_bg_top + 25), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 5, label_bg_top + 18), FONT, 0.6, (255, 255, 255), 2)

    return frame


# ──────────────────────────────────────────────
# Run modes
# ──────────────────────────────────────────────

def run_webcam(known_encodings, known_names):
    """Real-time face recognition from webcam."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    print("[INFO] Starting webcam. Press 'q' to quit, 's' to save a snapshot.")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process every 2nd frame to reduce CPU load
        if frame_count % 2 == 0:
            results = recognize_faces_in_frame(frame, known_encodings, known_names)
            # Save unknown faces automatically
            for (top, right, bottom, left, name) in results:
                if name == "Unknown":
                    save_unknown_face(frame, top, right, bottom, left)

        frame = draw_results(frame, results if frame_count % 2 == 0 else [])

        # FPS overlay
        cv2.putText(frame, f"Frame: {frame_count}", (10, 25), FONT, 0.6, (200, 200, 200), 1)
        cv2.imshow("Face Recognition — press q to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"snapshot_{ts}.jpg"
            cv2.imwrite(path, frame)
            print(f"[INFO] Snapshot saved: {path}")

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


def run_image(image_path, known_encodings, known_names, output_path=None):
    """Run face recognition on a static image."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return

    results = recognize_faces_in_frame(frame, known_encodings, known_names)
    frame   = draw_results(frame, results)

    print(f"[INFO] Detected {len(results)} face(s):")
    for (_, _, _, _, name) in results:
        print(f"       → {name}")

    if output_path:
        cv2.imwrite(output_path, frame)
        print(f"[INFO] Result saved to {output_path}")
    else:
        cv2.imshow("Result — press any key to close", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_video(video_path, known_encodings, known_names, output_path=None):
    """Run face recognition on a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    results     = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 2 == 0:
            results = recognize_faces_in_frame(frame, known_encodings, known_names)

        frame = draw_results(frame, results)

        if writer:
            writer.write(frame)

        cv2.imshow("Video — press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_count += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Processed {frame_count} frames.")


# ──────────────────────────────────────────────
# Unknown face saver
# ──────────────────────────────────────────────

def save_unknown_face(frame, top, right, bottom, left):
    """Crop and save an unknown face to the unknown_faces folder."""
    os.makedirs(UNKNOWN_DIR, exist_ok=True)
    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        return
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(UNKNOWN_DIR, f"unknown_{ts}.jpg")
    cv2.imwrite(path, crop)


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Face Detection & Recognition System")
    parser.add_argument("--mode",   choices=["webcam", "image", "video"], default="webcam",
                        help="Input mode (default: webcam)")
    parser.add_argument("--input",  type=str, help="Path to image or video file")
    parser.add_argument("--output", type=str, help="Path to save annotated output")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    known_encodings, known_names = load_encodings()

    if args.mode == "webcam":
        run_webcam(known_encodings, known_names)
    elif args.mode == "image":
        if not args.input:
            print("[ERROR] Provide --input path for image mode.")
        else:
            run_image(args.input, known_encodings, known_names, args.output)
    elif args.mode == "video":
        if not args.input:
            print("[ERROR] Provide --input path for video mode.")
        else:
            run_video(args.input, known_encodings, known_names, args.output)
