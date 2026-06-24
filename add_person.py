"""
add_person.py
─────────────
Interactive script to register a new person into the system.

Two sub-commands:
  capture   – open webcam, capture N photos, save to known_faces/<name>/
  add       – copy an existing image file into known_faces/<name>/

After registering, run encode_faces.py to update the encoding database.

Usage examples:
  python add_person.py capture --name "Alice" --count 10
  python add_person.py add     --name "Bob"   --image /path/to/bob.jpg
"""

import os
import cv2
import shutil
import argparse
from datetime import datetime

KNOWN_FACES_DIR = "known_faces"
FONT            = cv2.FONT_HERSHEY_SIMPLEX


# ──────────────────────────────────────────────
# Capture mode
# ──────────────────────────────────────────────

def capture_faces(name: str, count: int = 10, delay: int = 500) -> None:
    """
    Open the webcam, detect faces in real time, and capture `count`
    frames once the user presses SPACE.
    """
    person_dir = os.path.join(KNOWN_FACES_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot access webcam.")
        return

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    captured  = 0
    auto_mode = False
    print(f"[INFO] Capturing {count} photos for '{name}'.")
    print("       Press SPACE to start auto-capture | press 'q' to quit early.")

    while captured < count:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces   = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # HUD
        status = f"Captured: {captured}/{count}"
        cv2.putText(display, status, (10, 30), FONT, 0.8, (0, 220, 0), 2)
        cv2.putText(display, "SPACE: start | Q: quit", (10, 60), FONT, 0.6, (200, 200, 200), 1)
        cv2.imshow(f"Capturing: {name}", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            auto_mode = True

        if auto_mode and len(faces) > 0:
            ts       = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = os.path.join(person_dir, f"{name}_{ts}.jpg")
            cv2.imwrite(filename, frame)
            captured += 1
            print(f"  [{captured}/{count}] Saved {filename}")
            cv2.waitKey(delay)   # brief pause between captures

    cap.release()
    cv2.destroyAllWindows()

    if captured > 0:
        print(f"\n[SUCCESS] {captured} photo(s) saved to {person_dir}/")
        print("          Run  python encode_faces.py  to update the database.")
    else:
        print("[WARN] No photos captured.")


# ──────────────────────────────────────────────
# Add-image mode
# ──────────────────────────────────────────────

def add_image(name: str, image_path: str) -> None:
    """Copy an existing image into the person's known_faces folder."""
    if not os.path.isfile(image_path):
        print(f"[ERROR] File not found: {image_path}")
        return

    person_dir = os.path.join(KNOWN_FACES_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    ext      = os.path.splitext(image_path)[1].lower()
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest     = os.path.join(person_dir, f"{name}_{ts}{ext}")
    shutil.copy2(image_path, dest)

    print(f"[SUCCESS] Image copied to {dest}")
    print("          Run  python encode_faces.py  to update the database.")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Register a new person into the face recognition system."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # capture sub-command
    cap_p = sub.add_parser("capture", help="Capture photos from webcam.")
    cap_p.add_argument("--name",  required=True, help="Person's name (folder label).")
    cap_p.add_argument("--count", type=int, default=10,
                       help="Number of photos to capture (default: 10).")
    cap_p.add_argument("--delay", type=int, default=500,
                       help="Milliseconds between captures (default: 500).")

    # add sub-command
    add_p = sub.add_parser("add", help="Add an existing image file.")
    add_p.add_argument("--name",  required=True, help="Person's name (folder label).")
    add_p.add_argument("--image", required=True, help="Path to the image file.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.command == "capture":
        capture_faces(args.name, args.count, args.delay)
    elif args.command == "add":
        add_image(args.name, args.image)
