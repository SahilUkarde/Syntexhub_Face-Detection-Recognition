"""
encode_faces.py
───────────────
Scans the known_faces/ directory and builds a face-encoding database.

Folder structure expected:
    known_faces/
        Alice/
            photo1.jpg
            photo2.jpg
        Bob/
            photo1.png
        ...

Each sub-folder name becomes the person's label.
Run this script once (or whenever you add new people) before running
face_recognition_main.py.
"""

import os
import cv2
import face_recognition
import pickle
import argparse

KNOWN_FACES_DIR = "known_faces"
ENCODINGS_FILE  = "models/encodings.pkl"
SUPPORTED_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def encode_faces(known_faces_dir: str = KNOWN_FACES_DIR,
                 encodings_file: str  = ENCODINGS_FILE,
                 model: str           = "hog") -> None:
    """
    Iterate over every person's folder, detect faces, generate encodings,
    and save them all to a single pickle file.
    """
    known_encodings: list = []
    known_names:     list = []

    if not os.path.isdir(known_faces_dir):
        print(f"[ERROR] Directory '{known_faces_dir}' not found.")
        print("  Create it and add sub-folders named after each person.")
        return

    persons = sorted(os.listdir(known_faces_dir))
    if not persons:
        print(f"[WARN] No person folders found inside '{known_faces_dir}'.")
        return

    for person_name in persons:
        person_dir = os.path.join(known_faces_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        print(f"\n[INFO] Processing: {person_name}")
        image_files = [
            f for f in os.listdir(person_dir)
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
        ]

        if not image_files:
            print(f"  [WARN] No supported images found for {person_name}. Skipping.")
            continue

        person_count = 0
        for image_file in image_files:
            image_path = os.path.join(person_dir, image_file)
            print(f"  → {image_file}", end=" ... ")

            # Load image and convert BGR→RGB
            bgr = cv2.imread(image_path)
            if bgr is None:
                print("SKIP (unreadable)")
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            # Detect face locations
            boxes = face_recognition.face_locations(rgb, model=model)
            if not boxes:
                print("SKIP (no face detected)")
                continue

            # Use only the first (largest) face per image
            encodings = face_recognition.face_encodings(rgb, boxes)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(person_name)
                person_count += 1
                print(f"OK (face encoded)")
            else:
                print("SKIP (encoding failed)")

        print(f"  ✔ {person_count} encoding(s) added for {person_name}")

    if not known_encodings:
        print("\n[WARN] No faces were encoded. Nothing saved.")
        return

    # Persist to disk
    os.makedirs(os.path.dirname(encodings_file), exist_ok=True)
    with open(encodings_file, "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_names}, f)

    print(f"\n[SUCCESS] Encoded {len(known_encodings)} face(s) across "
          f"{len(set(known_names))} person(s).")
    print(f"          Saved → {encodings_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build face-encoding database from known_faces/ directory."
    )
    parser.add_argument("--dir",    default=KNOWN_FACES_DIR,
                        help="Root directory of known faces (default: known_faces/)")
    parser.add_argument("--output", default=ENCODINGS_FILE,
                        help="Output path for encodings pickle (default: models/encodings.pkl)")
    parser.add_argument("--model",  choices=["hog", "cnn"], default="hog",
                        help="Detection model: hog (CPU, fast) | cnn (GPU, accurate)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    encode_faces(args.dir, args.output, args.model)
