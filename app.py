import json
import os
import re
import time
import hashlib

import cv2
import numpy as np
from flask import Flask, jsonify, redirect, render_template, request, session, url_for

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-me")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACES_DIR = os.path.join(BASE_DIR, "faces")
MODEL_PATH = os.path.join(BASE_DIR, "trainer.yml")
LABELS_PATH = os.path.join(BASE_DIR, "labels.json")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "embeddings.npz")
USERS_PATH = os.path.join(BASE_DIR, "users.json")
FACE_SIZE = (200, 200)
ACCEPT_THRESHOLD = 70.0
EMBEDDING_FACE_SIZE = (64, 64)
EMBEDDING_DISTANCE_THRESHOLD = 12.0

CASCADE_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

os.makedirs(FACES_DIR, exist_ok=True)


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip())
    return cleaned[:60] if cleaned else "user"


def normalize_email(email: str) -> str:
    return email.strip().lower()


def is_valid_email(email: str) -> bool:
    return re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email) is not None


def email_hash(email: str) -> str:
    return hashlib.sha1(email.encode("utf-8")).hexdigest()[:10]


def load_users():
    if not os.path.exists(USERS_PATH):
        return {}
    with open(USERS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def save_users(users):
    with open(USERS_PATH, "w", encoding="utf-8") as file:
        json.dump(users, file, indent=2)


def decode_upload(file_storage):
    data = np.frombuffer(file_storage.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def extract_largest_face(img_bgr):
    if img_bgr is None:
        return None

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80),
    )

    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
    face = gray[y:y + h, x:x + w]
    return cv2.resize(face, FACE_SIZE)


def get_recognizer():
    if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
        return None
    return cv2.face.LBPHFaceRecognizer_create()


def face_vector(face_gray):
    normalized = cv2.equalizeHist(face_gray)
    small = cv2.resize(normalized, EMBEDDING_FACE_SIZE)
    return small.astype(np.float32).reshape(-1) / 255.0


def load_labels():
    if not os.path.exists(LABELS_PATH):
        return {}
    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def save_labels(labels):
    with open(LABELS_PATH, "w", encoding="utf-8") as file:
        json.dump(labels, file, indent=2)


def retrain_model():
    recognizer = get_recognizer()

    labels = {}
    images = []
    target_ids = []
    vectors = []
    names = []
    next_id = 0

    for person in sorted(os.listdir(FACES_DIR)):
        person_dir = os.path.join(FACES_DIR, person)
        if not os.path.isdir(person_dir):
            continue

        labels[str(next_id)] = person

        for filename in os.listdir(person_dir):
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            image_path = os.path.join(person_dir, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            if image.shape != FACE_SIZE:
                image = cv2.resize(image, FACE_SIZE)

            images.append(image)
            target_ids.append(next_id)
            vectors.append(face_vector(image))
            names.append(person)

        next_id += 1

    if len(images) == 0:
        return False, "No training images found."

    np.savez(EMBEDDINGS_PATH, vectors=np.array(vectors), names=np.array(names))

    if recognizer is None:
        return True, "Fallback model trained."

    recognizer.train(images, np.array(target_ids))
    recognizer.save(MODEL_PATH)
    save_labels(labels)
    return True, "Model trained."

# ---------------- HOME ----------------
@app.route('/')
def home():
    return render_template('index.html', logged_in_user=session.get('logged_in_user'))

@app.route('/register_page')
def register_page():
    return render_template('register.html')

@app.route('/login_page')
def login_page():
    return render_template('login.html')

# ---------------- REGISTER ----------------
@app.route('/register', methods=['POST'])
def register():
    name = request.form.get('name', '').strip()
    email = normalize_email(request.form.get('email', ''))
    files = request.files.getlist('images')

    if not name:
        return jsonify(success=False, name=None, message="Name is required")

    if not email:
        return jsonify(success=False, name=None, message="Email is required")

    if not is_valid_email(email):
        return jsonify(success=False, name=None, message="Please enter a valid email")

    if not files or len(files) == 0:
        return jsonify(success=False, name=None, message="No frames captured")

    users = load_users()
    if email in users:
        return jsonify(success=False, name=None, message="This email is already registered")

    person = f"{safe_name(name)}_{email_hash(email)}"
    person_dir = os.path.join(FACES_DIR, person)
    os.makedirs(person_dir, exist_ok=True)

    saved_count = 0

    # Process all captured frames from webcam
    for idx, file in enumerate(files):
        if file and file.filename != '':
            img_bgr = decode_upload(file)
            face = extract_largest_face(img_bgr)
            if face is not None:
                sample_path = os.path.join(person_dir, f"{int(time.time() * 1000)}_{idx}.png")
                cv2.imwrite(sample_path, face)
                saved_count += 1

    if saved_count == 0:
        return jsonify(success=False, name=person, message="No faces detected in captured frames ❌")

    ok, message = retrain_model()
    if not ok:
        return jsonify(success=False, name=person, message=message)

    users[email] = {
        "name": name,
        "email": email,
        "person_id": person,
        "created_at": int(time.time())
    }
    save_users(users)

    return jsonify(success=True, name=name, message=f"{name} registered with {saved_count} samples! ✅")

# ---------------- LOGIN ----------------
@app.route('/login', methods=['POST'])
def login():
    email = normalize_email(request.form.get('email', ''))
    file = request.files.get('image')

    if not email:
        return jsonify(success=False, message="Email is required")

    if not is_valid_email(email):
        return jsonify(success=False, message="Please enter a valid email")

    users = load_users()
    if email not in users:
        return jsonify(success=False, message="Email not registered")

    expected_person_id = users[email]["person_id"]
    display_name = users[email]["name"]

    if not file or file.filename == '':
        return jsonify(success=False, message="No file selected")

    recognizer = get_recognizer()

    img_bgr = decode_upload(file)
    face = extract_largest_face(img_bgr)

    if face is None:
        return jsonify(success=False, message="No face detected")

    if recognizer is not None and os.path.exists(MODEL_PATH):
        labels = load_labels()
        if labels:
            recognizer.read(MODEL_PATH)
            pred_id, confidence = recognizer.predict(face)
            pred_name = labels.get(str(pred_id), "Unknown")

            if confidence <= ACCEPT_THRESHOLD and pred_name == expected_person_id:
                session["logged_in_user"] = display_name
                return jsonify(
                    success=True,
                    name=display_name,
                    confidence=confidence,
                    message=f"Welcome {display_name}! ✅"
                )

            return jsonify(success=False, name=None, confidence=confidence, message="User not recognized ❌")

    if not os.path.exists(EMBEDDINGS_PATH):
        return jsonify(success=False, message="No trained model found. Please register first.")

    data = np.load(EMBEDDINGS_PATH)
    vectors = data["vectors"]
    names = data["names"]

    if len(vectors) == 0:
        return jsonify(success=False, message="No trained model found. Please register first.")

    unknown_vector = face_vector(face)
    distances = np.linalg.norm(vectors - unknown_vector, axis=1)
    best_idx = int(np.argmin(distances))
    best_distance = float(distances[best_idx])
    best_name = str(names[best_idx])

    if best_distance <= EMBEDDING_DISTANCE_THRESHOLD and best_name == expected_person_id:
        session["logged_in_user"] = display_name
        return jsonify(
            success=True,
            name=display_name,
            confidence=best_distance,
            message=f"Welcome {display_name}! ✅"
        )

    return jsonify(success=False, name=None, confidence=best_distance, message="User not recognized ❌")


@app.route('/logout')
def logout():
    session.pop('logged_in_user', None)
    return redirect(url_for('home'))

# ---------------- RUN ----------------
if __name__ == '__main__':
    app.run(debug=True)