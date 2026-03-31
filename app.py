import os
import requests
import sqlite3
from flask import Flask, render_template, request
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_KEY")
GROUP_ID = os.getenv("PERSON_GROUP_ID")

headers_json = {
    'Ocp-Apim-Subscription-Key': AZURE_KEY,
    'Content-Type': 'application/json'
}

headers_binary = {
    'Ocp-Apim-Subscription-Key': AZURE_KEY,
    'Content-Type': 'application/octet-stream'
}

# ---------------- DATABASE ----------------
def init_db():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            person_id TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_user(name, person_id):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users (name, person_id) VALUES (?, ?)", (name, person_id))
    conn.commit()
    conn.close()

def get_user(person_id):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM users WHERE person_id=?", (person_id,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

# ---------------- AZURE ----------------
def create_group():
    url = f"{AZURE_ENDPOINT}/face/v1.0/persongroups/{GROUP_ID}"
    requests.put(url, headers=headers_json, json={"name": "Users"})

def create_person(name):
    url = f"{AZURE_ENDPOINT}/face/v1.0/persongroups/{GROUP_ID}/persons"
    res = requests.post(url, headers=headers_json, json={"name": name})
    return res.json()["personId"]

def add_face(person_id, image):
    url = f"{AZURE_ENDPOINT}/face/v1.0/persongroups/{GROUP_ID}/persons/{person_id}/persistedFaces"
    requests.post(url, headers=headers_binary, data=image)

def train():
    url = f"{AZURE_ENDPOINT}/face/v1.0/persongroups/{GROUP_ID}/train"
    requests.post(url, headers=headers_json)

def detect_face(image):
    url = f"{AZURE_ENDPOINT}/face/v1.0/detect"
    res = requests.post(url, headers=headers_binary, data=image)
    faces = res.json()
    return faces[0]["faceId"] if faces else None

def identify(face_id):
    url = f"{AZURE_ENDPOINT}/face/v1.0/identify"
    body = {
        "personGroupId": GROUP_ID,
        "faceIds": [face_id],
        "maxNumOfCandidatesReturned": 1,
        "confidenceThreshold": 0.6
    }
    res = requests.post(url, headers=headers_json, json=body)
    return res.json()

# ---------------- ROUTES ----------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register_page')
def register_page():
    return render_template('register.html')

@app.route('/login_page')
def login_page():
    return render_template('login.html')

# REGISTER
@app.route('/register', methods=['POST'])
def register():
    name = request.form['name']
    image = request.files['image'].read()

    person_id = create_person(name)
    add_face(person_id, image)
    train()
    save_user(name, person_id)

    return f"{name} registered successfully!"

# LOGIN
@app.route('/login', methods=['POST'])
def login():
    image = request.files['image'].read()

    face_id = detect_face(image)
    if not face_id:
        return "No face detected"

    result = identify(face_id)

    if result[0]['candidates']:
        person_id = result[0]['candidates'][0]['personId']
        confidence = result[0]['candidates'][0]['confidence']
        name = get_user(person_id)

        return f"Welcome {name}! (Confidence: {round(confidence,2)})"
    else:
        return "Not recognized"

# ---------------- MAIN ----------------
if __name__ == '__main__':
    init_db()
    app.run(debug=True)