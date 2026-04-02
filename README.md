# Face Recognition Login System

A Flask-based face recognition login system that uses OpenCV for face processing, MediaPipe for live landmarks and blink-based liveness detection, and a clean dark UI for registration and login.

## Features

- Webcam-only registration with guided multi-angle capture
- Five-frame registration flow with voice instructions and countdown
- Face landmark overlay on the login page using MediaPipe Face Mesh
- Blink detection before login to reduce spoofing
- Email-based registration and login validation
- Duplicate email prevention during registration
- Session-based login state on the home page
- OpenCV LBPH recognition when available
- Fallback embedding-based recognition when OpenCV contrib face APIs are unavailable
- Dark, professional UI with no emoji-based labels

## How it works

### Registration

1. Enter full name and email.
2. Start webcam capture.
3. The app guides the user through five face angles:
   - Face forward
   - Turn slightly left
   - Turn left
   - Turn slightly right
   - Turn right
4. Each capture uses a countdown and voice prompt.
5. The app stores the face samples under a unique user folder and retrains the model.

### Login

1. Enter the registered email.
2. OpenCV camera scan starts in the browser.
3. MediaPipe draws live face landmarks.
4. The user must blink once to pass liveness verification.
5. The app collects multiple login samples and compares them against the registered identity.
6. If the face matches the registered email, the user is logged in and redirected to the home page.

## Tech Stack

- Python 3.11
- Flask
- OpenCV
- NumPy
- MediaPipe in the browser via CDN
- HTML, CSS, JavaScript

## Project Structure

```text
app.py
requirements.txt
static/
  style.css
templates/
  index.html
  register.html
  login.html
faces/
users.json
labels.json
embeddings.npz
trainer.yml
```

## Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Current dependencies:

- flask
- numpy
- opencv-contrib-python

## Run the project

From the project root:

```bash
python3.11 app.py
```

Then open the app in your browser at:

```text
http://127.0.0.1:5000
```

## Notes

- Registration uses webcam only. File upload has been removed.
- Login requires both email and a live face scan.
- MediaPipe is loaded in the browser from a CDN, so no Python package installation is needed for landmarks or blink detection.
- If `cv2.face` is available, the app uses LBPH recognition.
- If `cv2.face` is not available, the app falls back to an embedding-based matcher.

## Files created at runtime

The app may create these files and folders during use:

- `faces/` - stored face samples for each user
- `users.json` - email to user mapping
- `labels.json` - OpenCV label mapping
- `embeddings.npz` - fallback face embeddings
- `trainer.yml` - LBPH model file when contrib face APIs are available

## Troubleshooting

### Camera does not open

- Allow camera permission in the browser.
- Make sure no other application is using the webcam.
- Try a different browser if needed.

### Registration says email already exists

- Each email can only be registered once.
- Use a different email or delete the existing record if you want to re-register.

### Login fails even with correct email

- Make sure the email was registered first.
- Re-register in better lighting.
- Keep your face centered during registration and login.
- Blink once before login recognition starts.
- If recognition is still weak, capture a cleaner set of training frames.

### Recognition is inaccurate

Try these adjustments:

- Improve lighting.
- Keep the webcam stable.
- Re-register the user with clearer samples.
- Increase the number of login samples in `templates/login.html`.
- Tune `ACCEPT_THRESHOLD` or `EMBEDDING_DISTANCE_THRESHOLD` in `app.py`.

## Development summary

This project started as an Azure Face API idea and was converted into a local face recognition system using OpenCV and browser-based MediaPipe features. The current flow focuses on:

- local face processing
- email validation
- liveness verification
- multiple face samples for better accuracy
- a clean professional interface

## License

See [LICENSE](LICENSE) for details.
