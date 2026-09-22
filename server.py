import os
import cv2
import sqlite3
import datetime
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form

app = FastAPI()
DATASET_DIR = "dataset"
CASCADE_FILE = "haarcascade_frontalface_default.xml"
RECOGNITION_THRESHOLD = 65

face_cascade = cv2.CascadeClassifier(CASCADE_FILE)
recognizer = None

def get_recognizer():
    global recognizer
    model_path = os.path.join(DATASET_DIR, "trainer.yml")
    if os.path.exists(model_path):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(model_path)
    return recognizer

@app.post("/scan")
async def scan_face(action: str = Form(...), file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
    if len(faces) == 0:
        return {"status": "error", "message": "No face detected"}

    rec = get_recognizer()
    if rec is None:
        return {"status": "error", "message": "Model not trained"}

    x, y, w, h = faces[0]
    face_roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
    label, confidence = rec.predict(face_roi)

    if confidence > RECOGNITION_THRESHOLD:
        return {"status": "error", "message": "Unknown face"}

    # Fetch User Name
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM users WHERE id = ?", (label,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return {"status": "error", "message": "User ID not found"}

    user_name = row[0]
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO attendance (name, action, timestamp) VALUES (?, ?, ?)", (user_name, action, now))
    conn.commit()
    conn.close()

    return {"status": "success", "user": user_name, "action": action, "timestamp": now}

if __name__ == "__main__":
    import uvicorn
    # Make accessible across your mobile phone's Wi-Fi
    uvicorn.run(app, host="0.0.0.0", port=8000)
    