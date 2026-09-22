import os
import sys
import cv2
import sqlite3
import datetime
import urllib.request
import numpy as np
import customtkinter as ctk
from PIL import Image

# ============================================================
# CONFIGURATION
# ============================================================

APP_NAME = "VisionPass | AI Biometric Attendance"
ADMIN_PIN = "1234"  # Default Admin PIN to open Admin Panel

DATASET_DIR = "dataset"
CASCADE_FILE = "haarcascade_frontalface_default.xml"
TRAINING_IMAGES = 30
RECOGNITION_THRESHOLD = 65
REQUIRED_STABLE_FRAMES = 4
MIN_FACE_SIZE = (80, 80)

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


# ============================================================
# ADMIN PANEL WINDOW
# ============================================================

class AdminWindow(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title("VisionPass | Admin Dashboard")
        self.geometry("750x520")
        self.resizable(False, False)
        self.grab_set()  # Modal window focus

        # Layout Tabs
        self.tabview = ctk.CTkTabview(self, width=710, height=480)
        self.tabview.pack(padx=20, pady=20)
        self.tab_users = self.tabview.add("Registered Users")
        self.tab_logs = self.tabview.add("All Attendance Logs")

        self.setup_users_tab()
        self.setup_logs_tab()

    def setup_users_tab(self):
        header = ctk.CTkLabel(self.tab_users, text="Manage Registered Personnel", font=ctk.CTkFont(size=18, weight="bold"))
        header.pack(pady=10)

        self.user_textbox = ctk.CTkTextbox(self.tab_users, width=650, height=260)
        self.user_textbox.pack(pady=10)

        btn_frame = ctk.CTkFrame(self.tab_users, fg_color="transparent")
        btn_frame.pack(fill="x", padx=30, pady=10)

        refresh_btn = ctk.CTkButton(btn_frame, text="Refresh List", command=self.load_users)
        refresh_btn.pack(side="left", padx=10)

        delete_btn = ctk.CTkButton(btn_frame, text="Delete User by ID", fg_color="#DA3633", hover_color="#B62324", command=self.delete_user_prompt)
        delete_btn.pack(side="right", padx=10)

        self.load_users()

    def load_users(self):
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, created_at FROM users ORDER BY id ASC")
        rows = cursor.fetchall()
        conn.close()

        self.user_textbox.delete("1.0", "end")
        self.user_textbox.insert("end", f"{'ID':<6} | {'Name':<35} | {'Registered Date'}\n")
        self.user_textbox.insert("end", "-" * 68 + "\n")
        for r in rows:
            self.user_textbox.insert("end", f"{r[0]:<6} | {r[1]:<35} | {r[2]}\n")

    def delete_user_prompt(self):
        dialog = ctk.CTkInputDialog(text="Enter User ID to delete:", title="Delete User")
        user_id = dialog.get_input()
        if user_id and user_id.isdigit():
            user_id = int(user_id)
            conn = sqlite3.connect("attendance.db")
            cursor = conn.cursor()
            cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
            conn.commit()
            conn.close()

            user_dir = os.path.join(DATASET_DIR, f"user_{user_id}")
            if os.path.exists(user_dir):
                import shutil
                shutil.rmtree(user_dir)

            self.load_users()
            self.parent.load_users_map()
            self.parent.status_label.configure(text=f"Status: User {user_id} removed. Retrain model!", text_color="#D29922")

    def setup_logs_tab(self):
        header = ctk.CTkLabel(self.tab_logs, text="System Attendance Records", font=ctk.CTkFont(size=18, weight="bold"))
        header.pack(pady=10)

        self.full_log_textbox = ctk.CTkTextbox(self.tab_logs, width=650, height=260)
        self.full_log_textbox.pack(pady=10)

        btn_frame = ctk.CTkFrame(self.tab_logs, fg_color="transparent")
        btn_frame.pack(fill="x", padx=30, pady=10)

        refresh_btn = ctk.CTkButton(btn_frame, text="Refresh Records", command=self.load_logs)
        refresh_btn.pack(side="left", padx=10)

        clear_btn = ctk.CTkButton(btn_frame, text="Clear All Records", fg_color="#DA3633", hover_color="#B62324", command=self.clear_logs)
        clear_btn.pack(side="right", padx=10)

        self.load_logs()

    def load_logs(self):
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, action, timestamp FROM attendance ORDER BY id DESC")
        rows = cursor.fetchall()
        conn.close()

        self.full_log_textbox.delete("1.0", "end")
        self.full_log_textbox.insert("end", f"{'ID':<6} | {'Name':<28} | {'Action':<10} | {'Timestamp'}\n")
        self.full_log_textbox.insert("end", "-" * 68 + "\n")
        for r in rows:
            self.full_log_textbox.insert("end", f"{r[0]:<6} | {r[1]:<28} | {r[2]:<10} | {r[3]}\n")

    def clear_logs(self):
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        cursor.execute("DELETE FROM attendance")
        conn.commit()
        conn.close()
        self.load_logs()
        self.parent.refresh_logs()


# ============================================================
# MAIN APPLICATION
# ============================================================

class VisionPassApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title(APP_NAME)
        self.geometry("1050x710")
        self.resizable(False, False)

        self.is_running = True
        self.after_id = None
        self.latest_face_detected = False
        self.recognized_name = None

        self.last_candidate = None
        self.stable_count = 0

        self.recognizer = None
        self.model_ready = False
        self.user_map = {}  # Maps {user_id: user_name}

        self.init_db()
        os.makedirs(DATASET_DIR, exist_ok=True)
        self.face_cascade = self.load_cascade()
        self.load_users_map()
        self.load_recognition_model()

        self.create_ui()
        self.cap = self.init_camera()

        if self.cap is None or not self.cap.isOpened():
            self.video_label.configure(text="Webcam could not be opened.\nCheck connection/permissions.")
            self.status_label.configure(text="Status: Camera Error", text_color="#E5534B")
        else:
            self.status_label.configure(text="Status: Ready", text_color="#3B8ED0")
            self.update_video_stream()

        self.refresh_logs()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def init_db(self):
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        # Personnel Registry Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            )
        """)
        # Attendance Logs Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                action TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def load_users_map(self):
        """Loads all registered IDs and names into memory."""
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        cursor.execute("SELECT id, name FROM users")
        self.user_map = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()

    def create_ui(self):
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        # Video Frame
        self.camera_frame = ctk.CTkFrame(self, corner_radius=12)
        self.camera_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        self.video_label = ctk.CTkLabel(self.camera_frame, text="Initializing Camera...")
        self.video_label.pack(expand=True, fill="both", padx=10, pady=10)

        # Control Panel Frame
        self.control_frame = ctk.CTkFrame(self, corner_radius=12)
        self.control_frame.grid(row=0, column=1, padx=(0, 20), pady=20, sticky="nsew")

        self.header_label = ctk.CTkLabel(self.control_frame, text="VisionPass System", font=ctk.CTkFont(size=22, weight="bold"))
        self.header_label.pack(pady=(16, 2))

        self.status_label = ctk.CTkLabel(self.control_frame, text="Status: Starting...", font=ctk.CTkFont(size=14), text_color="#3B8ED0")
        self.status_label.pack(pady=3)

        self.recognition_label = ctk.CTkLabel(self.control_frame, text="Recognition: Waiting...", font=ctk.CTkFont(size=13))
        self.recognition_label.pack(pady=(0, 8))

        # Actions
        self.register_btn = ctk.CTkButton(self.control_frame, text="📷  Register New Face", fg_color="#8957E5", hover_color="#7042C2", height=36, command=self.register_face_flow)
        self.register_btn.pack(fill="x", padx=25, pady=4)

        self.train_btn = ctk.CTkButton(self.control_frame, text="⚙  Train Recognition Model", height=36, command=self.train_model)
        self.train_btn.pack(fill="x", padx=25, pady=4)

        self.scan_in_btn = ctk.CTkButton(self.control_frame, text="➔  Scan Entry", fg_color="#2EA043", hover_color="#238636", height=36, command=lambda: self.log_attendance("ENTRY"))
        self.scan_in_btn.pack(fill="x", padx=25, pady=4)

        self.scan_out_btn = ctk.CTkButton(self.control_frame, text="➔  Scan Exit", fg_color="#DA3633", hover_color="#B62324", height=36, command=lambda: self.log_attendance("EXIT"))
        self.scan_out_btn.pack(fill="x", padx=25, pady=4)

        # Admin Access Button
        self.admin_btn = ctk.CTkButton(self.control_frame, text="🔒  Admin Dashboard", fg_color="#30363D", hover_color="#484F58", height=32, command=self.open_admin_panel)
        self.admin_btn.pack(fill="x", padx=25, pady=(8, 4))

        # Recent Logs List
        self.log_label = ctk.CTkLabel(self.control_frame, text="Recent Logs", font=ctk.CTkFont(size=14, weight="bold"))
        self.log_label.pack(pady=(8, 2))

        self.log_textbox = ctk.CTkTextbox(self.control_frame, height=160)
        self.log_textbox.pack(fill="both", padx=20, pady=(0, 15))

    def init_camera(self):
        cap = None
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(0)

        if cap and cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return cap

    def load_cascade(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        local_path = os.path.join(base_dir, CASCADE_FILE)

        if not os.path.exists(local_path) or os.path.getsize(local_path) < 100000:
            url = f"https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/{CASCADE_FILE}"
            urllib.request.urlretrieve(url, local_path)

        cascade = cv2.CascadeClassifier(local_path)
        return cascade

    def load_recognition_model(self):
        if not hasattr(cv2, "face"):
            self.model_ready = False
            return

        model_path = os.path.join(DATASET_DIR, "trainer.yml")
        if not os.path.exists(model_path):
            self.model_ready = False
            return

        try:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.recognizer.read(model_path)
            self.model_ready = True
        except Exception:
            self.model_ready = False

    def open_admin_panel(self):
        dialog = ctk.CTkInputDialog(text="Enter Admin PIN:", title="Admin Access")
        pin = dialog.get_input()
        if pin == ADMIN_PIN:
            AdminWindow(self)
        elif pin is not None:
            self.status_label.configure(text="Status: Invalid Admin PIN!", text_color="#E5534B")

    def register_face_flow(self):
        dialog = ctk.CTkInputDialog(text="Enter Full Name for Registration:", title="Register Personnel")
        name = dialog.get_input()
        if not name or name.strip() == "":
            return
        name = name.strip()

        # Insert or lookup existing person
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE LOWER(name) = LOWER(?)", (name,))
        row = cursor.fetchone()

        if row:
            user_id = row[0]
        else:
            cursor.execute("INSERT INTO users (name, created_at) VALUES (?, ?)", (name, now))
            conn.commit()
            user_id = cursor.lastrowid
        conn.close()

        self.load_users_map()
        self.capture_user_samples(user_id, name)

    def capture_user_samples(self, user_id, user_name):
        user_dir = os.path.join(DATASET_DIR, f"user_{user_id}")
        os.makedirs(user_dir, exist_ok=True)

        for filename in os.listdir(user_dir):
            file_path = os.path.join(user_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception:
                pass

        self.status_label.configure(text=f"Look at camera: Registering {user_name}...", text_color="#D29922")
        self.register_btn.configure(state="disabled")

        captured = 0
        while captured < TRAINING_IMAGES and self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=MIN_FACE_SIZE)

            if len(faces) == 1:
                x, y, w, h = faces[0]
                face = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
                captured += 1
                cv2.imwrite(os.path.join(user_dir, f"sample_{captured}.jpg"), face)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{user_name}: {captured}/{TRAINING_IMAGES}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Show exactly ONE face", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(520, 390))
            self.video_label.configure(image=ctk_image, text="")
            self.video_label.image = ctk_image
            self.update()

        self.register_btn.configure(state="normal")
        if captured >= TRAINING_IMAGES:
            self.status_label.configure(text=f"Status: {user_name} registered! Click Train Model.", text_color="#57AB5A")
        else:
            self.status_label.configure(text="Status: Registration canceled.", text_color="#E5534B")

    def train_model(self):
        if not hasattr(cv2, "face"):
            self.status_label.configure(text="Status: Install opencv-contrib-python", text_color="#E5534B")
            return

        faces_data = []
        labels = []

        # Iterate through all user directories (user_1, user_2, ...)
        for item in os.listdir(DATASET_DIR):
            item_path = os.path.join(DATASET_DIR, item)
            if os.path.isdir(item_path) and item.startswith("user_"):
                try:
                    uid = int(item.split("_")[1])
                except ValueError:
                    continue

                for img_name in os.listdir(item_path):
                    if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                        img_path = os.path.join(item_path, img_name)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            faces_data.append(img)
                            labels.append(uid)

        if len(faces_data) == 0:
            self.status_label.configure(text="Status: No registered faces found.", text_color="#E5534B")
            return

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces_data, np.array(labels))

        model_path = os.path.join(DATASET_DIR, "trainer.yml")
        recognizer.write(model_path)

        self.recognizer = recognizer
        self.model_ready = True
        self.status_label.configure(text=f"Status: Model trained for {len(set(labels))} personnel!", text_color="#57AB5A")

    def update_video_stream(self):
        if not self.is_running or self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if ret and frame is not None:
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=MIN_FACE_SIZE)
            self.latest_face_detected = len(faces) > 0
            current_frame_candidate = None

            for (x, y, w, h) in faces:
                name_tag = "Unknown"
                box_color = (0, 0, 255)

                if self.model_ready and self.recognizer is not None:
                    face_roi = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
                    label, confidence = self.recognizer.predict(face_roi)

                    if confidence < RECOGNITION_THRESHOLD and label in self.user_map:
                        person_name = self.user_map[label]
                        current_frame_candidate = person_name
                        name_tag = f"{person_name} ({int(confidence)})"
                        box_color = (0, 255, 0)
                    else:
                        name_tag = f"Unknown ({int(confidence)})"

                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                cv2.putText(frame, name_tag, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

            if current_frame_candidate is not None:
                if current_frame_candidate == self.last_candidate:
                    self.stable_count += 1
                else:
                    self.last_candidate = current_frame_candidate
                    self.stable_count = 1

                if self.stable_count >= REQUIRED_STABLE_FRAMES:
                    self.recognized_name = current_frame_candidate
                    self.recognition_label.configure(text=f"Recognition: {self.recognized_name} (Verified)", text_color="#57AB5A")
            else:
                self.stable_count = 0
                self.recognized_name = None
                self.recognition_label.configure(
                    text="Recognition: Searching / Unknown", 
                    text_color="#D29922" if self.latest_face_detected else "#8B949E"
                )

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(520, 390))
            self.video_label.configure(image=ctk_image, text="")
            self.video_label.image = ctk_image

        if self.is_running:
            self.after_id = self.after(20, self.update_video_stream)

    def log_attendance(self, action_type):
        if not self.latest_face_detected:
            self.status_label.configure(text="Status: No face detected!", text_color="#E5534B")
            return
        if not self.model_ready:
            self.status_label.configure(text="Status: Model not trained!", text_color="#E5534B")
            return
        if self.recognized_name is None:
            self.status_label.configure(text="Status: Face not verified!", text_color="#E5534B")
            return

        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        name = self.recognized_name

        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        cursor.execute("INSERT INTO attendance (name, action, timestamp) VALUES (?, ?, ?)", (name, action_type, now))
        conn.commit()
        conn.close()

        self.status_label.configure(text=f"Status: {action_type} Logged for {name}", text_color="#57AB5A")
        self.refresh_logs()

    def refresh_logs(self):
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        cursor.execute("SELECT name, action, timestamp FROM attendance ORDER BY id DESC LIMIT 10")
        rows = cursor.fetchall()
        conn.close()

        self.log_textbox.delete("1.0", "end")
        for row in rows:
            self.log_textbox.insert("end", f"[{row[2]}]\n{row[0]} - {row[1]}\n\n")

    def on_close(self):
        self.is_running = False
        if self.after_id is not None:
            self.after_cancel(self.after_id)
        if hasattr(self, "cap") and self.cap is not None and self.cap.isOpened():
            self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = VisionPassApp()
    app.mainloop()