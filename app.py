from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
import os
import uuid
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from datetime import datetime, timedelta
import pandas as pd
import time

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

class L2Normalization(layers.Layer):
    def call(self, inputs):
        return tf.nn.l2_normalize(inputs, axis=1)

class CamApp(App):
    def build(self):
        self.base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "registered")
        os.makedirs(self.base_path, exist_ok=True)

        self.input_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "input_image")
        os.makedirs(self.input_path, exist_ok=True)

        self.log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "attendance_log")
        os.makedirs(self.log_path, exist_ok=True)

        self.model = tf.keras.models.load_model("siamese_triplet_optimal_v2.keras", custom_objects={'L2Normalization': L2Normalization})
        self.img_size = (self.model.input_shape[1], self.model.input_shape[2])

        self.user_embeddings = {}
        self.load_embeddings()

        self.web_cam = Image(size_hint=(1, .7))
        
        self.btn_register = Button(text="Register User", size_hint=(1, .1))
        
        self.status_label = Label(text="Ready", size_hint=(1, .08))
        self.blink_label = Label(text="Blink Count: 0", size_hint=(1, .05))

        self.btn_register.bind(on_press=self.register_face)

        layout = BoxLayout(orientation="vertical")
        layout.add_widget(self.web_cam)
        layout.add_widget(self.blink_label)
        layout.add_widget(self.btn_register)
        layout.add_widget(self.status_label)

        self.capture = cv2.VideoCapture(0)
        self.current_frame = None
        self.crop_face = None
        self.in_register = False
        self.register_name = None
        self.register_faces = []
        self.capture_count = 0

        # Blink detection variables
        self.blink_counter = 0
        self.blink_count = 0
        self.eyes_open_frames = 0
        self.eyes_closed_frames = 0
        self.min_closed_frames = 3
        self.min_open_frames = 3

        # Cooldown system
        self.last_verify_time = {}
        self.cooldown_minutes = 5

        Clock.schedule_interval(self.update, 1/33)
        return layout

    def update(self, *args):
        ret, frame = self.capture.read()
        if not ret:
            return

        self.current_frame = frame.copy()
        display_frame = frame.copy()

        if self.in_register:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 5)
            if len(faces) > 0:
                x, y, w, h = faces[0]
                
                # Draw bounding box for registration (no label)
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                face_region = gray[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(face_region, 1.2, 3)

                if len(eyes) >= 2:
                    eyes_sorted = sorted(eyes, key=lambda e: e[0])
                    left_eye = eyes_sorted[0]
                    right_eye = eyes_sorted[1]
                    
                    left_center = (int(x + left_eye[0] + left_eye[2]//2), int(y + left_eye[1] + left_eye[3]//2))
                    right_center = (int(x + right_eye[0] + right_eye[2]//2), int(y + right_eye[1] + right_eye[3]//2))
                    
                    keypoints = {
                        'left_eye': left_center,
                        'right_eye': right_center,
                        'nose': (int(x + w//2), int(y + h//2))
                    }
                    
                    aligned = self.align_crop(frame, keypoints)
                    if aligned is not None:
                        self.crop_face = aligned
                    else:   
                        face_crop = frame[y:y+h, x:x+w]
                        try:
                            self.crop_face = cv2.resize(face_crop, self.img_size)
                        except:
                            self.crop_face = None
                else:
                    expand = int(w * 0.3)
                    x = max(0, x - expand)
                    y = max(0, y - expand)
                    w = min(frame.shape[1] - x, w + expand * 2)
                    h = min(frame.shape[0] - y, h + expand * 2)
                    face_crop = frame[y:y+h, x:x+w]
                    try:
                        self.crop_face = cv2.resize(face_crop, self.img_size)
                    except:
                        self.crop_face = None
            
            buf = cv2.flip(display_frame, 0).tobytes()
            tex = Texture.create(size=(display_frame.shape[1], display_frame.shape[0]), colorfmt="bgr")
            tex.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
            self.web_cam.texture = tex
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        
        if len(faces) == 0:
            self.crop_face = None
            self.blink_counter = 0
            self.blink_count = 0
            self.eyes_open_frames = 0
            self.eyes_closed_frames = 0
            self.blink_label.text = "Blink Count: 0"
        else:
            x, y, w, h = faces[0]
            
            # Draw bounding box (no label)
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            face_region = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(face_region, 1.2, 3)

            if len(eyes) >= 2:
                eyes_sorted = sorted(eyes, key=lambda e: e[0])
                left_eye = eyes_sorted[0]
                right_eye = eyes_sorted[1]
                
                left_center = (int(x + left_eye[0] + left_eye[2]//2), int(y + left_eye[1] + left_eye[3]//2))
                right_center = (int(x + right_eye[0] + right_eye[2]//2), int(y + right_eye[1] + right_eye[3]//2))
                
                keypoints = {
                    'left_eye': left_center,
                    'right_eye': right_center,
                    'nose': (int(x + w//2), int(y + h//2))
                }
                
                aligned = self.align_crop(frame, keypoints)
                if aligned is not None:
                    self.crop_face = aligned
                else:
                    expand = int(w * 0.3)
                    x_exp = max(0, x - expand)
                    y_exp = max(0, y - expand)
                    w_exp = min(frame.shape[1] - x_exp, w + expand * 2)
                    h_exp = min(frame.shape[0] - y_exp, h + expand * 2)
                    face_crop = frame[y_exp:y_exp+h_exp, x_exp:x_exp+w_exp]
                    try:
                        self.crop_face = cv2.resize(face_crop, self.img_size)
                    except:
                        self.crop_face = None
            else:
                expand = int(w * 0.3)
                x_exp = max(0, x - expand)
                y_exp = max(0, y - expand)
                w_exp = min(frame.shape[1] - x_exp, w + expand * 2)
                h_exp = min(frame.shape[0] - y_exp, h + expand * 2)
                face_crop = frame[y_exp:y_exp+h_exp, x_exp:x_exp+w_exp]
                try:
                    self.crop_face = cv2.resize(face_crop, self.img_size)
                except:
                    self.crop_face = None

            # Blink detection logic
            if len(eyes) == 0:
                self.eyes_closed_frames += 1
                self.eyes_open_frames = 0
                
                if self.eyes_closed_frames >= self.min_closed_frames:
                    self.blink_counter += 1
            else:
                if self.eyes_closed_frames >= self.min_closed_frames:
                    self.eyes_open_frames += 1
                    
                    if self.eyes_open_frames >= self.min_open_frames:
                        self.blink_count += 1
                        self.eyes_closed_frames = 0
                        self.eyes_open_frames = 0
                        self.blink_label.text = f"Blink Count: {self.blink_count}"
                        
                        # Trigger verify when blink count reaches 3
                        if self.blink_count >= 3:
                            self.verify_face()
                            self.blink_count = 0
                            self.blink_counter = 0
                            self.blink_label.text = "Blink Count: 0"
                else:
                    self.eyes_closed_frames = 0
                    self.eyes_open_frames = 0

        buf = cv2.flip(display_frame, 0).tobytes()
        tex = Texture.create(size=(display_frame.shape[1], display_frame.shape[0]), colorfmt="bgr")
        tex.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
        self.web_cam.texture = tex

    def align_crop(self, image, keypoints, margin=10):
        left_eye = keypoints['left_eye']
        right_eye = keypoints['right_eye']

        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))

        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D(((w/2), (h/2)), angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

        cx = int(keypoints['nose'][0])
        cy = int(keypoints['nose'][1])

        size = max(w, h) // 2
        x1 = max(cx - size//2 - margin, 0)
        y1 = max(cy - size//2 - margin, 0)
        x2 = min(cx + size//2 + margin, w)
        y2 = min(cy + size//2 + margin, h)

        cropped = rotated[y1:y2, x1:x2]
        if cropped.size == 0:
            return None
        
        try:
            resized = cv2.resize(cropped, self.img_size, interpolation=cv2.INTER_CUBIC)
            return resized
        except:
            return None

    def preprocess_img(self, img):
        if img is None or img.size == 0:
            return None
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype("float32") / 255.0
        
        return np.expand_dims(img_norm, axis=0)

    def augment_face(self, img):
        augmented = []
        
        augmented.append(img)
        
        flipped = cv2.flip(img, 1)
        augmented.append(flipped)
        
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), 5, 1.0)
        rot1 = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        augmented.append(rot1)
        
        M = cv2.getRotationMatrix2D((w/2, h/2), -5, 1.0)
        rot2 = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        augmented.append(rot2)
        
        brightness = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
        augmented.append(brightness)
        
        darkness = cv2.convertScaleAbs(img, alpha=0.8, beta=-10)
        augmented.append(darkness)
        
        return augmented

    def embed_image(self, img):
        prep = self.preprocess_img(img)
        if prep is None:
            return None
        emb = self.model.predict(prep, verbose=0)[0]
        return emb / np.linalg.norm(emb)

    def load_embeddings(self):
        self.user_embeddings = {}
        if not os.path.isdir(self.base_path):
            return
        for user in os.listdir(self.base_path):
            emb_dir = os.path.join(self.base_path, user, "embeddings")
            if not os.path.isdir(emb_dir):
                continue
            all_embs = []
            for f in os.listdir(emb_dir):
                if f.endswith("_combined.npy"):
                    arr = np.load(os.path.join(emb_dir, f))
                    all_embs.append(arr)
            if len(all_embs) > 0:
                self.user_embeddings[user] = np.vstack(all_embs)

    def check_cooldown(self, username):
        """Check if user is in cooldown period"""
        if username in self.last_verify_time:
            time_passed = datetime.now() - self.last_verify_time[username]
            if time_passed < timedelta(minutes=self.cooldown_minutes):
                remaining = timedelta(minutes=self.cooldown_minutes) - time_passed
                minutes = int(remaining.total_seconds() // 60)
                seconds = int(remaining.total_seconds() % 60)
                return False, minutes, seconds
        return True, 0, 0

    def log_attendance(self, username, confidence):
        """Log attendance to CSV file"""
        today = datetime.now().strftime("%Y-%m-%d")
        csv_file = os.path.join(self.log_path, f"attendance_{today}.csv")
        
        timestamp = datetime.now()
        tanggal = timestamp.strftime("%Y-%m-%d")
        jam = timestamp.strftime("%H:%M:%S")
        
        # Create new entry
        new_entry = {
            'Nama': [username],
            'Tanggal': [tanggal],
            'Jam': [jam]
        }
        
        # Check if file exists
        if os.path.exists(csv_file):
            # Read existing data
            df_existing = pd.read_csv(csv_file)
            # Append new entry
            df_new = pd.DataFrame(new_entry)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            # Save to CSV
            df_combined.to_csv(csv_file, index=False)
        else:
            # Create new file
            df = pd.DataFrame(new_entry)
            df.to_csv(csv_file, index=False)

    def ask_name(self):
        layout = BoxLayout(orientation='vertical')
        self.name_input = TextInput(hint_text="Masukkan nama user", multiline=False)
        ok_btn = Button(text="OK", size_hint=(1, .3))
        layout.add_widget(self.name_input)
        layout.add_widget(ok_btn)
        self.popup = Popup(title="Register User", content=layout, size_hint=(.6, .35))
        ok_btn.bind(on_press=self.confirm_register)
        self.popup.open()

    def register_face(self, *args):
        if self.crop_face is None:
            self.status_label.text = "No face detected"
            return
        self.ask_name()

    def confirm_register(self, instance):
        name = self.name_input.text.strip()
        if name == "":
            self.status_label.text = "Nama tidak boleh kosong"
            return
        self.popup.dismiss()

        user_dir = os.path.join(self.base_path, name.lower())
        emb_dir = os.path.join(user_dir, "embeddings")
        os.makedirs(user_dir, exist_ok=True)
        os.makedirs(emb_dir, exist_ok=True)

        self.in_register = True
        self.register_name = name
        self.register_faces = []
        self.capture_count = 0
        self.status_label.text = f"Capturing faces for {name}: 0/10"
        Clock.schedule_interval(self.capture_step, 1.5)

    def capture_step(self, dt):
        if not self.in_register:
            return False
        if self.capture_count >= 10:
            self.in_register = False
            Clock.schedule_once(self.finish_embedding, 0.1)
            return False
        if self.crop_face is None:
            return True

        face = self.crop_face.copy()
        user_dir = os.path.join(self.base_path, self.register_name)
        filename = f"{uuid.uuid1()}.jpg"
        img_path = os.path.join(user_dir, filename)
        cv2.imwrite(img_path, face)

        self.register_faces.append(face)
        self.capture_count += 1
        self.status_label.text = f"Capturing faces for {self.register_name}: {self.capture_count}/10"
        return True

    def finish_embedding(self, dt):
        if len(self.register_faces) == 0:
            self.status_label.text = "No faces captured"
            self.register_faces = []
            self.register_name = None
            self.capture_count = 0
            return

        all_faces = []
        for face in self.register_faces:
            augmented = self.augment_face(face)
            all_faces.extend(augmented)

        valid_preps = [self.preprocess_img(img) for img in all_faces]
        valid_preps = [p for p in valid_preps if p is not None]
        
        if len(valid_preps) == 0:
            self.status_label.text = "No valid faces"
            self.register_faces = []
            self.register_name = None
            self.capture_count = 0
            return

        batch = np.vstack(valid_preps)
        embeddings = self.model.predict(batch, verbose=0, batch_size=32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        combined = np.mean(embeddings, axis=0)
        combined = combined / np.linalg.norm(combined)
        
        user_dir = os.path.join(self.base_path, self.register_name)
        emb_dir = os.path.join(user_dir, "embeddings")
        save_path = os.path.join(emb_dir, f"{self.register_name}_combined.npy")
        np.save(save_path, combined)

        self.user_embeddings[self.register_name] = np.array([combined])
        self.status_label.text = f"Registered: {self.register_name}"

        self.register_faces = []
        self.register_name = None
        self.capture_count = 0

    def verify_face(self, *args):
        if self.crop_face is None:
            self.status_label.text = "No face detected"
            return
        if not self.user_embeddings:
            self.status_label.text = "No registered embeddings"
            return

        # Start timing
        start_time = time.time()
        print("\n" + "="*60)
        print("FACE RECOGNITION TIMING ANALYSIS")
        print("="*60)

        save_img_path = os.path.join(self.input_path, "input.jpg")
        cv2.imwrite(save_img_path, self.crop_face)

        # Time: Preprocessing
        prep_start = time.time()
        prep = self.preprocess_img(self.crop_face)
        if prep is None:
            self.status_label.text = "Invalid face"
            return
        prep_time = (time.time() - prep_start) * 1000
        print(f"[1] Preprocessing Time: {prep_time:.2f} ms")

        # Time: Model prediction
        predict_start = time.time()
        emb = self.model.predict(prep, verbose=0).flatten()
        emb = emb / np.linalg.norm(emb)
        predict_time = (time.time() - predict_start) * 1000
        print(f"[2] Model Prediction Time: {predict_time:.2f} ms")

        # Time: Similarity comparison
        compare_start = time.time()
        best_user = None
        best_score = -1
        threshold = 0.8

        for user, embs in self.user_embeddings.items():
            embs_flat = embs.reshape(len(embs), -1)
            embs_norm = embs_flat / np.linalg.norm(embs_flat, axis=1, keepdims=True)
            sims = np.dot(embs_norm, emb)
            mx = np.max(sims)
            if mx > best_score:
                best_score = mx
                best_user = user
        compare_time = (time.time() - compare_start) * 1000
        print(f"[3] Comparison Time: {compare_time:.2f} ms")

        # Total time
        total_time = (time.time() - start_time) * 1000
        print("-"*60)
        print(f"TOTAL RECOGNITION TIME: {total_time:.2f} ms")
        print("="*60)

        if best_score >= threshold:
            # Check cooldown
            can_verify, mins, secs = self.check_cooldown(best_user)
            if not can_verify:
                self.status_label.text = f"{best_user} already verified."
                print(f"Result: {best_user} (Cooldown: {mins}m {secs}s)")
                return
            
            # Update last verify time
            self.last_verify_time[best_user] = datetime.now()
            
            # Log attendance
            p = best_score * 100
            self.log_attendance(best_user, p)
            
            self.status_label.text = f"Verified: {best_user} ({p:.1f}%) - Logged"
            print(f"Result: VERIFIED - {best_user} (Confidence: {p:.1f}%)")
        else:
            self.status_label.text = f"Unverified"
            print(f"Result: UNVERIFIED (Best match: {best_user}, Score: {best_score:.3f})")
        
        print("="*60 + "\n")

if __name__ == "__main__":
    CamApp().run()