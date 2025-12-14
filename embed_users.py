import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# ---------- CONFIG ----------
MODEL_PATH = "siamese_triplet_clean.keras"
BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "registered")

# Face detection model
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ---------- CUSTOM LAYER ----------
class L2Normalization(layers.Layer):
    """Custom L2 Normalization layer"""
    def __init__(self, **kwargs):
        super(L2Normalization, self).__init__(**kwargs)
    
    def call(self, inputs):
        return tf.nn.l2_normalize(inputs, axis=1)
    
    def get_config(self):
        return super(L2Normalization, self).get_config()

# ---------- LOAD MODEL ----------
model = tf.keras.models.load_model(
    MODEL_PATH, 
    custom_objects={'L2Normalization': L2Normalization}
)
print("[INFO] Model loaded!")


# ---------- FACE DETECTION & ALIGNMENT ----------
def detect_and_crop_face(img, margin=0.2):
    """
    Detect face dan crop dengan margin
    margin: persentase tambahan area di sekitar wajah (0.2 = 20%)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(30, 30)
    )
    
    if len(faces) == 0:
        print("[WARNING] No face detected, using center crop")
        # Fallback: center crop
        h, w = img.shape[:2]
        size = min(h, w)
        y = (h - size) // 2
        x = (w - size) // 2
        return img[y:y+size, x:x+size]
    
    # Ambil wajah terbesar
    (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
    
    # Tambah margin
    margin_w = int(w * margin)
    margin_h = int(h * margin)
    
    x1 = max(0, x - margin_w)
    y1 = max(0, y - margin_h)
    x2 = min(img.shape[1], x + w + margin_w)
    y2 = min(img.shape[0], y + h + margin_h)
    
    return img[y1:y2, x1:x2]


def preprocess(path):
    img = cv2.imread(path)
    if img is None:
        return None

    # Deteksi wajah
    face = detect_and_crop_face(img)
    
    # Convert ke RGB
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    # Resize sama seperti training
    face = cv2.resize(face, (160, 160))

    # Normalize
    face = face.astype("float32") / 255.0

    return np.expand_dims(face, axis=0)

def get_embedding(image_path):
    img = preprocess(image_path)
    if img is None:
        return None
    embedding = model.predict(img, verbose=0)
    return embedding

# ---------- COMBINE EMBEDDINGS FROM MULTIPLE IMAGES ----------
def get_averaged_embedding(image_paths):
    """
    Rata-rata embeddings dari multiple gambar untuk satu user
    """
    embeddings = []
    
    for path in image_paths:
        emb = get_embedding(path)
        if emb is not None:
            embeddings.append(emb)
        else:
            print(f"  [SKIP] Failed to process: {path}")
    
    if len(embeddings) == 0:
        print("  [ERROR] No valid embeddings generated!")
        return None
    
    # Rata-rata semua embeddings
    avg_embedding = np.mean(embeddings, axis=0)
    
    # L2 normalize hasil rata-rata (penting untuk Siamese network)
    avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
    
    return avg_embedding


# ---------- MAIN EMBEDDING LOOP ----------
print("\n" + "="*60)
print("GENERATING SINGLE EMBEDDING PER USER")
print("="*60)

for user in os.listdir(BASE_PATH):
    user_folder = os.path.join(BASE_PATH, user)

    if not os.path.isdir(user_folder):
        continue

    print(f"\n[USER] Processing: {user}")
    print(f"  Folder: {user_folder}")

    # Kumpulkan semua gambar untuk user ini
    image_paths = []
    for filename in os.listdir(user_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(user_folder, filename)
            image_paths.append(path)
    
    if len(image_paths) == 0:
        print("  [SKIP] No images found!")
        continue
    
    print(f"  Found {len(image_paths)} images")
    
    # Generate averaged embedding dari semua gambar
    print("  Generating embeddings...")
    avg_embedding = get_averaged_embedding(image_paths)
    
    if avg_embedding is None:
        print("  [ERROR] Failed to generate embedding for this user!")
        continue
    
    # Save embedding dengan nama user
    embed_dir = os.path.join(user_folder, "embeddings")
    os.makedirs(embed_dir, exist_ok=True)
    
    out_path = os.path.join(embed_dir, f"{user}_combined.npy")
    np.save(out_path, avg_embedding)
    
    print(f"  ✓ Saved: {out_path}")
    print(f"  ✓ Shape: {avg_embedding.shape}")
