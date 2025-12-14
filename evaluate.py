import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

MODEL_PATH = "siamese_triplet_optimal_v2.keras"
PERSON_DIR = "data/training"
THRESHOLD = 0.6   # semakin kecil semakin ketat

# ---------- CUSTOM LAYER ----------
class L2Normalization(layers.Layer):
    def __init__(self, **kwargs):
        super(L2Normalization, self).__init__(**kwargs)
    
    def call(self, inputs):
        import tensorflow as tf
        return tf.nn.l2_normalize(inputs, axis=1)
    
    def get_config(self):
        return super(L2Normalization, self).get_config()

print("📌 Loading embedding model...")
embedding = load_model(MODEL_PATH, custom_objects={'L2Normalization': L2Normalization})
print("✅ Model loaded!\n")

# ---------- Load Image ----------
def load_img(path, target_size=(160, 160)):
    img = image.load_img(path, target_size=target_size)
    img = image.img_to_array(img)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# ---------- Buat dataset pair ----------
def create_pairs(person_dir):
    anchors = []
    compares = []
    labels = []

    users = os.listdir(person_dir)

    for user in users:
        user_path = os.path.join(person_dir, user)
        anchor_dir = os.path.join(user_path, "anchor")
        pos_dir = os.path.join(user_path, "positive")

        if not os.path.exists(anchor_dir) or not os.path.exists(pos_dir):
            continue

        anchor_imgs = [os.path.join(anchor_dir, f) for f in os.listdir(anchor_dir)]
        positive_imgs = [os.path.join(pos_dir, f) for f in os.listdir(pos_dir)]

        # positive pairs (same person)
        for a in anchor_imgs:
            for p in positive_imgs:
                anchors.append(a)
                compares.append(p)
                labels.append(1)

        # negative pairs (different persons)
        other_users = [u for u in users if u != user]
        negative_imgs = []

        for ou in other_users:
            neg_path = os.path.join(person_dir, ou, "anchor")
            if os.path.exists(neg_path):
                negative_imgs.extend([os.path.join(neg_path, f) for f in os.listdir(neg_path)])

        np.random.shuffle(negative_imgs)
        neg_samples = negative_imgs[:len(anchor_imgs)]

        for a, n in zip(anchor_imgs, neg_samples):
            anchors.append(a)
            compares.append(n)
            labels.append(0)

    return anchors, compares, labels

# ---------- Evaluate ----------
def evaluate():
    print("📌 Preparing dataset...")
    anchors, compares, labels = create_pairs(PERSON_DIR)

    y_true = []
    y_pred = []

    print("📌 Running prediction...\n")

    for a, c, label in zip(anchors, compares, labels):
        img1 = load_img(a)
        img2 = load_img(c)

        emb1 = embedding.predict(img1, verbose=0)
        emb2 = embedding.predict(img2, verbose=0)

        # L2 distance
        dist = np.linalg.norm(emb1 - emb2)

        pred = 1 if dist < THRESHOLD else 0

        y_true.append(label)
        y_pred.append(pred)

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\n===== 📊 EVALUATION RESULT =====")
    print(f"Accuracy : {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall   : {recall*100:.2f}%")
    print(f"F1 Score : {f1*100:.2f}%")
    print("================================\n")

    print("🔍 Contoh 10 hasil prediksi:")
    for i in range(10):
        print(f"Pair {i+1}: True={y_true[i]}, Pred={y_pred[i]}")

    return accuracy, precision, recall, f1


if __name__ == "__main__":
    evaluate()