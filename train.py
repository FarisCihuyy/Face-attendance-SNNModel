import os
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV3Small

IMG_SIZE = (160, 160)
TRAINING_DIR = "data/training" 
LFW_DIR = "data/lfw"
REGISTERED_DIR = "data/registered"

BATCH_SIZE = 16
EPOCHS = 50
MARGIN = 1 
MODEL_PATH = "siamese_triplet_optimal_v3.keras"

MIN_IMAGES_PER_PERSON = 3

class L2Normalization(layers.Layer):
    
    def __init__(self, **kwargs):
        super(L2Normalization, self).__init__(**kwargs)
    
    def call(self, inputs):
        return tf.nn.l2_normalize(inputs, axis=1)
    
    def get_config(self):
        return super(L2Normalization, self).get_config()

def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img

@tf.function
def augment_image(img):
    """Augmentasi untuk training"""
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.2)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.image.random_saturation(img, 0.8, 1.2)
    img = tf.image.random_hue(img, 0.1)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img

def triplet_loss(y_true, y_pred):
    anchor = y_pred[:, 0]
    positive = y_pred[:, 1]
    negative = y_pred[:, 2]

    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)

    loss = tf.maximum(pos_dist - neg_dist + MARGIN, 0.0)
    return tf.reduce_mean(loss)

def build_embedding_model():
    base = MobileNetV3Small(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        pooling="avg",
        weights="imagenet",
    )
    
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    inp = layers.Input(shape=IMG_SIZE + (3,))
    x = base(inp)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128)(x)
    x = L2Normalization()(x)

    return keras.Model(inp, x, name="EmbeddingModel")

def build_siamese_model(embedding):
    anchor_in = layers.Input(name="input_1", shape=IMG_SIZE + (3,))
    positive_in = layers.Input(name="input_2", shape=IMG_SIZE + (3,))
    negative_in = layers.Input(name="input_3", shape=IMG_SIZE + (3,))

    anchor_emb = embedding(anchor_in)
    pos_emb = embedding(positive_in)
    neg_emb = embedding(negative_in)

    class StackEmbeddings(layers.Layer):
        def call(self, inputs):
            return tf.stack(inputs, axis=1)
    
    output = StackEmbeddings()([anchor_emb, pos_emb, neg_emb])

    siamese = keras.Model(
        inputs=[anchor_in, positive_in, negative_in],
        outputs=output
    )

    siamese.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss=triplet_loss
    )

    return siamese

def load_all_persons():
    persons = {}
    
    print("\n📂 Loading LFW dataset...")
    if os.path.exists(LFW_DIR):
        lfw_count = 0
        for person_name in os.listdir(LFW_DIR):
            person_path = os.path.join(LFW_DIR, person_name)
            
            if not os.path.isdir(person_path):
                continue
            
            images = []
            for img_file in os.listdir(person_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    images.append(os.path.join(person_path, img_file))
            
            if len(images) >= MIN_IMAGES_PER_PERSON:
                persons[f"lfw_{person_name}"] = images
                lfw_count += 1
        
        print(f"   ✓ Loaded {lfw_count} persons from LFW")
    else:
        print(f"   ⚠️  LFW directory not found: {LFW_DIR}")
    
    print("\n📂 Loading registered users...")
    if os.path.exists(REGISTERED_DIR):
        reg_count = 0
        for person_name in os.listdir(REGISTERED_DIR):
            person_path = os.path.join(REGISTERED_DIR, person_name)
            
            if not os.path.isdir(person_path):
                continue
            
            images = []
            for img_file in os.listdir(person_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Skip embeddings folder
                    if 'embedding' not in img_file.lower():
                        images.append(os.path.join(person_path, img_file))
            
            if len(images) >= MIN_IMAGES_PER_PERSON:
                persons[f"user_{person_name}"] = images
                reg_count += 1
        
        print(f"   ✓ Loaded {reg_count} persons from registered users")
    else:
        print(f"   ⚠️  Registered directory not found: {REGISTERED_DIR}")
    
    print("\n📂 Loading structured training data...")
    if os.path.exists(TRAINING_DIR):
        train_count = 0
        for person_name in os.listdir(TRAINING_DIR):
            person_path = os.path.join(TRAINING_DIR, person_name)
            
            if not os.path.isdir(person_path):
                continue
            
            images = []
            
            # Check for anchor folder
            anchor_dir = os.path.join(person_path, "anchor")
            if os.path.exists(anchor_dir):
                for img_file in os.listdir(anchor_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        images.append(os.path.join(anchor_dir, img_file))
            
            # Check for positive folder
            positive_dir = os.path.join(person_path, "positive")
            if os.path.exists(positive_dir):
                for img_file in os.listdir(positive_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        images.append(os.path.join(positive_dir, img_file))
            
            if len(images) >= MIN_IMAGES_PER_PERSON:
                persons[f"train_{person_name}"] = images
                train_count += 1
        
        print(f"   ✓ Loaded {train_count} persons from training data")
    else:
        print(f"   ⚠️  Training directory not found: {TRAINING_DIR}")
    
    return persons

# ===================== GENERATE TRIPLETS ===========================
def generate_triplets(persons_dict, num_triplets_per_person=10):
    """
    Generate triplets from person dictionary
    
    Args:
        persons_dict: {person_name: [image_paths]}
        num_triplets_per_person: How many triplets to generate per person
    
    Returns:
        List of (anchor, positive, negative) tuples
    """
    print("\n🔄 Generating triplets...")
    
    triplets = []
    person_names = list(persons_dict.keys())
    
    for person_name in person_names:
        images = persons_dict[person_name]
        
        # Need at least 2 images for anchor & positive
        if len(images) < 2:
            continue
        
        # Generate multiple triplets for this person
        for _ in range(num_triplets_per_person):
            # Sample anchor and positive from same person
            anchor, positive = random.sample(images, 2)
            
            # Sample negative from different person
            negative_person = random.choice([p for p in person_names if p != person_name])
            negative = random.choice(persons_dict[negative_person])
            
            triplets.append((anchor, positive, negative))
    
    # Shuffle triplets
    random.shuffle(triplets)
    
    print(f"   ✓ Generated {len(triplets)} triplets")
    return triplets

# ====================== TF DATASET LOADER ===========================
def make_dataset(triplets, training=True):
    anchors = [t[0] for t in triplets]
    positives = [t[1] for t in triplets]
    negatives = [t[2] for t in triplets]

    ds = tf.data.Dataset.from_tensor_slices((anchors, positives, negatives))

    def process_triplet(a, p, n):
        img_a = load_image(a)
        img_p = load_image(p)
        img_n = load_image(n)
        
        if training:
            img_a = augment_image(img_a)
            img_p = augment_image(img_p)
            img_n = augment_image(img_n)
        
        return (
            {
                'input_1': img_a,
                'input_2': img_p,
                'input_3': img_n,
            },
            tf.zeros((1,))
        )

    ds = ds.map(process_triplet, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

# ======================= MAIN TRAINING ==============================
def main():
    print("="*80)
    print("🚀 TRIPLET NETWORK TRAINING WITH LFW DATASET")
    print("="*80)
    
    # 1. Load all persons from all sources
    persons = load_all_persons()
    
    if len(persons) == 0:
        print("\n❌ ERROR: No persons found in any directory!")
        print("   Please check:")
        print(f"   - LFW_DIR: {LFW_DIR}")
        print(f"   - REGISTERED_DIR: {REGISTERED_DIR}")
        print(f"   - TRAINING_DIR: {TRAINING_DIR}")
        return
    
    print(f"\n📊 Total persons loaded: {len(persons)}")
    print(f"📊 Total images: {sum(len(imgs) for imgs in persons.values())}")
    
    # Show sample
    print("\n📋 Sample persons:")
    for i, (person, images) in enumerate(list(persons.items())[:10]):
        print(f"   {i+1}. {person}: {len(images)} images")
    
    # 2. Generate triplets
    triplets = generate_triplets(persons, num_triplets_per_person=15)
    
    if len(triplets) == 0:
        print("\n❌ ERROR: No triplets generated!")
        return
    
    # 3. Split train/val
    split = int(0.9 * len(triplets))
    train_triplets = triplets[:split]
    val_triplets = triplets[split:]
    
    train_dataset = make_dataset(train_triplets, training=True)
    val_dataset = make_dataset(val_triplets, training=False)

    print(f"\n📊 Train triplets: {len(train_triplets)}")
    print(f"📊 Val triplets: {len(val_triplets)}")

    # 4. Build model
    print("\n🏗️  Building model...")
    embedding = build_embedding_model()
    siamese = build_siamese_model(embedding)
    
    print("   ✓ Model architecture:")
    print(f"     - Base: MobileNetV3Small (ImageNet pretrained)")
    print(f"     - Embedding dim: 128")
    print(f"     - Loss: Triplet Loss (margin={MARGIN})")

    # 5. Callbacks
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=5, 
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            MODEL_PATH, 
            save_best_only=True, 
            monitor='val_loss',
            verbose=1
        )
    ]

    # 6. Training
    print("\n" + "="*80)
    print("🚀 STARTING TRAINING")
    print("="*80)
    
    history = siamese.fit(
        train_dataset, 
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # 7. Save final model
    print(f"\n💾 Saving final embedding model to {MODEL_PATH}")
    embedding.save(MODEL_PATH)

    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"Model saved to: {MODEL_PATH}")
    print("\nNext steps:")
    print("1. Run test_trained_model.py to evaluate quality")
    print("2. Run generate_embeddings.py with new model")
    print("3. Check similarity matrix")


if __name__ == "__main__":
    main()
