import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import pandas as pd

# ==================== CONFIG ====================
IMG_SIZE = (160, 160)
MODEL_PATH = "siamese_triplet_optimal_v3.keras"  # atau "stage2_finetuned.keras"
DATA_DIR = "data/training"  # atau "users_database" untuk attendance system

# ==================== CUSTOM LAYER ====================
class L2Normalization(keras.layers.Layer):
    """Custom L2 Normalization layer"""
    def __init__(self, **kwargs):
        super(L2Normalization, self).__init__(**kwargs)
    
    def call(self, inputs):
        return tf.nn.l2_normalize(inputs, axis=1)
    
    def get_config(self):
        return super(L2Normalization, self).get_config()

# ==================== LOAD MODEL ====================
def load_model(model_path):
    """Load trained embedding model"""
    print(f"📦 Loading model from: {model_path}")
    try:
        # Try loading with safe_mode=False for Lambda layers
        model = keras.models.load_model(
            model_path,
            custom_objects={'L2Normalization': L2Normalization},
            safe_mode=False  # Allow Lambda deserialization
        )
        print("   ✓ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        print("\n   💡 Trying alternative method...")
        
        try:
            # Alternative: Enable unsafe deserialization globally
            keras.config.enable_unsafe_deserialization()
            model = keras.models.load_model(
                model_path,
                custom_objects={'L2Normalization': L2Normalization}
            )
            print("   ✓ Model loaded successfully with unsafe mode!")
            return model
        except Exception as e2:
            print(f"   ✗ Still failed: {e2}")
            print("\n   💡 If you saved the embedding model separately, try:")
            print(f"      MODEL_PATH = 'embedding_model.keras'")
            return None

# ==================== PREPROCESSING ====================
def preprocess_image(image_path):
    """Preprocess image untuk model"""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 127.5 - 1.0  # Normalize to [-1, 1]
    return img

def get_embedding(model, image_path):
    """Get embedding vector dari image"""
    img = preprocess_image(image_path)
    img_batch = tf.expand_dims(img, axis=0)
    embedding = model.predict(img_batch, verbose=0)
    return embedding[0]

# ==================== LOAD DATASET ====================
def load_persons(data_dir, max_images_per_person=10):
    """
    Load dataset dari berbagai struktur:
    1. data/training/<person>/anchor/*.jpg + positive/*.jpg
    2. users_database/<person>/*.jpg
    """
    persons = {}
    
    print(f"\n📂 Loading data from: {data_dir}")
    
    if not os.path.exists(data_dir):
        print(f"   ✗ Directory not found!")
        return persons
    
    for person_name in os.listdir(data_dir):
        person_path = os.path.join(data_dir, person_name)
        if not os.path.isdir(person_path):
            continue
        
        images = []
        
        # Check struktur 1: anchor + positive folders
        anchor_dir = os.path.join(person_path, "anchor")
        positive_dir = os.path.join(person_path, "positive")
        
        if os.path.exists(anchor_dir):
            images += [os.path.join(anchor_dir, f) 
                      for f in os.listdir(anchor_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if os.path.exists(positive_dir):
            images += [os.path.join(positive_dir, f)
                      for f in os.listdir(positive_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Check struktur 2: langsung di folder person
        if len(images) == 0:
            images = [os.path.join(person_path, f)
                     for f in os.listdir(person_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Limit images per person
        if len(images) > max_images_per_person:
            images = images[:max_images_per_person]
        
        if len(images) > 0:
            persons[person_name] = images
            print(f"   ✓ {person_name}: {len(images)} images")
    
    return persons

# ==================== COMPUTE EMBEDDINGS ====================
def compute_all_embeddings(model, persons):
    """Compute embeddings untuk semua person"""
    print(f"\n🔄 Computing embeddings for {len(persons)} persons...")
    
    embeddings_dict = {}
    
    for person_name, images in persons.items():
        embeddings = []
        for img_path in images:
            emb = get_embedding(model, img_path)
            embeddings.append(emb)
        
        embeddings_dict[person_name] = {
            'embeddings': embeddings,
            'mean_embedding': np.mean(embeddings, axis=0),
            'std_embedding': np.std(embeddings, axis=0)
        }
        print(f"   ✓ {person_name}: {len(embeddings)} embeddings")
    
    return embeddings_dict

# ==================== SIMILARITY MATRIX ====================
def compute_similarity_matrix(embeddings_dict, method='cosine'):
    """
    Compute similarity matrix dengan berbagai method:
    - cosine: Cosine similarity [0, 1] (1 = sama)
    - euclidean: Euclidean distance [0, inf] (0 = sama)
    - dot: Dot product similarity (untuk L2 normalized)
    """
    person_names = list(embeddings_dict.keys())
    n = len(person_names)
    
    matrix = np.zeros((n, n))
    
    for i, name1 in enumerate(person_names):
        emb1 = embeddings_dict[name1]['mean_embedding']
        
        for j, name2 in enumerate(person_names):
            emb2 = embeddings_dict[name2]['mean_embedding']
            
            if method == 'cosine':
                # Cosine similarity: 1 = identical, 0 = orthogonal, -1 = opposite
                sim = cosine_similarity([emb1], [emb2])[0][0]
                matrix[i][j] = sim
            
            elif method == 'euclidean':
                # Euclidean distance: 0 = identical, larger = more different
                dist = np.linalg.norm(emb1 - emb2)
                matrix[i][j] = dist
            
            elif method == 'dot':
                # Dot product (untuk L2 normalized embeddings, mirip cosine)
                dot = np.dot(emb1, emb2)
                matrix[i][j] = dot
    
    return matrix, person_names

# ==================== VISUALIZATION ====================
def plot_similarity_matrix(matrix, person_names, method='cosine', save_path=None):
    """Plot similarity matrix dengan heatmap"""
    
    plt.figure(figsize=(12, 10))
    
    if method == 'cosine' or method == 'dot':
        # Untuk similarity: higher = more similar
        cmap = 'RdYlGn_r'  # Red (high) to Green (low)
        vmin, vmax = 0, 1
        fmt = '.3f'
    else:  # euclidean
        # Untuk distance: lower = more similar
        cmap = 'RdYlGn'  # Green (low) to Red (high)
        vmin, vmax = 0, np.max(matrix)
        fmt = '.2f'
    
    # Create heatmap
    sns.heatmap(
        matrix,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=person_names,
        yticklabels=person_names,
        vmin=vmin,
        vmax=vmax,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': method.capitalize()}
    )
    
    plt.title(f'Similarity Matrix ({method.upper()})', fontsize=16, fontweight='bold')
    plt.xlabel('Person', fontsize=12)
    plt.ylabel('Person', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   💾 Saved to: {save_path}")
    
    plt.show()

# ==================== ANALYSIS ====================
def analyze_matrix(matrix, person_names, method='cosine'):
    """Analyze similarity matrix dan berikan insight"""
    
    print("\n" + "="*80)
    print(f"📊 SIMILARITY MATRIX ANALYSIS ({method.upper()})")
    print("="*80)
    
    n = len(person_names)
    
    # Get diagonal and off-diagonal values
    diagonal = np.diag(matrix)
    off_diagonal = []
    for i in range(n):
        for j in range(n):
            if i != j:
                off_diagonal.append(matrix[i][j])
    
    off_diagonal = np.array(off_diagonal)
    
    print(f"\n📈 Statistics:")
    print(f"   • Number of persons: {n}")
    print(f"   • Total comparisons: {n * n}")
    print(f"   • Same-person (diagonal): {n}")
    print(f"   • Different-person (off-diagonal): {len(off_diagonal)}")
    
    if method == 'cosine' or method == 'dot':
        # Higher = more similar
        print(f"\n📊 Diagonal (same person - should be ~1.0):")
        print(f"   • Mean: {np.mean(diagonal):.4f}")
        print(f"   • Min: {np.min(diagonal):.4f}")
        print(f"   • Max: {np.max(diagonal):.4f}")
        
        print(f"\n📊 Off-diagonal (different persons - should be < 0.5):")
        print(f"   • Mean: {np.mean(off_diagonal):.4f}")
        print(f"   • Min: {np.min(off_diagonal):.4f}")
        print(f"   • Max: {np.max(off_diagonal):.4f}")
        print(f"   • Std: {np.std(off_diagonal):.4f}")
        
        # Quality assessment
        print(f"\n🎯 Quality Assessment:")
        
        # Check diagonal
        if np.min(diagonal) > 0.95:
            print(f"   ✅ Diagonal: EXCELLENT (all > 0.95)")
        elif np.min(diagonal) > 0.90:
            print(f"   ✓ Diagonal: Good (all > 0.90)")
        else:
            print(f"   ⚠️ Diagonal: Weak (min = {np.min(diagonal):.3f})")
        
        # Check off-diagonal
        if np.max(off_diagonal) < 0.4:
            print(f"   ✅ Separation: EXCELLENT (max < 0.4)")
        elif np.max(off_diagonal) < 0.5:
            print(f"   ✓ Separation: Good (max < 0.5)")
        elif np.max(off_diagonal) < 0.7:
            print(f"   ⚠️ Separation: Moderate (max = {np.max(off_diagonal):.3f})")
        else:
            print(f"   ❌ Separation: POOR (max = {np.max(off_diagonal):.3f})")
        
        # Confusion risk
        high_similarity = np.sum(off_diagonal > 0.6)
        if high_similarity > 0:
            print(f"   ⚠️ Confusion risk: {high_similarity} pairs with similarity > 0.6")
            
            # Find problematic pairs
            print(f"\n⚠️ Problematic pairs (similarity > 0.6):")
            for i in range(n):
                for j in range(i+1, n):
                    if matrix[i][j] > 0.6:
                        print(f"      • {person_names[i]} <-> {person_names[j]}: {matrix[i][j]:.3f}")
        
        # Recommended threshold
        threshold = np.max(off_diagonal) + 0.1
        threshold = min(threshold, 0.9)
        print(f"\n💡 Recommended threshold: {threshold:.2f}")
        
    else:  # euclidean
        # Lower = more similar
        print(f"\n📊 Diagonal (same person - should be ~0.0):")
        print(f"   • Mean: {np.mean(diagonal):.4f}")
        print(f"   • Min: {np.min(diagonal):.4f}")
        print(f"   • Max: {np.max(diagonal):.4f}")
        
        print(f"\n📊 Off-diagonal (different persons - should be > 1.0):")
        print(f"   • Mean: {np.mean(off_diagonal):.4f}")
        print(f"   • Min: {np.min(off_diagonal):.4f}")
        print(f"   • Max: {np.max(off_diagonal):.4f}")
        print(f"   • Std: {np.std(off_diagonal):.4f}")
        
        # Quality assessment
        print(f"\n🎯 Quality Assessment:")
        
        if np.max(diagonal) < 0.1:
            print(f"   ✅ Diagonal: EXCELLENT (all < 0.1)")
        elif np.max(diagonal) < 0.3:
            print(f"   ✓ Diagonal: Good (all < 0.3)")
        else:
            print(f"   ⚠️ Diagonal: Weak (max = {np.max(diagonal):.3f})")
        
        if np.min(off_diagonal) > 1.5:
            print(f"   ✅ Separation: EXCELLENT (min > 1.5)")
        elif np.min(off_diagonal) > 1.0:
            print(f"   ✓ Separation: Good (min > 1.0)")
        elif np.min(off_diagonal) > 0.5:
            print(f"   ⚠️ Separation: Moderate (min = {np.min(off_diagonal):.3f})")
        else:
            print(f"   ❌ Separation: POOR (min = {np.min(off_diagonal):.3f})")
        
        # Confusion risk
        low_distance = np.sum(off_diagonal < 0.8)
        if low_distance > 0:
            print(f"   ⚠️ Confusion risk: {low_distance} pairs with distance < 0.8")
            
            print(f"\n⚠️ Problematic pairs (distance < 0.8):")
            for i in range(n):
                for j in range(i+1, n):
                    if matrix[i][j] < 0.8:
                        print(f"      • {person_names[i]} <-> {person_names[j]}: {matrix[i][j]:.3f}")
        
        # Recommended threshold
        threshold = np.min(off_diagonal) - 0.2
        threshold = max(threshold, 0.5)
        print(f"\n💡 Recommended threshold: {threshold:.2f}")
    
    print("="*80)

# ==================== EXPORT TO CSV ====================
def export_to_csv(matrix, person_names, method='cosine', filename='similarity_matrix.csv'):
    """Export matrix ke CSV untuk analisis lebih lanjut"""
    df = pd.DataFrame(matrix, index=person_names, columns=person_names)
    df.to_csv(filename)
    print(f"\n💾 Matrix exported to: {filename}")

# ==================== MAIN ====================
def main():
    print("="*80)
    print("🔍 SIMILARITY MATRIX EVALUATION TOOL")
    print("="*80)
    
    # Load model
    model = load_model(MODEL_PATH)
    if model is None:
        print("\n❌ Failed to load model!")
        print(f"   Make sure {MODEL_PATH} exists")
        return
    
    # Load persons
    persons = load_persons(DATA_DIR, max_images_per_person=10)
    if len(persons) < 2:
        print("\n❌ Need at least 2 persons!")
        return
    
    # Compute embeddings
    embeddings_dict = compute_all_embeddings(model, persons)
    
    # Compute matrices dengan berbagai method
    methods = ['cosine', 'euclidean']
    
    for method in methods:
        print(f"\n{'='*80}")
        print(f"📊 Computing {method.upper()} matrix...")
        print(f"{'='*80}")
        
        matrix, person_names = compute_similarity_matrix(embeddings_dict, method=method)
        
        # Print matrix as table
        print(f"\n{method.upper()} Matrix:")
        print("-" * 80)
        
        # Header
        print("Person".ljust(15), end="")
        for name in person_names:
            print(f"{name[:12]:>12}", end="")
        print()
        print("-" * 80)
        
        # Rows
        for i, name1 in enumerate(person_names):
            print(f"{name1[:15]:15}", end="")
            for j, name2 in enumerate(person_names):
                value = matrix[i][j]
                
                # Color coding
                if i == j:
                    color = "\033[92m"  # Green for diagonal
                elif method == 'cosine':
                    if value > 0.7:
                        color = "\033[91m"  # Red if too similar
                    elif value > 0.5:
                        color = "\033[93m"  # Yellow moderate
                    else:
                        color = "\033[92m"  # Green good
                else:  # euclidean
                    if value < 0.5:
                        color = "\033[91m"  # Red if too close
                    elif value < 1.0:
                        color = "\033[93m"  # Yellow moderate
                    else:
                        color = "\033[92m"  # Green good
                
                if method == 'euclidean':
                    print(f"{color}{value:12.4f}\033[0m", end="")
                else:
                    print(f"{color}{value:12.4f}\033[0m", end="")
            print()
        
        print("-" * 80)
        
        # Analyze
        analyze_matrix(matrix, person_names, method)
        
        # Plot
        plot_path = f"similarity_matrix_{method}.png"
        plot_similarity_matrix(matrix, person_names, method, save_path=plot_path)
        
        # Export
        csv_path = f"similarity_matrix_{method}.csv"
        export_to_csv(matrix, person_names, method, csv_path)
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    print("\n📋 Generated files:")
    print("   • similarity_matrix_cosine.png")
    print("   • similarity_matrix_cosine.csv")
    print("   • similarity_matrix_euclidean.png")
    print("   • similarity_matrix_euclidean.csv")
    print("\n💡 Next steps:")
    print("   1. Check if off-diagonal values are low enough")
    print("   2. Use recommended thresholds in attendance system")
    print("   3. If separation poor, consider re-training with:")
    print("      - Larger margin (1.5 → 2.0)")
    print("      - More training data")
    print("      - More epochs")
    print("="*80)


if __name__ == "__main__":
    main()