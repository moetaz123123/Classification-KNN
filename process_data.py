import os
import cv2
import shutil
from tqdm import tqdm

# Définir les répertoires
SOURCE_DIR = "dataset"  # Répertoire source contenant les images originales
TARGET_DIR = "data"     # Répertoire cible pour les images prétraitées

# Paramètres globaux
TRAIN_RATIO = 0.7  # Proportion des données pour l'entraînement
TARGET_SIZE = (224, 224)  # Taille des images après redimensionnement

# Fonction : Correction d'illumination (CLAHE)
def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

# Fonction : Réduction du bruit (Filtrage bilatéral pour conserver les bords)
def reduce_noise(img):
    return cv2.bilateralFilter(img, 9, 75, 75)

# Fonction : Redimensionnement
def resize_image(img, target_size):
    return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

# Fonction principale de prétraitement
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = apply_clahe(img)          # Correction d'illumination
    img = reduce_noise(img)         # Réduction du bruit
    img = resize_image(img, TARGET_SIZE)  # Redimensionnement
    return img

# Fonction : Diviser les images entre "train" et "validation"
def split_data(source_dir, target_dir, train_ratio=TRAIN_RATIO):
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)  # Supprimer les anciens fichiers cibles
    for split in ['train', 'validation']:
        for category in os.listdir(source_dir):
            os.makedirs(os.path.join(target_dir, split, category), exist_ok=True)

    for category in tqdm(os.listdir(source_dir), desc="Traitement des catégories"):
        category_path = os.path.join(source_dir, category)
        images = os.listdir(category_path)
        train_count = int(len(images) * train_ratio)

        for idx, img_name in enumerate(images):
            src_path = os.path.join(category_path, img_name)
            target_split = "train" if idx < train_count else "validation"
            target_path = os.path.join(target_dir, target_split, category, img_name)

            # Prétraiter et sauvegarder
            preprocessed_img = preprocess_image(src_path)
            if preprocessed_img is not None:
                cv2.imwrite(target_path, preprocessed_img)

# Fonction principale pour exécuter le pipeline
def main():
    print("Début du prétraitement des images...")
    split_data(SOURCE_DIR, TARGET_DIR, TRAIN_RATIO)
    print(f"Prétraitement terminé. Les données sont dans le répertoire : {TARGET_DIR}")

if __name__ == "__main__":
    main()
