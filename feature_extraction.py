import numpy as np
import cv2
from glob import glob
import os


import cv2
import numpy as np
from skimage import feature

import os
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

def prepare_data(directory):
    features = []
    labels = []

    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)
        if not os.path.isdir(label_dir):
            continue

        for img_file in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_file)
            if img_file.endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                # Extraire les caractéristiques
                img_features = extract_features(img_path)
                features.append(img_features)
                labels.append(label)

    # Encodage des étiquettes
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    labels_one_hot = tf.keras.utils.to_categorical(labels_encoded)  # Convertir en one-hot encoding
    return np.array(features), np.array(labels_one_hot), label_encoder



def extract_features2(image_path):
    # Lire l'image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (150, 150))  # Redimensionner l'image

    # Convertir en RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 1. Histogramme des couleurs
    color_features = []
    for i in range(3):  # Pour chaque canal (R, G, B)
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        color_features.extend(hist.flatten())

    # 2. Textures (GLCM - Matrice de Co-occurrence)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    texture_features = glcm.flatten()

    # 3. Contours
    edges = cv2.Canny(gray_image, 50, 150)
    contour_features = np.sum(edges) / edges.size  # Fraction de contours

    # Concaténer toutes les caractéristiques
    features = np.concatenate([color_features, texture_features, [contour_features]])
    return features


def extract_feature1(image):
    # Convert the image to grayscale for feature extraction
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Edge detection using Canny
    edges = cv2.Canny(gray_image, 100, 200)

    # Local Binary Pattern (LBP) extraction
    lbp = feature.local_binary_pattern(gray_image, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)  # Normalize the LBP histogram

    # Combine edges and LBP histogram into a single feature vector
    features = np.hstack([lbp_hist, edges.ravel()])
    return features, gray_image, edges, lbp



# Extraction des caractéristiques de couleur
def color_histogram(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Définir les plages pour chaque canal
    h_hist = cv2.calcHist([hsv], [0], None, [256], [0, 256])  # Teinte (H)
    s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])  # Saturation (S)
    v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])  # Valeur (V)

    # Aplatir les histogrammes pour les rendre 1D
    h_hist = h_hist.flatten()
    s_hist = s_hist.flatten()
    v_hist = v_hist.flatten()

    # Retourner une liste de caractéristiques aplaties
    return h_hist, s_hist, v_hist

# Extraction des contours de l'image
def detect_contours(img):
    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Si l'image est en float32 (entre 0 et 1), convertissez-la en uint8
    if gray.dtype != np.uint8:
        gray = np.uint8(gray * 255)  # Assurez-vous que les valeurs sont entre 0 et 255
    
    # Appliquer un seuillage pour obtenir une image binaire
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Trouver les contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Convertir les contours en un format numérique (par exemple, en comptant le nombre de contours)
    contour_features = np.array([len(contours)])  # Nombre de contours comme caractéristique
    return contour_features

# Fonction d'extraction complète
def extract_features(img):
    color_features = color_histogram(img)
    contour_features = detect_contours(img)

    # Aplatir les caractéristiques de couleur
    color_features = np.concatenate(color_features, axis=0)  # Fusionner les trois histogrammes

    # Concaténer toutes les caractéristiques (faire attention à la dimension)
    all_features = np.concatenate([color_features, contour_features])

    return all_features


# Fonction de prétraitement des images et d'extraction des caractéristiques
def preprocess_and_extract_features(image_paths):
    features = []
    for path in image_paths:
        try:
            img = cv2.imread(path)  # Charger l'image sans modification
            if img is None:
                print(f"Erreur lors du chargement de l'image {path}")
                continue  # Passer à l'image suivante si l'image ne se charge pas

            # Extraire les caractéristiques sans redimensionner
            extracted_features = extract_features(img)

            # Ajouter les caractéristiques extraites à la liste
            features.append(extracted_features)

        except Exception as e:
            print(f"Erreur lors du traitement de l'image {path}: {e}")
            continue  # Continuer avec l'image suivante en cas d'erreur

    # Retourner les caractéristiques extraites sous forme de tableau numpy
    return np.array(features)
