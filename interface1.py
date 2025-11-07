import cv2
import numpy as np

from sklearn.preprocessing import StandardScaler
from PIL import Image
import tkinter as tk
from tkinter import filedialog
from skimage.feature import graycomatrix, graycoprops
import tensorflow as tf
# Charger le modèle
model = tf.keras.models.load_model('art_classification_model.keras')

# Fonction pour extraire les caractéristiques de l'image
def extract_features(image_path):
    # Charger l'image
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (150, 150))  # Redimensionner l'image

    # 1. Extraire l'histogramme des couleurs
    hist_b = cv2.calcHist([img_resized], [0], None, [256], [0, 256])  # Histogramme bleu
    hist_g = cv2.calcHist([img_resized], [1], None, [256], [0, 256])  # Histogramme vert
    hist_r = cv2.calcHist([img_resized], [2], None, [256], [0, 256])  # Histogramme rouge

    # 2. Extraire les caractéristiques de texture (GLCM, Grey Level Co-occurrence Matrix)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # Calculer la matrice GLCM
    glcm = graycomatrix(gray, [1], [0], symmetric=True, normed=True)
    
    # Extraire des propriétés à partir de la GLCM (exemple : contraste, homogénéité)
    contrast = graycoprops(glcm, prop='contrast')
    homogeneity = graycoprops(glcm, prop='homogeneity')
    
    # Prendre la moyenne de ces propriétés
    texture_features = np.hstack([contrast.mean(), homogeneity.mean()])

    # 3. Extraire les contours (méthode Canny)
    edges = cv2.Canny(gray, 100, 200)

    # Concaténer les caractéristiques extraites dans un seul vecteur
    features = np.hstack([hist_b.flatten(), hist_g.flatten(), hist_r.flatten(), texture_features, edges.flatten()])
    
    # Normalisation des caractéristiques (important pour la prédiction)
    scaler = StandardScaler()
    features = scaler.fit_transform(features.reshape(-1, 1)).flatten()

    return features

# Fonction pour effectuer la prédiction avec les caractéristiques extraites
def predict_image_with_features(image_path):
    # Extraire les caractéristiques
    features = extract_features(image_path)

    # Ajouter une dimension pour que le modèle puisse prédire
    features = np.expand_dims(features, axis=0)

    # Effectuer la prédiction
    predictions = model.predict(features)
    class_index = np.argmax(predictions)
    confidence = predictions[0][class_index]

    # Map des classes
    class_labels = ['portrait', 'paysage','car']
    predicted_class = class_labels[class_index]

    return predicted_class, confidence

# Fonction pour ouvrir la boîte de dialogue de sélection de fichier
def open_file_dialog():
    file_path = filedialog.askopenfilename(
        title="Sélectionner une image",
        filetypes=[("Images", "*.png;*.jpg;*.jpeg")]
    )
    
    if file_path:
        print(f"Fichier sélectionné: {file_path}")
        predicted_class, confidence = predict_image_with_features(file_path)
        print(f"Classe prédite : {predicted_class}")
        print(f"Confiance : {confidence:.2f}")

# Créer une fenêtre Tkinter
root = tk.Tk()
root.withdraw()  # Masquer la fenêtre principale

# Ouvrir la boîte de dialogue de sélection de fichier
open_file_dialog()

# Fermer la fenêtre Tkinter après l'exécution
root.quit()
