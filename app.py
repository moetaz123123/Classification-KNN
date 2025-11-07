import tensorflow as tf
import numpy as np
import os
from tkinter import Tk, filedialog

# Charger le modèle pré-entraîné
model = tf.keras.models.load_model('art_classification_model_from_features.keras')

# Charger le modèle de base VGG16 pour l'extraction de features
base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False  # Geler les poids du modèle pré-entraîné

# Fonction d'extraction des features
def extract_features(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)  # Convertir l'image en tableau numpy
    img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension pour le batch
    img_array = img_array / 255.0  # Normaliser les pixels
    
    # Extraire les features avec le modèle de base (VGG16 sans couche supérieure)
    feature_extractor = tf.keras.models.Model(inputs=base_model.input, outputs=base_model.output)
    features = feature_extractor.predict(img_array)  # Extraction des features
    features = features.flatten()  # Aplatir les features pour le modèle de classification
    return features

# Fonction pour prédire l'image
def predict_image(image_path):
    features = extract_features(image_path)  # Extraire les features de l'image
    features = np.expand_dims(features, axis=0)  # Ajouter une dimension pour le batch
    
    # Prédire avec le modèle
    predictions = model.predict(features)
    
    # Définir les classes (ajuste en fonction de tes classes réelles)
    class_names = ['car', 'paysage', 'portrait']  # Exemple de classes, ajuste-les si nécessaire
    
    # Choisir la classe avec la probabilité la plus élevée
    predicted_class = class_names[np.argmax(predictions)]
    probability = np.max(predictions)  # Probabilité de la classe prédite
    
    return predicted_class, probability

# Interface avec Tkinter pour choisir l'image
def main():
    print("Bienvenue dans l'outil de classification d'images de tableaux artistiques.")
    
    # Initialiser Tkinter et cacher la fenêtre principale
    root = Tk()
    root.withdraw()  # Cela cache la fenêtre principale Tkinter
    
    # Ouvrir une fenêtre de dialogue pour sélectionner l'image
    image_path = filedialog.askopenfilename(
        title="Sélectionner une image", 
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
    )

    # Vérifier si un fichier a été sélectionné
    if not image_path:
        print("Aucune image sélectionnée.")
        return

    # Vérifier si l'image existe
    if not os.path.isfile(image_path):
        print(f"Erreur : Le fichier '{image_path}' n'existe pas.")
        return
    
    # Prédire la classe de l'image
    predicted_class, probability = predict_image(image_path)
    
    # Afficher la classe et la probabilité prédite
    print(f"Cette image est classifiée comme : {predicted_class} avec une probabilité de {probability*100:.2f}%")

if __name__ == '__main__':
    main()
