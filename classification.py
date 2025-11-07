import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Définir les chemins des dossiers d'images
train_dir = 'data/train'
validation_dir = 'data/validation'

# Prétraitement des images (Data Augmentation et normalisation)
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,  # Normalisation des pixels entre 0 et 1
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,  # Flip vertical
    fill_mode='nearest',
    featurewise_center=True,  # Centrer les données en soustrayant la moyenne
    featurewise_std_normalization=True  # Normalisation par l'écart-type
)

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Chargement des données d'entraînement et de validation
train_generator = train_datagen.flow_from_directory(
    train_dir,
    batch_size=32,
    class_mode='categorical',  # Multiclasse
    target_size=(150, 150))  # Taille des images (résolution)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    batch_size=32,
    class_mode='categorical',
    target_size=(150, 150))

# Utilisation du modèle pré-entraîné VGG16 pour le Transfer Learning
base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False  # Geler les poids du modèle pré-entraîné

# Fonction d'extraction des features
def extract_features_from_generator(generator):
    features = []
    labels = []
    for i in range(len(generator)):
        batch_x, batch_y = generator[i]
        batch_features = base_model.predict(batch_x)
        batch_features = batch_features.reshape(batch_features.shape[0], -1)  # Aplatir les features
        features.append(batch_features)
        labels.append(batch_y)
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)

# Extraction des features d'entraînement et de validation
train_features, train_labels = extract_features_from_generator(train_generator)
validation_features, validation_labels = extract_features_from_generator(validation_generator)

# Construction du modèle pour apprendre à partir des features extraites
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(train_features.shape[1],)),  # Shape des features extraites
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),  # Dropout pour éviter le sur-apprentissage
    tf.keras.layers.Dense(3, activation='softmax')  # 3 classes : portrait, paysage, nature morte
])

# Compilation du modèle
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])

# Entraînement du modèle avec les features extraites
history = model.fit(
    train_features, 
    train_labels, 
    epochs=20, 
    batch_size=32,
    validation_data=(validation_features, validation_labels)
)

# Sauvegarder le modèle entraîné
model.save('art_classification_model_from_features.keras')

# Visualiser la courbe d'entraînement
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.title('Accuracy over Epochs')

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Loss over Epochs')

plt.show()
