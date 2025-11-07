## Projet – Classification d’images par KNN et Deep Learning ##
 
Période : Janvier 2024
Technologies : Python · OpenCV · YOLOv8 · Scikit-learn
Établissement : ENET’Com Sfax

## Description du projet ##

Ce projet consiste à développer un pipeline complet de traitement et de classification d’images appliqué aux œuvres d’art. L’objectif est d’automatiser l’identification et la catégorisation des images en utilisant à la fois des méthodes classiques (KNN) et des techniques avancées de deep learning (CNN).

Le projet couvre l’ensemble du processus, depuis l’acquisition des images jusqu’à la prédiction finale, en passant par le prétraitement, l’extraction de caractéristiques et l’entraînement des modèles.

 ## Fonctionnalités principales ##

Acquisition et prétraitement des images

Chargement des images depuis différents formats et sources.

Nettoyage et normalisation des données (redimensionnement, conversion en niveaux de gris, ajustement des contrastes).

Augmentation des données pour améliorer la robustesse des modèles (rotation, flips, changements de luminosité).

Détection et extraction des objets

Utilisation de YOLOv8 pour détecter les objets ou parties spécifiques dans les images, afin d’améliorer la précision de classification.

Extraction des caractéristiques

Transformation des images en vecteurs de caractéristiques pertinents pour l’apprentissage automatique.

Application de techniques telles que le flattening, histogrammes, ou des descripteurs plus complexes selon le modèle.

Implémentation de modèles de classification

K-Nearest Neighbors (KNN) pour classification simple basée sur les distances entre vecteurs de caractéristiques.

Convolutional Neural Networks (CNN) pour extraction automatique de caractéristiques et classification profonde, capable de capturer des patterns complexes.

Évaluation et visualisation

Mesure de la performance des modèles via précision, rappel et F1-score.

Affichage des résultats de classification et comparaison entre KNN et CNN.

Visualisation des prédictions sur des images test pour validation qualitative.

## Architecture du projet ##

Python & OpenCV : Prétraitement et manipulation d’images.

YOLOv8 : Détection d’objets dans les images.

Scikit-learn : Implémentation et entraînement du modèle KNN.

Framework Deep Learning (TensorFlow/Keras ou PyTorch) : Implémentation des CNN pour la classification.

## Objectifs pédagogiques et techniques ##

Maîtriser les concepts de prétraitement et d’augmentation d’images pour l’apprentissage automatique.

Comparer les performances entre des méthodes classiques (KNN) et des modèles de deep learning (CNN).

Développer un pipeline complet de traitement d’images, de l’acquisition à la prédiction.

Appliquer des techniques modernes de détection d’objets (YOLOv8) pour améliorer la précision des modèles.
## Démonstration vidéo ##

Lien vers la démonstration interactive :

[🎥 Watch the demo video on Google Drive](https://drive.google.com/file/d/1cybTWzbtbSB4nfPSig11sbf2dAt5VPj-/view)
