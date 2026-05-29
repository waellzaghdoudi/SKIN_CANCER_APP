# Skin Cancer Detection App

Application Web complète en Python (Flask) intégrant un modèle de Deep Learning (VGG16) pour la détection du cancer de la peau.

## Description
Cette application permet aux professionnels de santé de :
- Se connecter de manière sécurisée
- Soumettre une image d'une lésion cutanée
- Obtenir une prédiction (Bénin ou Malin) avec un taux de confiance
- Consulter l'historique des diagnostics

## Technologies utilisées
- Python / Flask
- TensorFlow / Keras (VGG16)
- MySQL
- Bootstrap 5
- HTML / CSS

## Installation
1. Installer les dépendances :
   pip install flask tensorflow numpy mysql-connector-python Pillow
2. Configurer MySQL et exécuter database.sql
3. Placer le modèle vgg16_malignant_benign.h5 dans /model
4. Lancer : python app.py
5. Ouvrir : http://127.0.0.1:5000

## Identifiants
- Username : admin
- Password : 1234

## Captures d'écran

### Page Login
![Login](screenshots/login.png)

### Dashboard
![Dashboard](screenshots/dashboard.png)

### Analyse Patient
![Predict](screenshots/predict.png)

### Résultat
![Result](screenshots/result.png)

### Liste Patients
![Patients](screenshots/patients.png)

## Auteur
Waell Zaghdoudi - 1ère année ingénieur Technologies Avancées (1TA)
