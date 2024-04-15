#!/bin/bash

# Chemin du répertoire du projet
PROJECT_DIR="VQGanomaly-ResNet-CareNet-Vit"

# Vérifier si le répertoire du projet existe déjà
if [ ! -d "$PROJECT_DIR" ]; then
    # Clone the GitHub repository
    git clone https://github.com/ACOOLS/VQGanomaly-ResNet-CareNet-Vit.git
    cd "$PROJECT_DIR"
else
    echo "Le répertoire $PROJECT_DIR existe déjà."
    cd "$PROJECT_DIR"
fi

# Vérifier si le dataset existe déjà
DATASET_ZIP="screw_last_version.zip"
folder="screw"
if [ ! -d "$folder" ]; then
    # Download datasets
    wget "https://github.com/ACOOLS/VQGanomaly-ResNet-CareNet-Vit/releases/download/screw/$DATASET_ZIP" && unzip "$DATASET_ZIP" && rm "$DATASET_ZIP"
else
    echo "Le répertoire $folder existe déjà."
fi

DATASET_ZIP="wood_last_Version.zip"
folder="wood"
if [ ! -d "$folder" ]; then
    # Download datasets
    wget "https://github.com/ACOOLS/VQGanomaly-ResNet-CareNet-Vit/releases/download/wood/$DATASET_ZIP" && unzip "$DATASET_ZIP" && rm "$DATASET_ZIP"
else
    echo "Le répertoire $folder existe déjà."
fi

DATASET_ZIP="breast_dataset_last_version.zip"
folder="breast_dataset"
if [ ! -d "$folder" ]; then
    # Download datasets
    wget "https://github.com/ACOOLS/VQGanomaly-ResNet-CareNet-Vit/releases/download/breast/$DATASET_ZIP" && unzip "$DATASET_ZIP" && rm "$DATASET_ZIP"
else
    echo "Le répertoire $folder existe déjà."
fi

DATASET_ZIP="brain_mri_last_version.zip"
folder="brain_mri"
if [ ! -d "$folder" ]; then
    # Download datasets
    wget "https://github.com/ACOOLS/VQGanomaly-ResNet-CareNet-Vit/releases/download/brain/$DATASET_ZIP" && unzip "$DATASET_ZIP" && rm "$DATASET_ZIP"
else
    echo "Le répertoire $folder existe déjà."
fi

# Chemin de l'environnement virtuel
VENV_PATH="myenv"

# Vérifier si l'environnement virtuel existe déjà
if [ ! -d "$VENV_PATH" ]; then
    # Install Python environment manager and create a virtual environment
    apt install python3.10-venv -y 
    python -m venv "$VENV_PATH"
fi

# Activer l'environnement virtuel
source "$VENV_PATH/bin/activate"

# Vérifier si les packages sont déjà installés
PACKAGES_INSTALLED=$(pip freeze)
REQUIRED_PACKAGES=(
    "pytorch-lightning==1.0.8"
    "omegaconf==2.0.0"
    "albumentations==0.4.3"
    "opencv-python==4.5.5.64"
    "pudb==2019.2"
    "imageio==2.9.0"
    "imageio-ffmpeg==0.4.2"
    "torchmetrics==0.4.0"
    "test-tube>=0.7.5"
    "streamlit>=0.73.1"
    "einops==0.3.0"
    "torch-fidelity==0.3.0"
    "wandb"
)

for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if ! echo "$PACKAGES_INSTALLED" | grep -q "$pkg"; then
        pip install "$pkg"
    fi
done

# Set environment variable
export WANDB_API_KEY=cab75a759f850c41f43a9ee4951f98aa6f4a1863

# Install necessary libraries
apt install -y libgl1-mesa-glx

# Upgrade OpenCV
pip install --upgrade opencv-python


# update config files to paperspace


# Définition des chemins
FILE="/notebooks/VQGanomaly-ResNet-CareNet-Vit/configs/custom_vqgan_1CH_brainmri_classique.yaml"
OLD_PATH="/home/aurelie/datasets"
NEW_PATH="/notebooks/VQGanomaly-ResNet-CareNet-Vit"

# Vérification si le nouveau chemin est déjà utilisé
if grep -q "$NEW_PATH" "$FILE"; then
    echo "Le remplacement a déjà été effectué. Aucune action nécessaire."
else
    # Exécution de sed pour remplacer l'ancien chemin par le nouveau
    sed -i "s|$OLD_PATH|$NEW_PATH|g" "$FILE"
    echo "Chemin mis à jour dans le fichier."
fi

FILE="/notebooks/VQGanomaly-ResNet-CareNet-Vit/configs/custom_vqgan_1CH_breast_classique.yaml"

# Vérification si le nouveau chemin est déjà utilisé
if grep -q "$NEW_PATH" "$FILE"; then
    echo "Le remplacement a déjà été effectué. Aucune action nécessaire."
else
    # Exécution de sed pour remplacer l'ancien chemin par le nouveau
    sed -i "s|$OLD_PATH|$NEW_PATH|g" "$FILE"
    echo "Chemin mis à jour dans le fichier."
fi

# TRAIN

python main.py --paperspace --base configs/custom_vqgan_1CH_breast_classique.yaml -t --gpus 0, > reast_classique.logs

python main.py --paperspace --base configs/custom_vqgan_1CH_brainmri_classique.yaml -t --gpus 0, > brainmri_classique.logs
