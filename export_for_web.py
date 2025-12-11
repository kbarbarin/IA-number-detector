#!/usr/bin/env python3
"""
Script pour réexporter le modèle ONNX sans données externes
Compatible avec les navigateurs web
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import os

# Définir l'architecture du modèle (identique au notebook)
class DigitRecognitionCNN(nn.Module):
    def __init__(self):
        super(DigitRecognitionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

print("🔄 Chargement du modèle PyTorch...")

# Créer l'architecture du modèle
model = DigitRecognitionCNN()

# Essayer de charger les poids entraînés
model_path = 'digit_model_full.pth'
state_dict_path = 'digit_model.pth'

if os.path.exists(model_path):
    print(f"📥 Chargement du modèle complet: {model_path}")
    model = torch.load(model_path, map_location='cpu')
    print("✅ Poids du modèle entraîné chargés!")
elif os.path.exists(state_dict_path):
    print(f"📥 Chargement des poids: {state_dict_path}")
    model.load_state_dict(torch.load(state_dict_path, map_location='cpu'))
    print("✅ Poids du modèle entraîné chargés!")
else:
    print("⚠️  ATTENTION: Aucun modèle entraîné trouvé!")
    print("   Le modèle exporté aura des poids aléatoires.")
    print("   Exécutez d'abord le notebook pour entraîner le modèle.")
    response = input("\nContinuer quand même? (o/n): ")
    if response.lower() != 'o':
        print("❌ Export annulé.")
        exit(1)

model.eval()
model.cpu()

# Créer une entrée factice
dummy_input = torch.randn(1, 1, 28, 28)

# Supprimer les anciens fichiers
onnx_path = "model.onnx"
data_path = onnx_path + ".data"

if os.path.exists(onnx_path):
    os.remove(onnx_path)
    print(f"🗑️  Ancien {onnx_path} supprimé")
    
if os.path.exists(data_path):
    os.remove(data_path)
    print(f"🗑️  Ancien {data_path} supprimé")

print("📤 Export du modèle ONNX...")

# Exporter le modèle
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=18,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print(f"✅ Modèle exporté: {onnx_path}")

# Vérifier la taille
size_mb = os.path.getsize(onnx_path) / (1024*1024)
print(f"📦 Taille: {size_mb:.2f} MB")

# Vérifier qu'il n'y a pas de fichier .data
if os.path.exists(data_path):
    print(f"⚠️  Fichier {data_path} détecté (peut poser problème sur le web)")
    print("   Le modèle est trop grand, les poids sont dans un fichier séparé.")
else:
    print("✅ Pas de fichier .data externe - parfait pour le web!")

# Vérifier le modèle
print("🔍 Vérification du modèle ONNX...")
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print("✅ Modèle vérifié avec succès!")

print("\n" + "="*60)
print("🎉 SUCCÈS! Le modèle est prêt pour le web!")
print("="*60)
print("\nInstructions:")
print("1. Assurez-vous que le serveur HTTP est lancé:")
print("   python3 -m http.server 8000")
print("2. Ouvrez: http://localhost:8000")
print("3. Dessinez un chiffre et testez!")
print("="*60)
