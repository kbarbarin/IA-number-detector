# 🚀 Instructions pour exporter le modèle

## Étape 1 : Sauvegarder les poids du modèle entraîné

Dans le notebook `train_model.ipynb`, exécutez la cellule 10 :

```python
# Sauvegarder le modèle entraîné
model_path = 'digit_model.pth'
torch.save(model.state_dict(), model_path)
print(f"✅ Modèle sauvegardé: {model_path}")

full_model_path = 'digit_model_full.pth'
torch.save(model, full_model_path)
print(f"✅ Modèle complet sauvegardé: {full_model_path}")
```

## Étape 2 : Exporter pour le web

Dans le terminal :

```bash
cd /Users/killianbarbarin/Desktop/IIM/IA-number-detector
source venv/bin/activate
python export_for_web.py
```

## Étape 3 : Tester le site

1. Le serveur HTTP est déjà lancé sur http://localhost:8000
2. Rafraîchissez la page dans votre navigateur (Cmd+R ou F5)
3. Dessinez un chiffre et cliquez sur "Prédire"

## Problème actuel

❌ Le fichier `model.onnx` actuel contient un modèle **non entraîné** (poids aléatoires)
✅ Une fois les poids sauvegardés et réexportés, le modèle aura 99.40% de précision!

## Fichiers attendus

Après l'étape 1, vous devriez avoir :
- `digit_model.pth` (poids uniquement, ~6 MB)
- `digit_model_full.pth` (architecture + poids, ~6 MB)

Après l'étape 2, vous devriez avoir :
- `model.onnx` (métadonnées, ~13 KB)
- `model.onnx.data` (poids, ~1.5 MB)
