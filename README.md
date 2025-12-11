# 🤖 IA Number Detector - Détecteur de Chiffres Manuscrits

Un projet complet de reconnaissance de chiffres manuscrits utilisant PyTorch et ONNX Runtime Web. Ce projet démontre l'entraînement d'un réseau de neurones convolutif (CNN) sur le dataset MNIST et son déploiement dans une interface web interactive.

## ✨ Fonctionnalités

- 🧠 **Modèle CNN** entraîné sur le dataset MNIST (60 000 images d'entraînement)
- 📊 **Notebook Jupyter complet** avec visualisations et métriques détaillées
- 🌐 **Interface web interactive** pour dessiner et reconnaître des chiffres en temps réel
- ⚡ **Inférence rapide** dans le navigateur grâce à ONNX Runtime Web
- 📈 **Affichage des probabilités** pour chaque chiffre (0-9)
- 🎨 **Design moderne et responsive** avec animations fluides

## 🚀 Installation

### Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)
- Un navigateur web moderne

### Étape 1 : Cloner le projet

```bash
git clone <url-du-repo>
cd IA-number-detector
```

### Étape 2 : Installer les dépendances Python

```bash
pip install -r requirements.txt
```

Les dépendances incluent :
- `torch` - Framework de deep learning
- `torchvision` - Utilitaires pour la vision par ordinateur
- `onnx` - Format d'échange de modèles
- `onnxruntime` - Runtime pour l'inférence ONNX
- `numpy` - Calculs numériques
- `matplotlib` - Visualisations
- `jupyter` - Environnement de notebooks

## 📚 Utilisation

### 1. Entraîner le modèle

Ouvrez le notebook Jupyter et exécutez toutes les cellules :

```bash
jupyter notebook train_model.ipynb
```

Le notebook va :
1. ✅ Télécharger automatiquement le dataset MNIST
2. ✅ Entraîner un modèle CNN pendant 5 epochs
3. ✅ Afficher les courbes d'apprentissage
4. ✅ Exporter le modèle au format ONNX (`model.onnx`)
5. ✅ Générer des graphiques de résultats

**Résultats attendus :**
- Accuracy sur le test : ~98-99%
- Temps d'entraînement : 5-10 minutes (CPU) / 1-2 minutes (GPU)

### 2. Tester l'application web

Une fois le modèle exporté (`model.onnx` généré), ouvrez l'interface web :

```bash
# Lancez un serveur web local
python -m http.server 8000
```

Puis ouvrez votre navigateur à l'adresse : `http://localhost:8000`

**Utilisation de l'interface :**
1. ✍️ Dessinez un chiffre (0-9) sur le canvas blanc
2. 🔮 Cliquez sur "Prédire" pour lancer la reconnaissance
3. 📊 Consultez le résultat et les probabilités par chiffre
4. 🗑️ Cliquez sur "Effacer" pour réinitialiser

## 🏗️ Architecture du Projet

```
IA-number-detector/
├── train_model.ipynb          # Notebook d'entraînement
├── requirements.txt           # Dépendances Python
├── model.onnx                 # Modèle exporté (généré)
├── index.html                 # Interface web
├── style.css                  # Styles de l'interface
├── script.js                  # Logique de prédiction web
├── data/                      # Dataset MNIST (téléchargé auto)
├── training_results.png       # Graphiques d'entraînement (généré)
└── README.md                  # Ce fichier
```

## 🧠 Architecture du Modèle

Le modèle CNN est composé de :

**Couches convolutives :**
- Conv2D (1→32 filtres) + BatchNorm + ReLU + MaxPool
- Conv2D (32→64 filtres) + BatchNorm + ReLU + MaxPool
- Conv2D (64→128 filtres) + BatchNorm + ReLU + MaxPool

**Couches fully connected :**
- Linear (1152→256) + ReLU + Dropout(0.5)
- Linear (256→10) - Sortie

**Total des paramètres :** ~300 000 paramètres entraînables

## 📊 Performances

- **Accuracy sur le test :** ~98-99%
- **Loss finale :** ~0.03-0.05
- **Taille du modèle ONNX :** ~1.2 MB
- **Temps d'inférence (web) :** <100ms

## 🛠️ Technologies Utilisées

- **Backend/Entraînement :**
  - Python 3.x
  - PyTorch 2.0+
  - ONNX 1.14+
  - Jupyter Notebook

- **Frontend/Déploiement :**
  - HTML5 Canvas
  - CSS3 (Gradients, Animations)
  - JavaScript ES6
  - ONNX Runtime Web

## 📝 Améliorations Possibles

- [ ] Ajouter un mode d'augmentation de données
- [ ] Tester d'autres architectures (ResNet, VGG)
- [ ] Implémenter la détection de plusieurs chiffres
- [ ] Ajouter un mode de dessin avec différentes couleurs
- [ ] Créer une API REST avec Flask/FastAPI
- [ ] Déployer sur Heroku/Vercel/Netlify

## 🐛 Dépannage

**Le modèle ne se charge pas :**
- Vérifiez que le fichier `model.onnx` existe dans le répertoire
- Exécutez d'abord le notebook pour générer le modèle
- Consultez la console du navigateur pour les erreurs

**Les prédictions sont incorrectes :**
- Assurez-vous de dessiner des chiffres clairs et centrés
- Le modèle fonctionne mieux avec des traits épais
- Évitez de dessiner trop petit ou trop près des bords

**Erreurs d'installation :**
```bash
# Mettez à jour pip
pip install --upgrade pip

# Installez les dépendances une par une en cas d'erreur
pip install torch torchvision
pip install onnx onnxruntime
pip install numpy matplotlib jupyter
```

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 👨‍💻 Auteur

Projet créé dans le cadre d'un cours d'Intelligence Artificielle à l'IIM.

## 🙏 Remerciements

- Dataset MNIST : Yann LeCun et al.
- PyTorch : Meta AI
- ONNX : Microsoft, Facebook, AWS et autres contributeurs

---

**Note :** Ce projet est à but éducatif et démontre les bases de l'apprentissage profond et du déploiement de modèles IA.