# TP1 – Classification MNIST (Réseau Fully-Connected)

Ce projet entraîne un réseau de neurones fully-connected (MLP) simple avec **TensorFlow / Keras** pour classifier les chiffres manuscrits du dataset **MNIST** (28x28 -> 10 classes).


## 1. Installation rapide
Créer (optionnel) un environnement virtuel puis installer TensorFlow :
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install tensorflow numpy
```

## 2. Exécution
```bash
python exercice1.py
```
Sortie attendue (exemple) :
```
Epoch 1/5
... (logs Keras) ...
Precision sur les donnees de test: 0.97xx
Modele sauvegarde sous mnist_model.h5
```

## 3. Explication des couches principales
### Dense
Couche entièrement connectée : effectue \( Y = XW + b \). La première couche apprend des représentations abstraites à partir des 784 pixels. La dernière génère une distribution de probabilité sur les 10 classes via `softmax`.

### Dropout (0.2)
Met aléatoirement à zéro 20% des unités pendant l'entraînement pour **réduire le sur-apprentissage**. Désactivé en phase d'inférence.


## 4. Structure mémoire des tenseurs
| Étape | Forme |
|-------|-------|
| Brut | (60000, 28, 28) |
| Aplati | (60000, 784) |
| Batch (ex.) | (128, 784) |
| Dense 512 | (128, 512) |
| Sortie | (128, 10) |

## 5. Résultats typiques
Précision test attendue après 5 époques : ~97–98%. (Peut varier selon hardware / seed.)

## 6. Fichier sauvegardé
`mnist_model.h5` : contient architecture + poids (format Keras HDF5). Pour recharger :
```python
from tensorflow import keras
model = keras.models.load_model("mnist_model.h5")
```

