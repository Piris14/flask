
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
import joblib
import os

# Cargar el conjunto de datos Iris
data = load_iris()
X = data.data
y = data.target

# Crear el modelo de árbol de decisión
model = DecisionTreeClassifier(random_state=42)

# Entrenar el modelo
model.fit(X, y)

# Guardar el modelo entrenado en la misma carpeta (src)
model_path = os.path.join(os.path.dirname(__file__), 'decision_tree_classifier_default_42.sav')
joblib.dump(model, model_path)

print("Modelo entrenado y guardado en:", model_path)
