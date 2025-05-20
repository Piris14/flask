from flask import Flask, request, render_template
from joblib import load  # Importamos joblib para cargar el modelo
import os
import pandas as pd  

app = Flask(__name__)

# Cargar el modelo
model_path = os.path.join(os.path.dirname(__file__), "decision_tree_classifier_default_42.sav")
model = load(model_path)  # Usamos joblib.load para cargar el modelo

# Diccionario de clases
class_dict = {
    "0": "Iris setosa",
    "1": "Iris versicolor",
    "2": "Iris virginica"
}

@app.route("/", methods=["GET", "POST"])
def index():
    pred_class = None

    if request.method == "POST":
        # Obtener valores del formulario
        val1 = float(request.form["val1"])  # petal width
        val2 = float(request.form["val2"])  # petal length
        val3 = float(request.form["val3"])  # sepal width
        val4 = float(request.form["val4"])  # sepal length

        # Reordenar los valores para coincidir con los nombres de columnas del modelo
        data = pd.DataFrame([[val4, val3, val2, val1]], columns=[
            "sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"
        ])

        # Predecir
        prediction = str(model.predict(data)[0])
        pred_class = class_dict[prediction]

    return render_template("index.html", prediction=pred_class)

if __name__ == "__main__":
    app.run(debug=True)



