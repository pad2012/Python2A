
# udostępnianie modelu jako usługi
from flask import Flask, request
import joblib
import sklearn
import numpy as np

# inicjalizacja Flaska
app = Flask("KNN")
# wczytanie modelu
model = joblib.load("iris.model")


# http://10.0.2.15:1234/predict?sl=5.6&sw=3.2&pl=5.2&pw=1.45
@app.route("/predict")
def predict():
    try:
        sl = float(request.args.get("sl"))
        sw = float(request.args.get("sw"))
        pl = float(request.args.get("pl"))
        pw = float(request.args.get("pw"))
        if sl<=0 or sw<=0 or pl<=0 or pw<=0:
            raise Exception("Sprawdź wartości!")
        result = model.predict( np.array([ [sl,sw,pl,pw] ]) )
        iris = ['setosa', 'versicolor', 'virginica']

        return iris[ result[0] ]
    except Exception as exc:
        return str(exc)

@app.route("/")
def hello():
    return "<h1>Hello KNN</h1>"

app.run(debug=True, port=1234, host="0.0.0.0")