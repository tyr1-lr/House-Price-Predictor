from website import app
from flask import render_template, request
import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from flask import render_template, request, send_file
import os

model = joblib.load("model/house_price_model.pkl")


@app.route("/")
def home():
    return render_template('home.html')


@app.route("/predict_house", methods=["GET", "POST"])
def predict_house():

    prediction = None

    if request.method == "POST":

        MedInc = float(request.form["MedInc"])
        HouseAge = float(request.form["HouseAge"])
        AveRooms = float(request.form["AveRooms"])
        AveBedrms = float(request.form["AveBedrms"])
        Population = float(request.form["Population"])
        AveOccup = float(request.form["AveOccup"])
        Latitude = float(request.form["Latitude"])
        Longitude = float(request.form["Longitude"])

        features = np.array([[
            MedInc, HouseAge, AveRooms, AveBedrms,
            Population, AveOccup, Latitude, Longitude
        ]])

        prediction = model.predict(features)
        prediction = prediction[0] * 100000
        prediction = round(prediction, 2)

    return render_template(
        "house_predict.html",
        prediction=prediction
    )


@app.route("/notebook")
def notebook():
    return render_template("notebook.html")


@app.route("/download_notebook")
def download_notebook():

    base_dir = os.path.dirname(os.path.dirname(__file__))  # go out of /website

    file_path = os.path.join(base_dir, "notebooks", "house_pricing.ipynb")

    return send_file(file_path, as_attachment=True)


@app.route("/performance")
def performance():

    model = joblib.load("model/house_price_model.pkl")

    data = fetch_california_housing()
    X = data.data
    y = data.target

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    return render_template(
        "performance.html",
        r2=round(r2, 3),
        mse=round(mse, 3),
        dataset_size=len(X_test)
    )


@app.route("/dataset")
def dataset():
    from sklearn.datasets import fetch_california_housing

    data = fetch_california_housing(as_frame=True)
    df = data.frame  # full dataset

    table = df.head(100).to_html(classes="table-auto w-full text-sm")

    return render_template("dataset.html", table=table)


@app.route("/about")
def about():
    return render_template("about.html")
