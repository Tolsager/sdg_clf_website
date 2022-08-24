import json

import requests
from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/", methods=["POST", "GET"])
def home():
    prediction = None
    if request.method == "POST":
        sdg_text = request.form["sdg_text"]
        # url = "https://sdg-clf-api-free.azurewebsites.net/predict"
        url = "https://sdg-clf-api.azurewebsites.net/predict"
        post_obj = {"text": sdg_text}
        prediction = requests.post(url, json=post_obj)
        prediction = json.loads(prediction.content)["prediction"]
    return render_template("home.html", prediction=prediction)


@app.route("/about")
def about():
    return render_template("about.html")
