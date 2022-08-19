import os

import psutil
from flask import Flask, render_template, request

from sdg_clf import modelling, utils, inference

app = Flask(__name__)

os.environ["OMP_NUM_THREADS"] = f"{psutil.cpu_count()}"
os.environ["OMP_WAIT_POLICY"] = "ACTIVE"

onnx_model = modelling.create_model_for_provider("sdg_clf/finetuned_models/roberta-large_1608124504.onnx")
tokenizer = utils.get_tokenizer("roberta-large")


@app.route("/", methods=["POST", "GET"])
def home():
    prediction = None
    if request.method == "POST":
        prediction = inference.predict_on_sample(request.form["sdg_text"], onnx_model, tokenizer)
    return render_template("home.html", prediction=prediction)


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)
