import flask
from flask import Flask

app = Flask(__name__)


@app.route("/", methods=["POST", "GET"])
def hello_world():
    if flask.request.method == "POST":
        sdg_text = flask.request.form["sdg_text"]
        print(sdg_text)
    return flask.render_template("index.html")
