import os

import fastapi
import psutil
import pydantic
from flask import request

from sdg_clf import modelling, utils, inference

os.environ["OMP_NUM_THREADS"] = f"{psutil.cpu_count()}"
os.environ["OMP_WAIT_POLICY"] = "ACTIVE"

onnx_model = modelling.create_model_for_provider("sdg_clf/finetuned_models/roberta-large_1608124504.onnx")
tokenizer = utils.get_tokenizer("roberta-large")


app = fastapi.FastAPI()

class SDGText(pydantic.BaseModel):
    text: str


@app.post("/predict")
def predict(SDGText: SDGText):
    prediction = inference.predict_on_sample(SDGText.text, onnx_model, tokenizer)
    return {"prediction": prediction}


@app.get("/")
def index():
    return "Server is running"
