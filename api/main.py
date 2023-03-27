from flask import Flask, request, jsonify
from pydantic import BaseModel
import pandas as pd
import joblib

app = Flask(__name__)


class RequestFeatures(BaseModel):
    bedroom: float
    layout_type: int
    locality: int
    area: float
    furnish_type: int
    bathroom: float
    city: int
    agent: int
    builder: int
    owner: int
    apartment: int
    independent_floor: int
    independent_house: int
    penthouse: int
    studio_apartment: int
    villa: int


with open('../model.joblib', 'rb') as f:
    model = joblib.load(f)


@app.route('/', methods=['POST'])
def predict():
    req = RequestFeatures(**request.json)
    df = pd.DataFrame([req.dict()], columns=req.dict().keys())
    try:
        res = model.predict([list(req.dict().values())])
        return jsonify({"prediction": res[0]})
    except Exception as e:
        return jsonify({"error": str(e)})
