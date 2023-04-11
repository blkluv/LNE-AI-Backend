from flask import Flask, request, jsonify
from pydantic import BaseModel
import pandas as pd
import joblib
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


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


with open('model.joblib', 'rb') as f:
    model = joblib.load(f)


@app.route('/', methods=['POST'])
@cross_origin()
def predict():
    req = RequestFeatures(**request.json)
    df = pd.DataFrame([req.dict()], columns=req.dict().keys())
    try:
        res = model.predict([list(req.dict().values())])
        return jsonify({"prediction": res[0]})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.after_request
def add_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response
