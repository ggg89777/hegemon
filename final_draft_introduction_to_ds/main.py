import dill as dill
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()
with open("model/sber_auto_pipe.pkl", 'rb') as file:
    model = dill.load(file)


class Form(BaseModel):
    client_id: float
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_adcontent: str
    utm_keyword: str
    device_category: str
    device_os: str
    device_brand: str
    device_screen_resolution: str
    device_browser: str
    geo_country: str
    geo_city: str



class Prediction(BaseModel):
    client_id: float
    pred: int


@app.get("/status")
def status():
    return "I’m OK"


@app.get("/version")
def version():
    return model["metadata"]


@app.post("/predict", response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame(form.dict(), index=[0])
    prob = model["model"].predict_proba(df)
    y = prob[:,1].copy()
    if y > 0.01:
        x = 1
    else:
        x = 0

    return {
        "client_id": form.client_id,
        "pred": x
    }


def main():
    with open("model/sber_auto_pipe.pkl", 'rb') as file:
        model = dill.load(file)
        print(model["metadata"])