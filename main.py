from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import numpy as np
import pandas as pd
import joblib
from helper import prepare_inference_input

app = FastAPI()

# Load the pre-trained model and other assets
model = joblib.load("random_forest_model.pkl")


def convert_to_required_format(response):
    scoreTable = []
    for feature, bins in response.items():
        feature_obj = {
            "id": feature,
            "title": feature.replace("_", " ").title(),
            "values": [],
        }
        for bin, score in bins.items():
            value_obj = {"value": bin.replace("_", " ").title(), "score": score}
            feature_obj["values"].append(value_obj)
        scoreTable.append(feature_obj)
    return scoreTable


with open("training_columns.json", "r") as file:
    training_columns = json.load(file)

# Load the bins information from the provided JSON file
with open("bins.json", "r") as f:
    features_bins = json.load(f)


# Define a model input class
class Description(BaseModel):
    text: str


# API endpoint for model prediction
@app.post("/predict/")
async def predict(description: Description):
    # Extract the text from the request
    text = description.text

    # Prepare the input data for the model using the predefined function
    # This function will use the text to extract embeddings and format the DataFrame
    # Note: Ensure `training_columns` are defined correctly or imported
    input_df = prepare_inference_input(text, training_columns)

    # Make predictions using the model
    # Assume `model` is a global variable or imported from a module
    predictions = model.predict(input_df)

    # Convert predictions to a nested JSON format
    response = {}
    index = 0
    for feature, bins in features_bins.items():
        response[feature] = {
            bin: float(predictions[index + i]) for i, bin in enumerate(bins)
        }
        index += len(bins)

    # Convert response to the required format
    formatted_response = convert_to_required_format(response)

    return formatted_response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
