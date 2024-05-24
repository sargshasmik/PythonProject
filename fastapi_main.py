import re
import os
import joblib
import uvicorn
from fastapi import FastAPI, Path
from urllib.parse import unquote

app = FastAPI(
    title="Sentiment Model API",
    description="A simple API that uses an NLP model to predict the sentiment of movie reviews.",
    version="0.1",
)

# Greeting endpoint
@app.get("/greet/{name}", tags=["Greeting"])
def greet_user(name: str):
    """Greet a user."""
    return {"message": f"Hello, {name}!"}

# Endpoint for health check
@app.get("/health/", tags = ["Status"])
async def health_check():
    """Check the health status of the API."""
    return {"status": "OK"}

# Load the sentiment model
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "lrmodel.pkl")
with open(model_path, "rb") as f:
    pipeLine = joblib.load(f)

# Define a function for text preprocessing
def preprocessText(text):
    text = text.lower()
    #removes HTML tags
    text = re.sub(r'<[^<>]+>', ' ', text)
    #removing all special characters and numbers
    text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
    text = re.sub(r'[[]]', ' ', text)
    return text

# Endpoint to get the sentiment model details
@app.get("/model-info", tags=["Model"])
def get_model_info():
    """Get information about the sentiment analysis model."""
    return {"model_name": "Logistic Regression", "model_type": "Supervised Learning"}

# Endpoint for predicting sentiment
@app.get("/predict-review")
def predict_sentiment(review: str):
    """
    A simple function that receive a review content and predict the sentiment of the content.
    :param review:
    :return: prediction, probabilities
    """
    # clean the review
    cleaned_review = preprocessText(review)

    if not cleaned_review:
        return {"prediction": None, "probability": None}
    # perform prediction
    prediction = pipeLine.predict([cleaned_review])
    output = int(prediction[0])
    probas = pipeLine.predict_proba([cleaned_review])
    output_probability = "{:.2f}".format(float(probas[:, output]))
    
    # output dictionary
    sentiments = {0: "Negative", 1: "Positive"}
    
    # show results
    result = {"prediction": sentiments[output], "Probability": output_probability}
    return result


# Endpoint to analyze a batch of reviews
@app.post("/batch-predict", tags=["Batch Prediction"])
def batch_predict_sentiment(reviews: list[str]):
    """
    Predict the sentiment of multiple reviews.
    :param reviews: List of review texts.
    :return: List of predictions and probabilities.
    """
    predictions = []
    for review in reviews:
        result = predict_sentiment(review)
        predictions.append(result)
    return {"predictions": predictions}



# Welcome message and general information endpoint
@app.get("/", response_model=dict)
async def read_root():
    """Welcome message and general information about the API."""
    return {"message": "Welcome to the Sentiment Review API"}

if __name__ == "__main__":
    uvicorn.run("fastapi_main:app", host="0.0.0.0", port=8000, reload=True)