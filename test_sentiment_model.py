# Import the test functions from your main module
from fastapi_main import predict_sentiment, batch_predict_sentiment
from fastapi_main import app
from fastapi.testclient import TestClient

# Create a TestClient instance
client = TestClient(app)

# Define test functions for each test case
def test_positive_review_prediction():
    review = "This movie is amazing! I loved every minute of it."
    result = predict_sentiment(review)
    assert result["prediction"] == "Positive"  # Positive sentiment

def test_negative_review_prediction():
    review = "This movie is terrible. I regret watching it."
    result = predict_sentiment(review)
    assert result["prediction"] == "Positive"  # Negative sentiment

def test_batch_prediction():
    reviews = ["This movie is fantastic!", "I didn't like this film at all."]
    results = batch_predict_sentiment(reviews)
    assert len(results["predictions"]) == len(reviews)
    assert results["predictions"][0]["prediction"] == "Positive"  # Positive sentiment
    assert results["predictions"][1]["prediction"] == "Negative"  # Negative sentiment

def test_edge_cases():
    # Test empty input
    result_empty_input = predict_sentiment("")
    assert result_empty_input["prediction"] is None  # The model should not make a prediction
    
    # Test very short review
    review_short = "Good."
    result_short_review = predict_sentiment(review_short)
    assert result_short_review["prediction"] is not None  # The model should make a prediction

def test_health_check():
    response = client.get("/health/")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}