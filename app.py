from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load("linear_regression_model.joblib")

# Initialize Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    # Get JSON data from the request
    data = request.get_json()
    
    # Extract input value
    median_income = data["median_income"]
    
    # Convert to a 2D array (since scikit-learn expects it)
    input_data = np.array([[median_income]])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Return the prediction as JSON
    return jsonify({"predicted_price": prediction[0]})

if __name__ == "__main__":
    app.run(debug=True)
