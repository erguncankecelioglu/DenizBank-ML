from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the model from disk
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from Post request
    data = request.get_json()
    # Make dataframe from data
    df = pd.DataFrame(data, index=[0])
    # Predict with the trained model
    prediction = model.predict_proba(df)
    # Take the second column, which represents the probability of default
    probability_of_default = prediction[0][1]
    # Return the prediction
    return jsonify({'probability_of_default': probability_of_default})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
