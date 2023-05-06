import pickle
from flask import Flask, jsonify, request

app = Flask(__name__)

# Load the trained model from the saved file
with open('auto_arima_model.pkl', 'rb') as f:
    auto_arima_model = pickle.load(f)

# Define a route to handle incoming prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    # Parse the request body as JSON
    req_body = request.get_json()

    # Extract the input data from the request body
    input_data = req_body['input_data']

    # Make predictions using the loaded model
    predictions = auto_arima_model.predict(n_periods=len(input_data))

    # Return the predictions as a JSON response
    res_body = {'predictions': predictions.tolist()}
    return jsonify(res_body)

# Start the Flask app on port 5000
if __name__ == '__main__':
    app.run(port=5000, debug=True)
