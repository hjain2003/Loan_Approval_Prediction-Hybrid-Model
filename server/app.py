from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the model and scaler
model = joblib.load('loan_approval_model.joblib')
scaler = joblib.load('loan_approval_scaler.joblib')

@app.route('/')
def home():
    return "Welcome to Loan Approval API"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = [data['no_of_dependents'], data['education'], data['self_employed'], data['income_annum'],
                    data['loan_amount'], data['loan_term'], data['cibil_score'], data['residential_assets_value'],
                    data['commercial_assets_value'], data['luxury_assets_value'], data['bank_asset_value']]

        features[1] = 1 if features[1] == ' Graduate' else 0 #yes space is there before the word kyuki csv file fucked
        features[2] = 1 if features[2] == ' Yes' else 0  #same as above
        features = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)

        return jsonify({'prediction': (prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
