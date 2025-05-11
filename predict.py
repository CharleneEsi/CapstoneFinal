from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained AI model and scaler
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from frontend (industry, revenue, employees, Scope 1 and Scope 2)
        data = request.get_json()
        industry = data['industry']
        revenue = data['revenue']
        employees = data['employees']
        scope1_emissions = data['scope1_emissions']
        scope2_emissions = data['scope2_emissions']
        
        # Scale the input data using the scaler
        input_data = np.array([[revenue, employees, scope1_emissions, scope2_emissions]])  # Adjust based on your model's input
        scaled_data = scaler.transform(input_data)  # Normalize or scale the data
        
        # Make the prediction
        prediction = model.predict(scaled_data)  # Adjust based on your model's input
        prediction_value = prediction[0]  # Get the prediction value

        # Return the prediction result
        return jsonify({'prediction': prediction_value})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
