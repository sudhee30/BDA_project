from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
MODEL_PATH = "model (1).pkl"
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from form inputs
        feature1 = float(request.form['feature1'])
        feature2 = float(request.form['feature2'])
        feature3 = float(request.form['feature3'])
        
        # Add more features if needed
        features = np.array([feature1, feature2, feature3]).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)
        return render_template('index.html', prediction=prediction[0])
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
