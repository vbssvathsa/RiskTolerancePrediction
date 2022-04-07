import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('final_risk_tolerance_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    #return render_template('index.html', prediction_text = 'Risk Tolerance level: {}'.format(features))
    prediction = model.predict(np.array(features).reshape(1,-1))
    
    output = round(prediction[0],4)
    
    return render_template('index.html', prediction_text = 'Risk Tolerance level: {}'.format(output*100))

if __name__ == '__main__':
    app.run(debug=True)