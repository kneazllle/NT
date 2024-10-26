from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

scaler = joblib.load('models/scaler_model.joblib')
placement_model = joblib.load('models/placement_model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        inputs = [
            float(request.form['gender']),      # Gender (1 for Male, 0 for Female)
            float(request.form['ssc_p']),       # SSC Percentage
            float(request.form['ssc_b']),       # SSC Board (0 for Others, 1 for CBSE)
            float(request.form['hsc_p']),       # HSC Percentage
            float(request.form['hsc_b']),       # HSC Board (0 for Others, 1 for CBSE)
            float(request.form['hsc_s']),       # HSC Stream (1 for Science, 2 for Commerce)
            float(request.form['degree_p']),    # Degree Percentage
            float(request.form['degree_t']),    # Degree Type (0 for Others, 1 for B.Sc, 2 for B.Com)
            float(request.form['workex']),      # Work Experience (1 for Yes, 0 for No)
            float(request.form['etest_p']),     # Entrance Test Percentage
            float(request.form['mca_p'])        # MCA Percentage
        ]
        
        scaled_inputs = scaler.transform(np.array(inputs).reshape(1, -1))
        prediction = placement_model.predict(scaled_inputs)

        result = 'Placed' if prediction[0] == 1 else 'Not Placed'
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
