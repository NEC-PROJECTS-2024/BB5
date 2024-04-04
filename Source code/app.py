import pickle
import numpy as np
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Replace the following line with the actual number of features your model expects
expected_number_of_features = 8

@app.route('/')
def home():
    return render_template('index.html')

def is_positive_number(x):
    try:
        num = int(x)
        if num >= 0:
            return True
        else:
            return False
    except ValueError:
        return False

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from the form
        features = [x.strip() for x in request.form.values()]
        print(features)

        # Ensure that the features have the correct length and format
        if len(features) == expected_number_of_features and all(is_positive_number(x) for x in features):
            features = [int(x) for x in features]
            final_features = np.array(features).reshape(1, -1)

            # Use the loaded model to make predictions
            prediction = model.predict(final_features)

            # Round the prediction if needed
            output = round(prediction[0], 1)

            return render_template('index.html', prediction_text=output)

        else:
            return render_template('index.html', prediction_text='Error: Invalid input. Please enter correct input only.')

    except Exception as e:
        return render_template('index.html', prediction_text='Error: {}'.format(str(e)))

if __name__ == "__main__":
    model = joblib.load(r'C:/Users/addan/OneDrive/Desktop/4-2/restaurant/mymodel.pkl')
    app.run(host="0.0.0.0", port=5000, debug=True)
