import numpy as np
from flask import Flask, request, render_template
import joblib
import requests

app = Flask(__name__)

# Load trained local model
model = joblib.load('power_prediction.sav')


# ---------------- HOME PAGE ----------------
@app.route('/')
def home():
    return render_template('intro.html')


# ---------------- PREDICTION PAGE ----------------
@app.route('/predict')
def predict():
    return render_template('predict.html')


# ---------------- WEATHER API (Optional) ----------------
@app.route('/windapi', methods=['POST'])
def windapi():
    city = request.form.get('city')

    apikey = "efb8067a0f1bf425dc13bb6f4864e083"  # Replace with your API key
    url = "http://api.openweathermap.org/data/2.5/weather?q=" + city + "&appid=" + apikey

    resp = requests.get(url).json()

    temp = str((resp["main"]["temp"] - 273.15)) + " °C"
    humid = str(resp["main"]["humidity"]) + " %"
    pressure = str(resp["main"]["pressure"]) + " hPa"
    speed = str((resp["wind"]["speed"]) * 3.6) + " Km/h"

    return render_template('predict.html',
                           temp=temp,
                           humid=humid,
                           pressure=pressure,
                           speed=speed)


# ---------------- MODEL PREDICTION ----------------
@app.route('/y_predict', methods=['POST'])
def y_predict():
    '''
    Take user input and predict wind energy output
    '''

    # Convert input values to float
    x_test = [[float(x) for x in request.form.values()]]

    print("Input Received:", x_test)

    # Make prediction using local model
    prediction = model.predict(x_test)

    output = prediction[0]

    return render_template('predict.html',
                           prediction_text='The energy predicted is {:.2f} KWh'.format(output))


# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(debug=True)
