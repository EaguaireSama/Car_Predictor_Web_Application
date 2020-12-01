from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = pickle.load(open('final_model_car_detection.pkl', 'rb'))


@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    Fuel_Type_Diesel = 0
    if request.method == 'POST':
        Year = int(request.form['year'])
        Driven = int(request.form['driven'])
        Seats = int(request.form['seats'])
        Mileage = float(request.form['mil'])
        CC = float(request.form['eng'])
        Power = float(request.form['pow'])
        FuelType = int(request.form['fuel'])
        Seller = int(request.form['seller'])
        Trans = int(request.form['trans'])
        Owner = int(request.form['own'])
        Company = int(request.form['comp'])
        l = [Year,Driven,Seats,Mileage,CC,Power,0,0,0,0,0,0,0,0,0,Owner,0,0,0]
        l[FuelType] = 1
        l[Seller] = 1
        l[Trans] = 1
        l[Company] = 1
        print(l)
        prediction = model.predict([l])
        return render_template('index.html', result=prediction)
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
