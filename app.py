from re import template
from urllib import request
from flask import Flask,request, jsonify,render_template
import numpy as np
import pickle

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():    
    data=request.form
    print(data)
    print("*"*56)
    CRIM =float( data['CRIM'])
    ZN =float( data['ZN'])
    INDUS = float(data['INDUS'])
    CHAS =float( data['CHAS'])
    NOX = float(data['NOX'])
    RM = float(data['RM'])
    AGE = float(data['AGE'])
    DIS = float(data['DIS'])
    RAD = float(data['RAD'])
    TAX = float(data['TAX'])
    PTRATIO= float(data['PTRATIO'])
    B =float(data['B'])
    LSTAT = float(data['LSTAT'])

    array=np.array([CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS,RAD, TAX, PTRATIO, B, LSTAT],ndmin=2)
    print(array)
    with open('model.pkl','rb') as file:
        model= pickle.load(file)
    result= np.around(model.predict(array),2)[0]
    print(result)




    return render_template('index.html', pred = result)


if __name__=="__main__":
    app.run(debug=True)