import numpy as np
import pickle

class hpp():
    def __init__(self,data):
        self.data = data

    def load_model(self):
        with open(r'artifacts/model.pkl','rb') as file:
            self.model = pickle.load(file)

    def predict(self):
        self.load_model()

        CRIM = float(self.data['CRIM'])
        ZN = float(self.data['ZN'])
        INDUS=float(self.data['INDUS'])
        CHAS=float(self.data['CHAS'])
        NOX=float(self.data['NOX'])
        RM=float(self.data['RM'])
        AGE=float(self.data['AGE'])
        DIS=float(self.data['DIS'])
        RAD=float(self.data['RAD'])
        TAX=float(self.data['TAX'])
        PTRATIO=float(self.data['PTRATIO'])
        B=float(self.data['B'])
        LSTAT=float(self.data['LSTAT'])
        array = np.array ([CRIM,ZN,INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX,PTRATIO, B, LSTAT], ndmin = 2)
        print(array)
        print("*"*50)

        res = np.around(self.model.predict(array),2)[0]
        print(res)

        return res




if __name__ == "__main__": 
    data = {
        'CRIM' : 0.34,
        'ZN' : 0.01,
        'INDUS':15.89000,
        'CHAS': 2.2000,
        'NOX':0.55000,
        'RM':6.95100,
        'AGE':86.80000,
        'DIS':2.98930,
        'RAD':4.00000,
        'TAX':230.00000,
        'PTRATIO':11.40000,
        'B':350.90000,
        'LSTAT': 16.92000

    }


    hpp_obj = hpp(data)

    hpp_obj.predict()