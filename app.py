# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 14:35:27 2020

@author: bibek
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
pickle_wb = open('model.pickle','rb')
model = pickle.load(pickle_wb)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
            clump_thickness= int(request.form['clump_thickness'])
            unif_cell_size = int(request.form['unif_cell_size'])
            unif_cell_shape = int(request.form['unif_cell_shape'])
            marge_adhesion= int(request.form['marge_adhesion'])
            single_epith_cell_size = int(request.form['single_epith_cell_size'])
            bare_nuclei = int(request.form['bare_nuclei'])
            bland_chrom = int(request.form['bland_chrom'])
            norm_nucleoli = int(request.form['norm_nucleoli'])
            mitoses = int(request.form['mitoses'])
            data = np.array([clump_thickness,unif_cell_size,unif_cell_shape,marge_adhesion,single_epith_cell_size,bare_nuclei,bland_chrom,norm_nucleoli,mitoses])
            data = data.reshape(1,-1)
            prediction = model.predict(data)
           
            if prediction[0] == 2:
               result = 'Breast Cancer is Benign'
            else:
               result= 'Breast Cancer is Malignant'
            return render_template('index.html', prediction_text=result)


if __name__ == "__main__":
    app.run(debug=True)