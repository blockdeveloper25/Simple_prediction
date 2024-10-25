from flask import Flask, request, render_template
import pickle

import numpy as np

app = Flask(__name__)


#Load the trained Model
with open('linear_regression_model.pkl','rb') as file:
    model = pickle.load(file)
    
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    #Get input Data from the Form
    mid_sem_marks = request.form['ME']
    
    #make a prediction with loaded model
    input_data = [[float(mid_sem_marks)]]
    reshaped_data = np.array(input_data).reshape(1,-1)
    prediction = model.predict(reshaped_data)
    
    return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)