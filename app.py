from flask import Flask, render_template, request
import pickle
import numpy as np


with open("rf.pkl",'rb') as model_file:
    model = pickle.load(model_file)


app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        s_length = request.form['s_length']
        s_width = request.form['s_width']
        p_length = request.form['p_length']
        p_width = request.form['p_width']

        prediction = model.predict(np.array([[s_length, s_width, p_length, p_width]]))
        #return str(prediction) 
        if prediction == 0:
            iris_type = 'Iris-Setosa'
        elif prediction == 1:
            iris_type = 'Iris-Versicolor'
        else:
            iris_type = 'Iris-Verginica'
            
        return render_template('index.html',prediction=iris_type)
   # return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)