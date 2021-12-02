import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('wine_svc.pkl','rb'))

@app.route('/')
def home():
    return render_template('white_wine.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    for renduring results on HTML GUI
    '''
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    output = round(prediction[0],1)

#convert binary result into low or high quality
    if output == 0:
        output = 'Low Quality Wine'
    else:
        output = 'High Quality Wine'

    return render_template('white_wine.html',prediction_text='Result: {}'.format(output))
    

if __name__ == "__main__":
    app.run(debug=True)