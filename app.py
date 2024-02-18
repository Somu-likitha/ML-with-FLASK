from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the svm model
sc=pickle.load(open('scaling.pkl','rb'))
model= pickle.load(open('classifier.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    feat=[float(x) for x in request.form.values()]
    ffeat=[np.array(feat)]
    pred=model.predict(sc.transform(ffeat))        
    return render_template('result.html',prediction= pred)

if __name__ == '__main__':
	app.run(debug=True)