import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__,template_folder='template')
s,c = pickle.load(open('diabetes.pkl', 'rb'))   

@app.route('/')
def home():
    return render_template('diabetes.html')    

@app.route('/predict',methods=['POST'])
def predict():                            
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]    
    input_data_as_np_array = np.array(int_features)

    inp_reshape = input_data_as_np_array.reshape(1,-1)

    test_standardize = s.transform(inp_reshape)

    prediction = c.predict(test_standardize)
    
    output = round(prediction[0], 2)
    if(prediction[0] == 0):
        op = "Not Diabetic"
    else:
        op = "Diabetic"
    return render_template('diabetes.html', prediction_text='{}'.format(op)) 



@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run('localhost',debug=True)