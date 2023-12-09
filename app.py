# Import the Flask class from the flask module
from flask import Flask, render_template, request, jsonify
# from utils import make_prediction, NumpyEncoder
from flask_cors import CORS
import os
import json
import joblib
import tensorflow as tf
import pandas as pd
import numpy as np
import json
from os import path
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

os.environ.get('KEY')

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})


# Register a route
@app.route("/")
def home():
    return render_template("index.html")

class AIModel():

    # This class will handle the model file and the csv files.
    def __init__(self):
        #load the model
        self.model = tf.keras.models.load_model('models/model.h5', compile=False)
        self.ohe = joblib.load("models/ohe.pkl")
        self.scaler = joblib.load("models/scaler.pkl")

        # Load data
        self.deploy_folder = r'data'

        self.X_test_df = pd.read_csv(path.join(self.deploy_folder,"X_test_data.csv"))
        self.y_test_df = pd.read_csv(path.join(self.deploy_folder,"y_test_data.csv")).to_numpy()


@app.route("/api/predict", methods=["POST"])
def predict():
    
    # AI Model object
    nn_model = AIModel()
    
    # Get values from the POST Json Data\
    input_data = request.json
    
    input_df = pd.DataFrame([input_data])
    new_input_numeric = nn_model.scaler.transform(input_df.iloc[:,0:3])
    new_input_ohe = nn_model.ohe.transform(input_df.iloc[:,3:])
    new_input = np.concatenate((new_input_numeric, new_input_ohe), axis=1)
    
    # Make a prediction
    prediction = nn_model.model.predict(new_input)

    # Since it's a binary classification, you can convert this probability to a class label
    predicted_class = (prediction > 0.5).astype(int)

    prediction_result = "Student will not Persist"
    if predicted_class[0][0] == 1:
        prediction_result = "Student will Persist"
    
    return json.dumps({'output': prediction_result})


@app.route("/api/summary", methods=['GET'])
def summary():
    
    # AI Model object
    nn_model = AIModel()
    
    # Empty list
    string_list = []

    # Get Summary of the AI model (Tensorflow, Keras) and fill the list string_list LIST
    nn_model.model.summary(line_length=42, print_fn=lambda x: string_list.append(x))

    # Transform the list into string variable
    summary_json = "\n".join(string_list)

    return jsonify({'output':summary_json})


@app.route("/api/scores", methods=['GET']) #use decorator pattern for the route
#@cross_origin()
def scores():
    # AI Model object
    nn_model = AIModel()

    # Predict the Test data
    y_pred = nn_model.model.predict(nn_model.X_test_df)

    # Convert probabilities to class labels
    final_predict_labeled = [1 if x > 0.5 else 0 for x in y_pred]

    # get score and stats of the model compared the Test Data with the predict values
    accuracy = round(accuracy_score(nn_model.y_test_df, final_predict_labeled) * 100, 3)
    precision = round(precision_score(nn_model.y_test_df, final_predict_labeled) * 100, 3)
    recall = round(recall_score(nn_model.y_test_df, final_predict_labeled) * 100, 3)
    f1 = round(f1_score(nn_model.y_test_df, final_predict_labeled) * 100, 3)
    roc_auc = round(roc_auc_score(nn_model.y_test_df, final_predict_labeled) * 100, 3)
    conf_matrix = confusion_matrix(nn_model.y_test_df, final_predict_labeled)         

    # Json Response with all stats/scores
    resJson = {
        "accuracy": accuracy,
        "precision": precision,
        "recall":recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confussion_matrix": str(conf_matrix),
    }

    return jsonify({'output':resJson})

@app.route('/get_random_data')
def get_csv_data():
    
    nn_model = AIModel()
    
    # Just reshape it directly
    y_test_reshaped = nn_model.y_test_df.reshape(92,1)
    
    test_set = np.concatenate((nn_model.scaler.inverse_transform(nn_model.X_test_df.iloc[:,0:3]), 
                               nn_model.ohe.inverse_transform(nn_model.X_test_df.iloc[:,3:]), 
                               y_test_reshaped), axis=1)
    
    random_index = np.random.randint(0, test_set.shape[0])
    
    # Convert NumPy array to list before jsonify
    random_row = test_set[random_index].tolist()
    
    return random_row

# Run the Flask application
if __name__ == '__main__':
    port = 5000
    app.run(host='0.0.0.0', debug=True)
    #app.run(host='127.0.0.1', port=port, debug=True)