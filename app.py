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
    """ 
    AIModel Model: This class will open the model file, and the csv files.
    """

    def __init__(self):
        #TODO load the model
        self.model = tf.keras.models.load_model('models/model.h5', compile=False)
        self.ohe = joblib.load("models/ohe.pkl")
        self.scaler = joblib.load("models/scaler.pkl")

        # Load data
        self.deploy_folder = r'data'

        self.X_test_df = pd.read_csv(path.join(self.deploy_folder,"X_test_data.csv"))
        self.y_test_df = pd.read_csv(path.join(self.deploy_folder,"y_test_data.csv")).to_numpy()

        #self.label_dict = {'FirstYearPersistence_no': 0, 'FirstYearPersistence_yes': 1}
        #self.class_names = ['FirstTermGpa', 'SecondTermGpa', 'FirstLanguage', 'Funding', 'School', 'FastTrack', 'Coop', 'Residency', 'Gender', 'PreviousEducation', 'AgeGroup', 'HighSchoolAverageMark', 'MathScore', 'EnglishGrade']


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
    nn_model.model.summary(line_length=80, print_fn=lambda x: string_list.append(x))

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

    # Label predict values
    #final_predict_labeled = np.argmax(y_pred, axis=1)
    
    # Convert probabilities to class labels
    final_predict_labeled = [1 if x > 0.5 else 0 for x in y_pred]

    # Scores / Stats of the model compared the Test Data with the predict values
    accuracy = (accuracy_score(nn_model.y_test_df, final_predict_labeled) * 100)
    precision = (precision_score(nn_model.y_test_df, final_predict_labeled) * 100)
    recall = (recall_score(nn_model.y_test_df, final_predict_labeled) * 100)
    f1 = (f1_score(nn_model.y_test_df, final_predict_labeled) * 100)
    roc_auc = (roc_auc_score(nn_model.y_test_df, final_predict_labeled) * 100)
    conf_matrix = confusion_matrix(nn_model.y_test_df, final_predict_labeled)         
    #classification_report_var = classification_report(ai_model.y_test_df.argmax(axis=1), final_predict_labeled, target_names=ai_model.label_dict)

    # Json Response with all stats/scores
    resJson = {
        "accuracy": accuracy,
        "precision": precision,
        "recall":recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confussion_matrix": str(conf_matrix),
        #"classification_report_var": classification_report_var
    }

    return jsonify({'output':resJson})

# Run the Flask application
if __name__ == '__main__':
    port = 5000
    
    app.run(host='127.0.0.1', port=port, debug=True)