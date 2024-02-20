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
import random
from os import path
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

os.environ.get('KEY')

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})


#Register a route
@app.route("/")
def home():
    return render_template("index.html")

class AIModel():

    # This class will handle the model file and the csv files.
    def __init__(self):
        #load the model
        self.model = tf.keras.models.load_model('models/model.keras', compile=False)
        #self.ohe = joblib.load("models/ohe.pkl")
        self.scaler = joblib.load("models/scaler.pkl")

        # Load data
        #self.deploy_folder = r'data'

        #self.X_test_df = pd.read_csv(path.join(self.deploy_folder,"X_test_data.csv"))
        #self.y_test_df = pd.read_csv(path.join(self.deploy_folder,"y_test_data.csv")).to_numpy()


@app.route("/api/predict", methods=["POST"])
def predict():
    
    # AI Model object
    nn_model = AIModel()
    
    # Get values from the POST Json Data\
    input_data = request.json
    
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Check if Second_Term_GPA is None and process accordingly
    input_df['Second_Term_GPA'] = pd.to_numeric(input_df['Second_Term_GPA'], errors='coerce')  # Convert non-numeric to NaN
    input_df['Second_Term_GPA_missing'] = input_df['Second_Term_GPA'].isnull().apply(lambda x: 'Y' if x else 'N')
    input_df['Second_Term_GPA'] = input_df['Second_Term_GPA'].fillna(0)
    
    # Apply Ordinal Encoding to English_Grade
    english_grade_map = {
        "Level-130": 1,
        "Level-131": 2,
        "Level-140": 3,
        "Level-141": 4,
        "Level-150": 5,
        "Level-151": 6,
        "Level-160": 7,
        "Level-161": 8,
        "Level-170": 9,
        "Level-171": 10,
        "Level-180": 11 
    }
    
    # Map english grade to the ordinal encoding
    input_df['English_Grade'] = input_df['English_Grade'].map(english_grade_map ).fillna("Unknown")
    
    # Identify the categorical columns excluding 'English_Grade'
    categorical_columns = ['First_Language','Funding','FastTrack','Previous_Education','Age_Group','Second_Term_GPA_missing'] 

    # One hot encode the categorical columns using pd.get_dummies
    input_df = pd.get_dummies(input_df, columns=categorical_columns, dtype=int) 
    
    # Ensure all necessary columns are present, filling missing ones with 0 (for the one-hot encoded features)
    encoded_columns = ['First_Term_GPA', 'Second_Term_GPA', 'English_Grade',
       'First_Language_English', 'First_Language_French',
       'First_Language_Other', 'Funding_Apprentice_PS', 'Funding_GPOG_FT',
       'Funding_Intl Regular', 'Funding_Intl Transfer',
       'Funding_Second Career Program', 'FastTrack_N', 'FastTrack_Y',
       'Previous_Education_HighSchool', 'Previous_Education_PostSecondary',
       'Age_Group_0 to 18', 'Age_Group_19 to 20', 'Age_Group_21 to 25',
       'Age_Group_26 to 30', 'Age_Group_31 to 35', 'Age_Group_36 to 40',
       'Age_Group_41 to 50', 'Age_Group_51 to 60', 'Second_Term_GPA_missing_N',
       'Second_Term_GPA_missing_Y']
    
    # Ensure all necessary columns are present, filling missing ones with 0 (for the one-hot encoded features)
    for column in encoded_columns:
        if column not in input_df.columns:
            input_df[column] = 0
    
    # Remove any columns in input_df that are not in X_feature_encoded
    columns_to_remove = [column for column in input_df.columns if column not in encoded_columns]
    input_df.drop(columns=columns_to_remove, inplace=True)
    
    # Standardization
    numeric_cols = ['First_Term_GPA','Second_Term_GPA']
    input_df[numeric_cols] = nn_model.scaler.transform(input_df[numeric_cols])
    
    # Make a prediction
    prediction = nn_model.model.predict(input_df)

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
    
    # nn_model = AIModel()
    
    # # Just reshape it directly
    # y_test_reshaped = nn_model.y_test_df.reshape(92,1)
    
    # test_set = np.concatenate((nn_model.scaler.inverse_transform(nn_model.X_test_df.iloc[:,0:3]), 
    #                            nn_model.ohe.inverse_transform(nn_model.X_test_df.iloc[:,3:]), 
    #                            y_test_reshaped), axis=1)
    
    # random_index = np.random.randint(0, test_set.shape[0])
    
    # # Convert NumPy array to list before jsonify
    # random_row = test_set[random_index].tolist()
    
    
    random_row = {
        'First_Term_GPA': round(random.uniform(0.0, 5.0), 5),
        'Second_Term_GPA': 'None' if random.choice([True, False]) else round(random.uniform(0.0, 5.0), 5),
        'First_Language': random.choice(['English', 'French', 'Other']),
        'Funding': random.choice(['Apprentice_PS', 'GPOG_FT', 'Intl Offshore', 'Intl Regular', 'Intl Transfer', 'Joint Program Ryerson', 'Joint Program UTSC', 'Second Career Program', 'Work Safety Insurance Board']),
        'FastTrack': random.choice(['Y', 'N']),
        'Previous_Education': random.choice(['HighSchool', 'PostSecondary']),
        'Age_Group': random.choice(['0 to 18', '19 to 20', '21 to 25', '26 to 30', '31 to 35', '36 to 40', '41 to 50', '51 to 60', '61 to 65', '66+']),
        'English_Grade': random.choice(['Level-130', 'Level-131', 'Level-140', 'Level-141', 'Level-150', 'Level-151', 'Level-160', 'Level-161', 'Level-170', 'Level-171', 'Level-180'])
    }
    
    return random_row

# Run the Flask application
if __name__ == '__main__':
    port = 5000
    app.run(host='0.0.0.0', debug=True)
    #app.run(host='127.0.0.1', port=port, debug=True)