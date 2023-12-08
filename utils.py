import joblib
import tensorflow as tf
import pandas as pd
import numpy as np
import json

ohe = joblib.load("models/ohe.pkl")
scaler = joblib.load("models/scaler.pkl")
nn_model = tf.keras.models.load_model('models/model.h5', compile=False)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def make_prediction(input_data):
    input_df = pd.DataFrame([input_data])
    new_input_numeric = scaler.transform(input_df.iloc[:,0:3])
    new_input_ohe = ohe.transform(input_df.iloc[:,3:])
    new_input = np.concatenate((new_input_numeric, new_input_ohe), axis=1)
    
    # Make a prediction
    prediction = nn_model.predict(new_input)

    # Since it's a binary classification, you can convert this probability to a class label
    predicted_class = (prediction > 0.5).astype(int)

    prediction_result = "Student will not Persist"
    if predicted_class[0] == 1:
        prediction_result = "Student will Persist"

    return prediction_result