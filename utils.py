import joblib
import tensorflow as tf
import pandas as pd

ohe = joblib.load("models/ohe.pkl")
scaler = joblib.load("models/scaler.pkl")
nn_model = tf.keras.models.load_model('models/model.h5', compile=False)

def make_prediction(input_data):
    categorical_cols = ["First_Language", "Funding", "School", "FastTrack", "Coop", "Residency", "Gender", "Previous_Education", "Age_Group"]
    numeric_cols = ["First_Term_GPA", "Second_Term_GPA", "English_Grade"]

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Apply One-Hot Encoding on categorical columns
    input_df_categorical = ohe.transform(input_df[categorical_cols])

    # Create a DataFrame from the encoded data
    input_df_categorical = pd.DataFrame(input_df_categorical, columns=ohe.get_feature_names_out())

    # Drop original categorical columns and concatenate the new one-hot encoded columns
    input_df.drop(categorical_cols, axis=1, inplace=True)
    input_df = pd.concat([input_df, input_df_categorical], axis=1)

    # Scale numerical columns
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # Make a prediction
    prediction = nn_model.predict(input_df)

    # Since it's a binary classification, you can convert this probability to a class label
    predicted_class = (prediction > 0.5).astype(int)

    prediction_result = "Student will not Persist"
    if predicted_class[0] == 1:
        prediction_result = "Student will Persist"