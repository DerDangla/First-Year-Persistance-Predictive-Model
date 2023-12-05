# Import the Flask class from the flask module
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import joblib
import pandas as pd

ohe = joblib.load("models/ohe.pkl")
scaler = joblib.load("models/scaler.pkl")
nn_model = tf.keras.models.load_model('models/model.h5', compile=False)
# Create an instance of the Flask class
app = Flask(__name__)

# Register a route
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        input_data = {
            'First_Term_GPA': request.form.get('first_term_gpa'),
            'Second_Term_GPA': request.form.get('second_term_gpa'),
            'First_Language': request.form.get('first_language'),
            'Funding': request.form.get('funding'),
            'School': request.form.get('school'),
            'FastTrack': request.form.get('fast_track'),
            'Coop': request.form.get('coop'),
            'Residency': request.form.get('residency'),
            'Gender': request.form.get('gender'),
            'Previous_Education': request.form.get('previous_education'),
            'Age_Group': request.form.get('age_group'),
            'English_Grade': request.form.get('english_grade')
        }

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

    return render_template("index.html", prediction=prediction_result)

@app.route("/api/predict", methods=["POST"])
def predict_api():
    input_data = request.get_json(force=True)

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
            
    return jsonify({prediction: prediction_result})

# Run the Flask application
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)