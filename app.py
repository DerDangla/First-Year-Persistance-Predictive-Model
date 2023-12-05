# Import the Flask class from the flask module
from flask import Flask, render_template, request, jsonify
from utils import make_prediction
import pandas as pd
import os

os.environ.get('KEY')

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
        prediction_result = make_prediction(input_data)

    return render_template("index.html", prediction_result=prediction_result)

@app.route("/api/predict", methods=["POST"])
def predict_api():
    input_data = request.get_json(force=True)

    prediction_result = make_prediction(input_data)
            
    return jsonify({prediction_result: prediction_result})

# Run the Flask application
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)