# Import the Flask class from the flask module
from flask import Flask, render_template

# Create an instance of the Flask class
app = Flask(__name__)

# Register a route
@app.route('/')
def home():
    return render_template("index.html")

# Run the Flask application
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)