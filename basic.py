
from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def home():
    return "<h1>Hello! SUJITH REDDY</h1>"

if __name__ == "__main__":
    app.run(debug=True)