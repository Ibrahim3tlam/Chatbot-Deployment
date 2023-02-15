import nltk
from nltk.stem.isri import ISRIStemmer
import numpy as np
import random
from flask import Flask, render_template, request
import joblib
from keras.models import load_model


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route("/get")
def get_bot_reponse():
    userText = request.args.get('msg')
    results, tag = process(userText)
    return 'good'


if __name__ == '__main__':
    app.run(debug=True)
