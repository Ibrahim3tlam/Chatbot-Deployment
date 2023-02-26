import nltk
from nltk.stem.isri import ISRIStemmer
import numpy as np
import random
from flask import Flask, render_template, request
import joblib
from keras.models import load_model
nltk.download('punkt')
model = load_model('chatbot_model.h5')


## Load the data object from the pickled file

data = joblib.load('intents.pkl')
classes = joblib.load('classes.pkl')
words = joblib.load('words.pkl')


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    stemmer = ISRIStemmer()  # Stemmer function
    sentence_words = [stemmer.stem(w) for w in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return (np.array(bag))


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, data):
    tag = ints[0]['intent']
    list_of_intents = data['intents']

    for i in list_of_intents:
        if (i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result, tag


def process(text):
    ints = predict_class(text, model)
    res = getResponse(ints, data)
    return res


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route("/get")
def get_bot_reponse():
    userText = request.args.get('msg')
    results, tag = process(userText)
    return results


if __name__ == '__main__':
    app.run(debug=True)
