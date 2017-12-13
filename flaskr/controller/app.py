from flask import Flask, request, jsonify
from naive_bayes_classifier import NBClassifier
import json

app = Flask(__name__)

@app.route("/",methods=['GET'])
def howto():
    return "Please POST request via JSON"

@app.route("/classify", methods=['POST'])
def classify():
    term = request.form['term']
    return NBClassifier.classify([term])

@app.route("/classifyjson", methods=['POST'])
def classifyjson():
    data = request.get_json()
    owner = data['owner']
    usage = data['usage']
    booking_text = data['text']

    print(owner + " " + usage + " " + booking_text)
    query = booking_text + " " + owner + " " + usage
    return NBClassifier.classify(query)

if __name__ == '__main__':
    app.run(debug=True)