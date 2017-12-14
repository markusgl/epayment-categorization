from flask import Flask, request, render_template
from naive_bayes_classifier import NBClassifier

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
    req_data = request.get_json()
    if 'text' and 'usage' and 'owner' in req_data:
        booking_text = req_data['text']
        usage = req_data['usage']
        owner = req_data['owner']
        #print(booking_text + " " + usage + " " + owner)
        query = [booking_text, usage, owner]
    else:
        return render_template('404.html'), 404
    classifier = NBClassifier()
    return classifier.classify(query)

if __name__ == '__main__':
    app.run(debug=True)