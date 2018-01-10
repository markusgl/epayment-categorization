from flask import Flask, request, render_template
from booking_classifier import BookingClassifier
from booking import Booking, BookingSchema
from persistence.db_persist import DBClient

app = Flask(__name__)
classifier = BookingClassifier()

@app.route("/",methods=['GET'])
def howto():
    return "Please POST request via JSON"

@app.route("/classify", methods=['POST'])
def classify():
    term = request.form['term']
    return BookingClassifier.classify([term])

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
    classifier = BookingClassifier()
    return classifier.classify(query)

@app.route("/addbooking", methods=['POST'])
def add_booking():
    req_data = request.get_json()
    booking_schema = BookingSchema()
    booking, errors = booking_schema.load(req_data).data
    if errors:
        return render_template('404.html'), 404
    DBClient().add_booking(booking)
    return "added booking"

#TODO new rules from user and admin

if __name__ == '__main__':
    app.run(debug=True)