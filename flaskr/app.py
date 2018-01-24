from flask import Flask, request, render_template
from flask_pymongo import PyMongo
from booking_classifier import BookingClassifier
from booking import Booking, BookingSchema
from persistence.db_persist import DBClient
from file_handling.file_handler import FileHandler

app = Flask(__name__)
classifier = BookingClassifier()
file_handler = FileHandler()

app.config['MONGO_DBNAME'] = 'bookingset'
app.config['MONGO_URI'] = 'mongodb://localhost:27017/bookingset'
mongo = PyMongo(app, config_prefix='MONGO')

@app.route("/",methods=['GET'])
def howto():
    return render_template('howto.html'), 200

@app.route("/classifyterm", methods=['POST'])
def classify():
    term = request.form['term']
    return BookingClassifier.classify([term])

@app.route("/classify", methods=['POST'])
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
    #classifier = BookingClassifier()
    return classifier.classify(query)

@app.route("/addbooking", methods=['POST'])
def add_booking():
    #bookings = mongo.db.bookings
    req_data = request.get_json()
    booking_schema = BookingSchema()
    booking, errors = booking_schema.load(req_data)
    if errors:
        print(errors)
        return render_template('404.html'), 404
    else:
        # Insert new booking into CSV
        file_handler.write_csv(booking)

    #booking_id = bookings.insert_one(req_data).inserted_id

    #DBClient().add_booking(booking)
    return "booking added"

#TODO new rules from user and admin

if __name__ == '__main__':
    app.run(debug=True)