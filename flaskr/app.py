from flask import Flask, request, render_template, session
from flask_pymongo import PyMongo
from bson.json_util import dumps
from booking_classifier import BookingClassifier
from booking import Booking, BookingSchema
from persistence.db_persist import DBClient
from file_handling.file_handler import FileHandler
from categories import FallbackCategorie as fbcat
from flaskr.session_handler import ItsdangerousSessionInterface

app = Flask(__name__)
app.secret_key = 'test123' #TODO secure
classifier = BookingClassifier()
file_handler = FileHandler()

app.config['MONGO_DBNAME'] = 'bookingset'
app.config['MONGO_URI'] = 'mongodb://localhost:27017/bookingset'
mongo = PyMongo(app, config_prefix='MONGO')


@app.route("/", methods=['GET'])
@app.route("/howto", methods=['GET'])
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
        query = [booking_text, usage, owner]

        # save term temporary to db for feedback
        booking_schema = BookingSchema()
        booking, errors = booking_schema.load(req_data)

        if errors:
            print('invalid request body')
            resp = render_template('400.html'), 400
        else:
            print('request body is valid')
            category = classifier.classify(query)
            resp = category, 200
            if category == fbcat.SONSTIGES.name:
                print('unknown booking. saving to mongodb')
                bookings = mongo.db.bookings
                booking_id = bookings.insert_one(req_data).inserted_id
                # DBClient().add_booking(booking) #kontextwechsel
                # save mongoid to session cookie

                session['value'] = str(booking_id)
            # TODO if category sonstiges feedback
    else:
        resp = render_template('400.html'), 400
    return resp


@app.route("/correctbooking", methods=['POST'])
def correct_booking():
   req_data = request.get_json()
   cookie = request.cookies.get('session')
   if 'category' and cookie:
       print(cookie)

   return "ok", 200


@app.route("/addbooking", methods=['POST'])
def add_booking():
    req_data = request.get_json()
    booking_schema = BookingSchema()
    booking, errors = booking_schema.load(req_data)
    if errors:
        print(errors)
        return render_template('404.html'), 404
    else:
        # Insert new booking into CSV
        file_handler.write_csv(booking)

    #bookings = mongo.db.bookings
    #booking_id = bookings.insert_one(req_data).inserted_id
    #DBClient().add_booking(booking)
    return "booking added"

@app.route("/feedback", methods=['POST'])
def feedback():
    booking_id = session['value']
    req_data = request.get_json()
    if 'category' in req_data:
        category = req_data['category']
        bookings = mongo.db.bookings
        booking_schema = BookingSchema()
        booking = bookings.find_one({"_id": booking_id})

        if booking:
            add_booking(booking)
    return "Thanks for the feedback", 200

if __name__ == '__main__':
    app.session_interface = ItsdangerousSessionInterface()
    app.run(debug=True)