from flask import Flask, request, render_template, session
from flask_pymongo import PyMongo
from booking_classifier import BookingClassifier
from booking import Booking, BookingSchema
from file_handling.file_handler import FileHandler
from categories import FallbackCategorie as fbcat
from categories import Categories as cat
from marshmallow import ValidationError
from flask.sessions import session_json_serializer, SecureCookieSessionInterface
from itsdangerous import URLSafeTimedSerializer
from hashlib import sha1
from bson.objectid import ObjectId
import json
import ast
import pymongo

app = Flask(__name__)
app.secret_key = 'test123' #TODO secure for production environment
classifier = BookingClassifier(flaskr=True)
file_handler = FileHandler()

app.config['MONGO_DBNAME'] = 'bookingset'
app.config['MONGO_URI'] = 'mongodb://localhost:27017/bookingset'
mongo = PyMongo(app, config_prefix='MONGO')

# check mongodb connection
try:
    maxSevSelDelay = 1
    client = pymongo.MongoClient('mongodb://localhost:27017/bookingset',
                                     serverSelectionTimeoutMS=maxSevSelDelay)
    client.server_info()
except pymongo.errors.ServerSelectionTimeoutError as err:
    print("WARNING no mongodb connection available")


s = URLSafeTimedSerializer(
    app.secret_key, salt='cookie-session',
    serializer=session_json_serializer,
    signer_kwargs={'key_derivation': 'hmac', 'digest_method': sha1}
)


@app.route("/", methods=['GET'])
@app.route("/howto", methods=['GET'])
def howto():
    return render_template('howto.html'), 200


@app.route("/classifyterm", methods=['POST']) # DEPRECATED
def classifyterm():
    term = request.form['term']
    return classifier.classify([term])


def categorize_json(req_data):
    req_data = ast.literal_eval(str(req_data))

    if not req_data['booking_date']:
        req_data['booking_date'] = None
    if not req_data['valuta_date']:
        req_data['valuta_date'] = None
    if not req_data['creditor_id']:
        req_data['creditor_id'] = None
    if not req_data['iban']:
        req_data['iban'] = None
    if not req_data['bic']:
        req_data['bic'] = None

    # schema validation and deserialization
    try:
        booking_schema = BookingSchema()
        booking, errors = booking_schema.load(req_data)
        category, probabilities = classifier.classify(booking)

        # if creditor id was found in mongodb probability is 0
        if probabilities == '0':
            resp = {"category": category, "probability": "n/a"}
        else:
            print(round(max(max(probabilities))))
            resp = {"category": category, "probability": round(ast.literal_eval(str(max(max(probabilities)))), 4) * 100}
        if category == fbcat.SONSTIGES.name:
            print('unknown booking. saving to mongodb')
            # save booking temporarily to mongodb for feedback
            bookings = mongo.db.bookings
            booking_id = bookings.insert_one(req_data).inserted_id
            # save mongoid to session cookie
            session['value'] = str(booking_id)
            resp = {"category": category, "probability": round(ast.literal_eval(str(max(max(probabilities)))), 4) * 100}

    except ValidationError as err:
        print(err.messages)
        resp = {"category": "", "probability": 0}

    return resp


def categorize_form(req_data):
    req_data = ast.literal_eval(str(req_data))

    if not req_data['booking_date']:
        req_data['booking_date'] = None
    if not req_data['valuta_date']:
        req_data['valuta_date'] = None
    if not req_data['creditor_id']:
        req_data['creditor_id'] = None
    if not req_data['iban']:
        req_data['iban'] = None
    if not req_data['bic']:
        req_data['bic'] = None

    # schema validation and deserialization
    try:
        booking_schema = BookingSchema()
        booking, errors = booking_schema.load(req_data)
        category, probabilities = classifier.classify(booking)
        wf_category = well_formed_category(category)

        # if creditor id was found in mongodb probability is 0
        if probabilities == '0':
            resp = render_template('result.html', category=wf_category,
                                   prob='n/a')
        else:
            resp = render_template('result.html', category=wf_category,
                                   data=probabilities,
                                   prob=round(ast.literal_eval(
                                       str(max(max(probabilities)))), 4) * 100)
        if category == fbcat.SONSTIGES.name:
            print('unknown booking. saving to mongodb')
            # save booking temporarily to mongodb for feedback
            bookings = mongo.db.bookings
            booking_id = bookings.insert_one(req_data).inserted_id
            # save mongoid to session cookie
            session['value'] = str(booking_id)
            resp = render_template('feedback.html', category=wf_category,
                                   prob=round(ast.literal_eval(str(max(max(probabilities)))), 4) * 100)

    except ValidationError as err:
        print(err.messages)
        resp = render_template('400.html'), 400

    return resp


@app.route("/categorize", methods=['POST'])
def classify_json():
    return json.dumps(categorize_json(request.get_json()))


@app.route("/classifyform", methods=['POST'])
def classify_inputform():
    return categorize_form(json.dumps(request.form))


@app.route("/inputform", methods=['GET'])
def input_form():
    return render_template('inputform.html'), 200

@app.route("/addbooking", methods=['POST'])
def add_booking(booking_req=None):
    booking_schema = BookingSchema()
    if booking_req:
        booking = booking_req
        errors = None
    else:
        req_data = request.get_json()
        booking, errors = booking_schema.load(req_data)

    if errors:
        print(errors)
        return render_template('404.html'), 404
    else:
        # Insert new booking into CSV
        file_handler.write_csv(booking)
        # train the classifier
        classifier.train_classifier()

    return "booking added", 200


@app.route("/feedback", methods=['POST'])
def feedback():
    booking_id = session['value']
    req_data = json.dumps(request.form)
    req_data = ast.literal_eval(str(req_data))
    if 'category' in req_data:
        bookings = mongo.db.bookings
        booking_entry = bookings.find_one({"_id": ObjectId(booking_id)})
        booking = Booking()
        booking.category = req_data['category']
        booking.booking_date = booking_entry['booking_date']
        booking.valuta_date = booking_entry['valuta_date']
        booking.text = booking_entry['text']
        booking.usage = booking_entry['usage']
        booking.creditor_id = booking_entry['creditor_id']
        booking.owner = booking_entry['owner']
        booking.receiver_iban = booking_entry['iban']
        booking.receiver_bic = booking_entry['bic']
        booking.amount = booking_entry['amount']

        # delete booking from mongodb
        bookings = mongo.db.bookings
        bookings.delete_one({"_id": ObjectId(booking_id)})

        if booking:
            add_booking(booking)
    return render_template('/feedback_success.html'), 200


def well_formed_category(category):
    if category.upper() == cat.BARENTNAHME.name:
        return 'Barentnahme'
    elif category.upper() == cat.FINANZEN.name:
        return 'Finanzen'
    elif category.upper() == cat.FREIZEITLIFESTYLE.name:
        return 'Freizeit & Lifestyle'
    elif category.upper() == cat.LEBENSHALTUNG.name:
        return 'Lebenshaltung'
    elif category.upper() == cat.MOBILITAETVERKEHR.name:
        return 'Mobilitaet & Verkehrsmittel'
    elif category.upper() == cat.VERSICHERUNGEN.name:
        return 'Versicherungen'
    elif category.upper() == cat.WOHNENHAUSHALT.name:
        return 'Wohnen & Haushalt'
    else:
        return 'Sonstiges'

if __name__ == '__main__':
    app.session_interface = SecureCookieSessionInterface()
    app.run()