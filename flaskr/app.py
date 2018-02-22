from flask import Flask, request, render_template, session
from flask_pymongo import PyMongo
from booking_classifier import BookingClassifier
from booking import Booking, BookingSchema, BookingCatSchema
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

app = Flask(__name__)
app.secret_key = 'test123' #TODO secure
classifier = BookingClassifier()
file_handler = FileHandler()

# TODO check MongoDB connection during startup
app.config['MONGO_DBNAME'] = 'bookingset'
app.config['MONGO_URI'] = 'mongodb://localhost:27017/bookingset'
mongo = PyMongo(app, config_prefix='MONGO')

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
    return BookingClassifier.classify([term])


def categorize(req_data):
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
        #print(type(booking))
        category, probability = classifier.classify(booking)

        wf_category = well_form_category(category)
        resp = wf_category, round(ast.literal_eval(probability), 4)*100
        if category == fbcat.SONSTIGES.name:
            print('unknown booking. saving to mongodb')
            # save booking temporarily to mongodb for feedback
            bookings = mongo.db.bookings
            booking_id = bookings.insert_one(req_data).inserted_id
            # DBClient().add_booking(booking) #ontextwechsel
            # save mongoid to session cookie

            session['value'] = str(booking_id)
        # TODO if category sonstiges feedback
    except ValidationError as err:
        print(err.messages)
        resp = render_template('400.html'), 400

    return resp


@app.route("/categorize", methods=['POST'])
def classify_json():
    cat, prob = categorize(request.get_json())
    return render_template('result.html', category=cat, prob=prob)


@app.route("/classifyform", methods=['POST'])
def classify_inputform():
    cat, prob = categorize(json.dumps(request.form))
    return render_template('result.html', category=cat, prob=prob)


@app.route("/inputform", methods=['GET'])
def input_form():
    return render_template('inputform.html'), 200


@app.route("/correctbooking", methods=['POST'])
def correct_booking():
    req_data = request.get_json() # TODO use categorize method

    # schema validation and deserilization
    try:
        booking_schema = BookingCatSchema()
        booking, errors = booking_schema.load(req_data)

        session_data = s.loads(request.cookies.get('session'))
        bookings = mongo.db.bookings
        print(session_data['value'])

        # Convert to object id
        booking_entry = bookings.find_one({"_id":ObjectId(session_data['value'])})
        #booking, errors = booking_schema.load(booking_entry)
        #print(booking)

        # Insert booking to training set
        file_handler.write_csv(booking)

        resp = 'ok', 200
    except ValidationError as err:
        print(err.messages)
        resp = render_template('400.html'), 400

    return resp


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
    return "booking added", 200


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


def well_form_category(category):
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
    #app.session_interface = ItsdangerousSessionInterface()
    app.session_interface = SecureCookieSessionInterface()
    app.run(debug=True)