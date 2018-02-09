""" schema for a bank booking """

from marshmallow import Schema, fields, pprint, post_load


class Booking:
    def __init__(self, category=None, booking_date=None, valuta_date=None,
                 text=None, usage=None, creditor_id=None, owner=None,
                 iban=None, bic=None, amount=None):
        self.category = category
        self.booking_date = booking_date
        self.valuta_date = valuta_date
        self.text = text
        self.usage = usage
        self.creditor_id = creditor_id
        self.owner = owner
        self.receiver_iban = iban
        self.receiver_bic = bic
        self.amount = amount

    def to_array(self):
        return [self.category, self.booking_date, self.valuta_date, self.text,
                self.usage, self.creditor_id, self.owner,
                self.receiver_iban, self.receiver_bic, self.amount]


class BookingSchema(Schema):
    category = fields.Str()
    booking_date = fields.Date()
    valuta_date = fields.Date()
    text = fields.Str(required=True)
    usage = fields.Str(required=True)
    creditor_id = fields.Str()
    owner = fields.Str(required=True)
    iban = fields.Str()
    bic = fields.Str()
    amount = fields.Float(required=True)

    @post_load
    def make_booking(self, data):
        return Booking(**data)


class BookingCatSchema(Schema):
    category = fields.Str(required=True)
    booking_date = fields.Date()
    valuta_date = fields.Date()
    text = fields.Str(required=True)
    usage = fields.Str(required=True)
    creditor_id = fields.Str()
    owner = fields.Str(required=True)
    iban = fields.Str()
    bic = fields.Str()
    amount = fields.Float()

    @post_load
    def make_booking(self, data):
        return Booking(**data)

'''
b1 = dict(category='wohnenhaushalt', booking_date='01.09.2017', valuta_date='01.09.2017',
               text='FOLGELASTSCHRIFT', usage='KTO 778019565 Abschlag 46,00 EUR faellig 01.09.17 Spenglerstr. 17',
               creditor_id='DE05NAG00000005699', owner='N-Ergie Aktiengesellschaft;DE19700500000000055162',
               bic='BYLADEMMXXX', amount='-46.00')
'''
'''
req_data = {'category':'wohnenhaushalt', 'booking_date':'01.09.2017', 'valuta_date':'01.09.2017',
               'text':'FOLGELASTSCHRIFT', 'usage':'KTO 778019565 Abschlag 46,00 EUR faellig 01.09.17 Spenglerstr. 17',
               'creditor_id':'DE05NAG00000005699', 'owner':'N-Ergie Aktiengesellschaft', 'iban':'DE19700500000000055162',
               'bic':'BYLADEMMXXX', 'amount':'-46.00'}
'''

#booking_schema = BookingSchema()
#booking, errors = booking_schema.load(req_data)

#print(booking.to_array())


#if errors:
#    print(errors)
#else:
#    print(booking.usage)
#pprint(result.data, indent=2)
#booking = result.data
#print(type(booking))
#print(booking)

