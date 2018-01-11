from marshmallow import Schema, fields, pprint, post_load

class Booking():
    def __init__(self, category, booking_date, valuta_date, text, usage, creditor_id, owner, amount, bic):
        self.category = category
        self.booking_date = booking_date
        self.valuta_date = valuta_date
        self.text = text
        self.usage = usage
        self.creditor_id = creditor_id
        self.owner = owner
        self.bic = bic
        self.amount = amount

class BookingSchema(Schema):
    category = fields.Str()
    booking_date = fields.Date()
    valuta_date = fields.Date()
    text = fields.Str()
    usage = fields.Str()
    creditor_id = fields.Str()
    owner = fields.Str()
    bic = fields.Str()
    amount = fields.Float()

    @post_load
    def make_user(self, data):
        return Booking(**data)

'''
b1 = dict(category='wohnenhaushalt', booking_date='01.09.2017', valuta_date='01.09.2017',
               text='FOLGELASTSCHRIFT', usage='KTO 778019565 Abschlag 46,00 EUR faellig 01.09.17 Spenglerstr. 17',
               creditor_id='DE05NAG00000005699', owner='N-Ergie Aktiengesellschaft;DE19700500000000055162',
               bic='BYLADEMMXXX', amount='-46.00')
'''
b2 = {'category':'wohnenhaushalt', 'booking_date':'01.09.2017', 'valuta_date':'01.09.2017',
               'text':'FOLGELASTSCHRIFT', 'usage':'KTO 778019565 Abschlag 46,00 EUR faellig 01.09.17 Spenglerstr. 17',
               'creditor_id':'DE05NAG00000005699', 'owner':'N-Ergie Aktiengesellschaft;DE19700500000000055162',
               'bic':'BYLADEMMXXX', 'amount':'-46.00'}

booking_schema = BookingSchema()
booking = booking_schema.load(b2).data
#pprint(result.data, indent=2)
#booking = result.data
print(type(booking))
print(booking)

