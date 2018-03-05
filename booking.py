""" schema for a bank booking """

from marshmallow import Schema, fields, post_load


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
    category = fields.Str(required=False, allow_none=True)
    booking_date = fields.Date(required=False, allow_none=True)
    valuta_date = fields.Date(required=False, allow_none=True)
    text = fields.Str(required=True, allow_none=True)
    usage = fields.Str(required=True, allow_none=True)
    creditor_id = fields.Str(required=False, allow_none=True)
    owner = fields.Str(required=True, allow_none=True)
    iban = fields.Str(required=False, allow_none=True)
    bic = fields.Str(required=False, allow_none=True)
    amount = fields.Float(required=True, allow_none=True)

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

