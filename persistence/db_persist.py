from pymongo import MongoClient

class DBClient:
    def __init__(self):
        client = MongoClient('mongodb://localhost:27017/')
        db = client.bookingset
        self.bookings = db.bookings

    def add_booking(self, booking):
        id = self.bookings.insert_one(booking).inserted_id
        print("successfully added " + str(id))



#booking = {"owner":"testo", "usage":"testu", "text":"testt"}
#post_id = bookings.insert_one(booking).inserted_id
#print(post_id)

