# epayment categorization

### Description
This is an approach to automatically classify SEPA-Bookings to one of seven predefined classes.
The booking-classifier uses a Support Vector Machine that was trained on a set of several bookings.  
**Note**: the words 'classify' and 'categorize' are used synonymously below. 

Also part of this project is a 'comparison'-package which compares the performance of 
Naive Bayes Classifier, K-Nearest-Neighbor and Support Vector Machine on a provided training set.
Consider the training set is not part of this repository due to privacy reasons.

comparison:
- comparison of different classifier algorithms
- hyperparameter estimation with grid search
- plotter for plotting confusion matrix

file_handling:
- reading from and writing to training set 

flaskr:
- REST-API and templates

resources:
- ML model and extracted features

### Run the classifier
To start the booking classifier simply start the 'flaskr/app' module.
The API offers following routes:

| Route       | Description                                                                                                                                                                                         | HTTP methods | Message body                                                                                                                                  |
|-------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| /howto      | Root path and common description                                                                                                                                                                    | GET          | -                                                                                                                                             |
| /categorize | Categorizes a booking and returns the category and probability.                                                                                                                                                                                                    | POST         | {"booking_date":" ","valuta_date":" ","text":" ","usage":" ","creditor_id":" ","owner":" ","iban":" ","bic":" ","amount':" "}                 |
| /inputform  | HTML form for manually categorize a booking. After submitting the categorize path is used.                                                                                                          | GET          | -                                                                                                                                             |
| /addbooking | Adds a booking to the training set and retrains the classifier model.<br/>**Note**: Needs the training set which is not part of this repository!                                                            | POST         | {"category":" ", "booking_date":" ","valuta_date":" ","text":" ","usage":" ","creditor_id":" ","owner":" ","iban":" ","bic":" ","amount':" "} |
| /feedback   | Adds an unrecognized booking to the training set and retrains the classifier model. The category must be provided by the user. <br/>**Note**: Needs the training set which is not part of this repository! | POST         | {"category":" ", "booking_date":" ","valuta_date":" ","text":" ","usage":" ","creditor_id":" ","owner":" ","iban":" ","bic":" ","amount':" "} | 

For '/feedback' there also need to be a running monogodb at localhost to temporarily save the bookings.
