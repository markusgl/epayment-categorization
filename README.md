# epayment categorization

This is an approach to automatically classify SEPA-Bookings to one of seven predefined classes.
The booking-classifier uses a Support Vector Machine that was trained on a set of several bookings.

Also part of this project is a 'comparison'-package which compares the performance of 
Naive Bayes Classifier, K-Nearest-Neighbor and Support Vector Machine on a provided training set.
Consider the training set is not part of this repository due to privacy reasons!

comparison:
- comparision of different classifier algorithms
- hyperparameter estimation with grid search
- plotter for plotting confusion matrix

file_handling:
- reading an writing to training set 

flaskr:
- REST-API and templates

resources:
- model and features

To start the booking classifier simply start the 'app' module.

Consider there need to be a monogodb at localhost for the 'feedback feature'.
Otherwise this won't work but classification will also work without mongodb.
