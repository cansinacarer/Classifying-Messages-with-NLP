# Classifying Tweets with NLP

## Summary

Following a disaster, typically you get millions of communications when the disaster response organizations have the least capacity. Often only one in a thousand messages that require a response from these organizations. In this project, we have created a machine learning model that is used to classify these communications so that they can be directed to the right response organization.

_This project is submitted in partial fulfillment of the [Data Scientist Nanodegree Program](https://www.udacity.com/course/data-scientist-nanodegree--nd025) from Udacity._

## Steps to be Followed for the First Time Run

1. Create a virtual environment (e.g. `python -m venv .venv` on Windows) and activate it.
2. Install the dependencies with `pip install -r requirements.txt`.
3. Run `python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db` in the `data` directory directory to run the ETL pipeline that gets the tweets and categories, cleans this data, and loads it into an sqlite database.
4. Run `python train_classifier.py ../data/DisasterResponse.db classifier.pkl` in the `models` directory to train the model and create a pickle file.
5. Run `python run.py` in the `app` directory to run the web application that uses this pickle file.

## File Descriptions

-   `data/process_data.py` loads the data from `disaster_categories.csv` and `disaster_messages.csv`, does the necessary transformations to clean and merge these datasets, then loads it into the SQLite database in `DisasterResponse.db`.
-   `train_classifier.py` loads the data from the SQLite database, splits the dataset into training and test sets, builds a text processing and machine learning pipeline, trains and tunes a model using GridSearchCV, outputs results on the test set, and exports the final model as a pickle file.
-   `run.py` is a simple Flask app that used to create a user interface for classifying new messages manually and to display visualizations about the data.

## Acknowledgements

Data used for this project is from Figure Eight.
