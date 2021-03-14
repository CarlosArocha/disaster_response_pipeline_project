import sys
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet')

# import libraries
import numpy as np
import pandas as pd
import os
import sqlite3
from sqlalchemy import create_engine
import re
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk import ne_chunk
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

BASE_PATH1 = '/Users/carlosarocha/Dropbox/AI/GITHUB/UDACITY/DATA_SCIENCE/disaster_response_pipeline_project/data/'
BASE_PATH = '/'

def load_data(database_filepath):
    # load data from database
    target_columns = ['related', 'request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'child_alone', 'water', 'food', 'shelter',
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']

    message_column = 'message'

    # Leave this out if the file doesn't exist yet
    #assert os.path.exists(filepath), "The file doesn't exist"
    #conn = sqlite3.connect(filepath)

    #final_path = 'sqlite://'+BASE_PATH+database_filepath
    final_path = '../data/'+database_filepath
    print('final_path 1: ', final_path )
    final_path = os.path.abspath(final_path)
    print('final_path 2: ', final_path )
    final_path = 'sqlite://' + final_path
    print('final_path 3: ', final_path )

    '''app.config[database_filepath.split('/')[-1][:-3]] = 'sqlite:///' + os.path.join(final_path)
    db = SQLAlchemy(app)'''

    #engine = sqlite3.connect(final_path) #, pool_pre_ping=True)

    engine = create_engine(final_path)
    print('one pass +++++++++')
    df = pd.read_sql("SELECT * FROM final", engine)
    print('two pass +++++++++')
    X = df[message_column].values
    Y = df[target_columns].values
    print(df.tail())

    return X, Y, target_columns


def tokenize(text):
    pass


def build_model():
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
