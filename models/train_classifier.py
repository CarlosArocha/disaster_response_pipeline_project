import sys
import nltk
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('stopwords')
#nltk.download('maxent_ne_chunker')
#nltk.download('words')
#nltk.download('wordnet')

# import libraries
import numpy as np
import pandas as pd
import os
import sqlite3
import pickle
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
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


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

    final_path = 'sqlite:///' + database_filepath
    engine = create_engine(final_path)
    df = pd.read_sql('DisasterResponse_table', engine)
    X = df[message_column].values
    Y = df[target_columns].values
    print(df.tail())

    return X, Y, target_columns


def tokenize(text):
    url_regex = 'http[s]?[\s]?[:]?[\s]?[\/\/]?(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        text = re.sub('[^a-zA-Z0-9]',' ',text)

    tokens = word_tokenize(text)
    #words = [w for w in tokens if w.lower() not in stopwords.words('english')]

    #poswords = pos_tag(words)

    #chunkwords = ne_chunk(poswords)

    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok, pos='v')
        clean_tokens.append(clean_tok)


    return  clean_tokens


def build_model():
    pipeline = Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultiOutputClassifier(SGDClassifier()))
                    ])
    '''parameters = {
            #'vect__ngram_range': ((1, 1), (1,2)),
            #'vect__max_df': (0.5, 0.75, 1.0),
            'vect__max_features': (None, 5000), #10000
            #'tfidf__use_idf': (True, False),
            #"clf": [RandomForestClassifier()],
            #"clf__n_estimators": [10, 100, 250],
            #"clf__max_depth":[8],
            #"clf__random_state":[42],
            }

    cv = GridSearchCV(pipeline, param_grid=parameters)'''

    return pipeline #cv


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))

def save_model(model, model_filepath):

    print(BASE_PATH1, model_filepath)
    final_path = BASE_PATH1+model_filepath
    pickle.dump(model, open(final_path, 'wb'))

    # some time later...
    # load the model from disk
    # loaded_model = pickle.load(open(filename, 'rb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        '''classifiers = [  #KNeighborsClassifier(3),\ 0.77-0.74
                            #DecisionTreeClassifier(max_depth=5),\ 0.77-0.64
                            #RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),\ 0.76-0.76
                            #RandomForestClassifier() #0.84-0.72
                            #GradientBoostingClassifier(),
                            #GaussianNB()
                            #SVC()
                            #LogisticRegression()
                ]'''

        print('Building model...')
        model = build_model()

        print(X_train.shape, Y_train.shape)
        print('Training model...')
        model.fit(X_train, Y_train)

        score = model.score(X_test, Y_test)
        print(score)

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
