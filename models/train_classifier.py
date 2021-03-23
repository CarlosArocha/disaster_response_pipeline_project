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
from sqlalchemy import create_engine
import re
import pickle
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk import ne_chunk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import tree2conlltags
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score


#BASE_PATH1 = '/Users/carlosarocha/Dropbox/AI/GITHUB/UDACITY/DATA_SCIENCE/disaster_response_pipeline_project/data/'
#BASE_PATH = '/'

def load_data(database_filepath):
    # load data from database
    target_columns = ['related', 'request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'water', 'food', 'shelter', 'clothing', 'money',
       'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report', 'not_related']

    message_column = 'message'

    final_path = 'sqlite:///' + database_filepath
    engine = create_engine(final_path)
    df = pd.read_sql('DisasterResponse_table', engine)

    return df, target_columns

# A simple function to return the X and Y data arrays for training and testing
def XY_values(df, X_columns, Y_columns):
    X = df[X_columns].values
    Y = df[Y_columns].values
    return X, Y

# An approach to fix a little Oversampling the imbalanced data
# Just a bit, too much oversampling push our model to overfit
def fix_imbalanced(data, target_col, factor=2, top_data=0.1):
    new_data = data.copy()
    for col in target_col:
        # taking a sub dataframe where all rows with this column is '1'
        d_base = data[data[col] == 1]
        #print(col, len(d_base), len(data), (len(d_base)/len(data)), int((factor-1)*len(d_base)))
        if (len(d_base) > 0) and ((len(d_base)/len(data))<top_data):
            d_samples = d_base.sample(n = int((factor-1)*len(d_base)), replace=True)
            new_data = pd.concat([new_data, d_samples], ignore_index=True)
            #print(col, len(d_samples))
    return new_data

def imbalanced_rows(data, target_columns, factor=1):
    db = data.copy()
    l = int(len(db[db[target_columns].sum(axis=1) == 1]) * factor)
    d_samples = db[db[target_columns].sum(axis=1) == 1].sample(n = l, replace=False)
    db = pd.concat([db[db[target_columns].sum(axis=1) > 1],d_samples], ignore_index=True)
    new_columns = [col for col in target_columns if db[col].sum(axis=0) > 0]

    return db, new_columns


def tokenize(text):
    # Changing every webpage for a space.
    # With this regex we delete webpages with these characteristics:
    #       1. http://www.name.ext or similar
    #       2. http : www.name.ext or similar
    #       3. http www.name.ext or similar
    url_regex = 'http[s]?[\s]?[:]?[\s]?[\/\/]?(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # Then clean the texts of webpages addresses
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, " ")

    # Forgetting about the numbers and any non letter char
    text = re.sub('[^a-zA-Z]',' ',text)

    tokens = word_tokenize(text.lower())

    tokens = [w for w in tokens if w not in stopwords.words('english')]

    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok, pos='v').lower().strip()
        clean_tok = lemmatizer.lemmatize(clean_tok, pos='n')
        clean_tokens.append(clean_tok)#+'_'+tag)


    return  clean_tokens



def build_model():
    pipeline = Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultiOutputClassifier(RandomForestClassifier()))
                    ])

    parameters = {
                 #'vect__max_df': (0.5, 0.75, 1.0),
                 #'vect__max_features': (None, 5000, 10000)
                 #'vect__min_df': (0.5, 1.0),
                 #'vect__ngram_range': ((1, 1),(2,2))
                 #'tfidf__norm': ('l1','l2'),
                 #'tfidf__use_idf': (True, False),

                 #'clf__estimator__criterion': ('gini','entropy'),
                 #'clf__estimator__max_depth': None,
                 #'clf__estimator__max_leaf_nodes': None,
                 #'clf__estimator__min_samples_split': [2],
                 #'clf__estimator__n_estimators': [100]#, 250, 500],
                 #'clf__estimator__random_state': (None, 0.2)
                 }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):

    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))

def save_model(model, model_filepath):

    #print(BASE_PATH1, model_filepath)
    #final_path = BASE_PATH1+model_filepath
    pickle.dump(model, open(model_filepath, 'wb'))

    # some time later...
    # load the model from disk
    # loaded_model = pickle.load(open(filename, 'rb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))

        df, category_names = load_data(database_filepath)
        df = fix_imbalanced(df, category_names, 1.5, 0.035)

        X, Y = XY_values(df, 'message', category_names)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

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
