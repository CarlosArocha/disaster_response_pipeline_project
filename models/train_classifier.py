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
import time
from sqlalchemy import create_engine
import re
import pickle
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import ne_chunk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
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


def load_data(database_filepath):
    # load data from database

    final_path = 'sqlite:///' + database_filepath
    engine = create_engine(final_path)
    df = pd.read_sql('DisasterResponse_table', engine)

    target_columns = ['related', 'request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'water', 'food', 'shelter', 'clothing', 'money',
       'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report', 'not_related']

    return df, target_columns

# A simple function to return the X and Y data arrays for training and testing
def XY_values(df, X_columns, Y_columns):
    X = df[X_columns].values
    Y = df[Y_columns].values
    return X, Y

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

    tokens = word_tokenize(text)

    tags = {"J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV}

    particular_words = ['kg']
    total_stopwords = particular_words + stopwords.words('english')

    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok, tags.get(pos_tag([tok])[0][1][0].upper(), wordnet.NOUN)).lower()
        clean_tok = stemmer.stem(clean_tok)
        if clean_tok not in total_stopwords:
            clean_tokens.append(clean_tok)


    return  clean_tokens

class OrganizationPresence(BaseEstimator, TransformerMixin):

    def checking_org(self, text):
        #
        words = word_tokenize(text)
        words = [w for w in words if w.lower() not in stopwords.words('english')]

        ptree = pos_tag(words)

        for w in tree2conlltags(ne_chunk(ptree)):
            if (w[2][2:] == 'ORGANIZATION') and (w[1] == 'NNP'):
                return True

            return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        #
        X_org = pd.Series(X).apply(self.checking_org)

        return pd.DataFrame(X_org)

class TextLengthExtractor(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        #
        return pd.DataFrame(pd.Series(X).apply(lambda x: len(x)))

def build_model():
    # Then our final pipeline with best parameters and feature union implementation
    pipeline = Pipeline([\
                        ('features', FeatureUnion([\
                            ('text_pipeline', Pipeline([\
                                ('vect', CountVectorizer(tokenizer=tokenize,\
                                                         max_df=1.0,\
                                                         max_features=None,)),\
                                ('tfidf', TfidfTransformer(norm='l2',\
                                                           use_idf=True,))\
                            ])),\
                            ('org_presence', OrganizationPresence()),\
                            ('text_length', TextLengthExtractor())\
                        ])),\
                        ('clf', RandomForestClassifier(criterion='gini',\
                                                        n_estimators=250,\
                                                        random_state=42,))\
                        ])

    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):

    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))
    print('accuracy:', accuracy_score(Y_test, y_pred))

def save_model(model, model_filepath):

    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))

        df, category_names = load_data(database_filepath)

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
