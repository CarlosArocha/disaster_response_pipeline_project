# Initial importing
import sys
import nltk
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('stopwords')
#nltk.download('maxent_ne_chunker')
#nltk.download('words')
#nltk.download('wordnet')

# Necessary libraries
import numpy as np
import pandas as pd
import re
import pickle

from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import ne_chunk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk import tree2conlltags
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split


def load_data(database_filepath):
    '''
    This function is encharge of download the cleaned data file to be trained.

    Function Parameters:

        Required:

            database_filepath : file path ; the exact data path or address where
                                the file is located.

        Return:

            df : pd.Dataframe ; the data loaded.

            target_columns : list ; the feature column names that will be the
                             target of our data to be trained.

    '''

    final_path = 'sqlite:///' + database_filepath
    # First create the engine representing the SQLite file and tables.
    engine = create_engine(final_path)
    # Extracting the table in 'df'
    df = pd.read_sql('DisasterResponse_table', engine)
    # In this case we will declare the list of column target instead of create
    # a function to find them. We know them well.
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
    '''
    This function was declared for practical purposes. It split the data set in
    source(X) and target(Y) data arrays for machine learning training purposes.

    Function Parameters:

        Required:

            df : pd.Dataframe ; The original dataframe to be splitted.

            X_columns :  str ; The name of the column that will be took to
                        generate the source data array.

            Y_columns : list of str ; The list of the names that represent the
                        feature column names.

        Return:

            X, Y : values array ; Source (X) and Target(Y) data respectively.
    '''
    # Just taking the values of the columns received in each case. X = input or
    # source, Y = target or results.
    X = df[X_columns].values
    Y = df[Y_columns].values

    return X, Y

def tokenize(text):
    '''
    The tokenizer. The function in charge of processing text data, dividing and
    analyzing it in each call. This function will clean the text from web page
    addresses, it will split the text into word tokens, clean them from numbers
    and quotation marks or other trademark symbols, classify them, clean them
    from often words, and finally simplify the words to send a successful
    result.

    Function Parameters:

        Required:

            text : str ; the text to be tokenized.

        Return:

            clean_tokens : list of str ; the list of cleaned and treated words.
    '''
    # Changing every webpage for a space.
    # With this regex we delete webpages with these characteristics:
    #       1. http://www.name.ext or similar
    #       2. http : www.name.ext or similar
    #       3. http www.name.ext or similar
    # We make it in two part not to pass the 80 columns rule.
    url_regex_1 = 'http[s]?[\s]?[:]?[\s]?[\/\/]?(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
    url_regex_2 = '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    url_regex = url_regex_1 + url_regex_2

    # Then clean the texts of webpages addresses
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, " ")

    # Forgetting about the numbers and any non letter char
    text = re.sub('[^a-zA-Z]',' ',text)

    # Starting our bag of words.
    tokens = word_tokenize(text)
    # Declaring the kind of tags will our lemmatizer work
    tags = {"J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV}
    # Creating a list of words to be add to our stopwords list=total_stopwords,
    # to eliminate them from our bag of words list. Use this list to improve our
    # selection and have a cleaner results.
    particular_words = ['kg']
    total_stopwords = particular_words + stopwords.words('english')

    # Declaration of our lemmatizer and stemmer.
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    # A loop to iterate in the list of words=tokens, for lemmitizing and
    # stemming purposes. Adding the results to a new clean_tokens list.
    clean_tokens = []
    for tok in tokens:
        # The lemmitizer will act depending of the tag of each word.
        clean_tok = lemmatizer.lemmatize(tok,\
                                    tags.get(pos_tag([tok])[0][1][0].upper(),\
                                    wordnet.NOUN)).lower()
        clean_tok = stemmer.stem(clean_tok)
        if clean_tok not in total_stopwords:
            clean_tokens.append(clean_tok)


    return  clean_tokens

class OrganizationPresence(BaseEstimator, TransformerMixin):
    '''
    This transforming class will detect, helped by the 'ne_chunk' function, the
    presence of an organization's name in the text. That will help us to add
    features to our training data.

        Internal function:

            checking_org :

                parameters : text : str ; the text to be searched of an
                            organization's names.

                returns : True or False ; the presence of an organization's name

            fit :

                returns : self data, no changes.

            transform :

                returns : pd.Dataframe ; of a serie of True/False values of an
                        organization's names presenced in each text.
    '''

    # The function that performs really the transformation in this class.
    # It will tokenize the words of the text received, delete the stopwords,
    # and finally will check, helped by ne_chunk function if the any word
    # represent an organization.
    def checking_org(self, text):
        # First list of words, and cleaning from stopwords.
        words = word_tokenize(text)
        words = [w for w in words \
                    if w.lower() not in stopwords.words('english')]
        # Tagging the list.
        ptree = pos_tag(words)
        # FInally we simplify the tree and check if any word represents an
        # organization. This check can be definitvely inproved.
        for w in tree2conlltags(ne_chunk(ptree)):
            if (w[2][2:] == 'ORGANIZATION') and (w[1] == 'NNP'):
                return True

            return False

    # Fit function with just a structure purpose.
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        # The transform function that call to check every text in the input
        # series.
        X_org = pd.Series(X).apply(self.checking_org)

        return pd.DataFrame(X_org)

class TextLengthExtractor(BaseEstimator, TransformerMixin):
    '''
    This transforming class will calculate the length of each text in the course
    and delivery a dataframe of them.

        Internal function:

            fit :

                returns : self data, no changes.

            transform :

                returns : pd.Dataframe ; of a serie of numbers representing the
                            length of each text.
    '''
    # Fit function with just a structure purpose.
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # The function that transform the class object in a dataframe with the
        # length of every text in the data serie received.
        return pd.DataFrame(pd.Series(X).apply(lambda x: len(x)))

def build_model():
    '''
    The model funtion. Declaring the pipeline of our classification and Training
    process. In our case we make a bag-of-words, creates a tdidf array and add
    two custom features. Finally we use Random Forest Classifier as our machine
    learning algorithm.

        Function parameters: None.

        Return:

            pipeline : a pipeline object with our sequence for training.
    '''

    # Then our final pipeline with the best parameters founded.
    # Implementing Feature Union to add two new features to our data, and
    # a Random Forest Classifier.
    pipeline = Pipeline([\
                        ('features', FeatureUnion([\
                            # The text pipeline transformers.
                            ('text_pipeline', Pipeline([\
                                ('vect', CountVectorizer(tokenizer=tokenize,\
                                                         max_df=1.0,\
                                                         max_features=None,)),\
                                ('tfidf', TfidfTransformer(norm='l2',\
                                                           use_idf=True,))\
                            ])),\
                            # The new two features added.
                            ('org_presence', OrganizationPresence()),\
                            ('text_length', TextLengthExtractor())\
                        ])),\
                        # Our final ML algorithm.
                        ('clf', RandomForestClassifier(criterion='gini',\
                                                        n_estimators=250,\
                                                        random_state=42,))\
                        ])

    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    This function will evaluate the trained model with the testing data. Then
    will print the results.

        Function parameters:

            model : pipeliine object ; the trained model to be tested.

            X_test : array ; our input data for testing purposes.

            Y_test : multi dimensional array ; the results for testing purposes.

            category_names: list of str ; the list of feature column names in
                            the test resulting data.

        Return :  None : the results will be printed.

    '''
    # Prediction of our test results.
    y_pred = model.predict(X_test)
    # Printing the classification report and accuracy results.
    print(classification_report(Y_test, y_pred, target_names=category_names))
    print('accuracy:', accuracy_score(Y_test, y_pred))

def save_model(model, model_filepath):
    '''
    Simply this function will save the trained model into a pickle or seried
    file.

        Function Parameters:

            model : pipeline object ; the model to be saved.

            model_filepath: file path ; the location where the file will be
                            saved.

        Returns :  None : the file will be saved in the location addressed.
    '''
    # Savinig the model as a pickle series data file.
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
