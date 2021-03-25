import json
import plotly
import pandas as pd
import re
import joblib

from sqlalchemy import create_engine
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
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar


app = Flask(__name__)

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
        clean_tok = lemmatizer.lemmatize(tok, \
                tags.get(pos_tag([tok])[0][1][0].upper(), wordnet.NOUN)).lower()
        clean_tok = stemmer.stem(clean_tok)
        if clean_tok not in total_stopwords:
            clean_tokens.append(clean_tok)

    return  clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse_table', engine)

# load model
pkl_file_location = "../models/DisasterResponseModel.pkl"
model = joblib.load(pkl_file_location)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
