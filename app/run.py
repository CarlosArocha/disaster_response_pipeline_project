import json
import plotly
import pandas as pd
import re
import joblib
import sys
sys.path.append('../models')
from train_classifier import OrganizationPresence, TextLengthExtractor
from train_classifier import tokenize

from sqlalchemy import create_engine
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar


app = Flask(__name__)

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

    # Now define a list with the categories to be count and graph.
    category_names = ['related', 'request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'water', 'food', 'shelter', 'clothing', 'money',
       'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report', 'not_related']
    # Second graph counting the related categories in total
    category_counts = df[category_names].sum()
    category_names = [re.sub('_',' ', text) for text in category_names]


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
                'title': 'Distribution of Messages by Genre',
                'yaxis': {
                    'title': "Total"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },{
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],
            # For the second graph we repeat the same structure with some
            # particular attributes.
            'layout': {
                'title': 'Distribution of Messages by Category',
                'yaxis': {
                    'title': "Total"
                },
                'xaxis': {
                    'title': "Categories",
                    'tickangle': 37,
                    'tickfont': {
                        'size':10
                    }

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
