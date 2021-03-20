import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine
import joblib


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays visuals and takes user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data required for visuals
    genre_number = df.groupby('genre').count()['message']
    genre_names = list(genre_number.index)
    
    # create visuals
    categories =  df[df.columns[4:]]
    # rename columns for visuals
    categories.rename(columns=lambda x: x[9:], inplace=True)
    categories_numbers = (categories.mean()*categories.shape[0]).sort_values(ascending=False)
    categories_names = list(categories_numbers.index)

    dist_by_genre = df[df.genre == 'direct']
    dist_by_genre.rename(columns=lambda x: x[9:], inplace=True)
    dist_by_genre_numbers = (dist_by_genre.mean()*dist_by_genre.shape[0]).sort_values(ascending=False)
    dist_by_genre_counts = list(dist_by_genre_numbers.values)
    dist_by_genre_counts = dist_by_genre_counts [1:]
    dist_by_genre_names = list(dist_by_genre_numbers.index)
    dist_by_genre_names = dist_by_genre_names[1:]

    
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_number
                )
            ],

            'layout': {
                'title': 'Messages by genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=categories_names,
                    y=categories_numbers
                )
            ],

            'layout': {
                'title': 'Message by categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
            
        },

        {
            'data': [
                Bar(
                    x=dist_by_genre_names,
                    y=dist_by_genre_counts
                )
            ],

            'layout': {
                'title': 'Categories distribution in direct genre',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories in genre"
                }
            }
        }

        
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render webpage with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# webpage that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to classify the query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    #Deploying on localhost to make it more robust:
    #app.run(host='0.0.0.0', port=3001, debug=True)
    app.run(host='localhost', debug=True)


if __name__ == '__main__':
    main()
