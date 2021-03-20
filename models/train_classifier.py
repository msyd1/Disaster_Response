''' This script makes a pipeline for Random Forest classifier, which classifies
messages, and create pikle file as a result.

Syntax:
python train_classifier.py path_to_database name_of_pkl_file

Example:
python train_classifier.py ../data/DisasterResponse.db classifier.pkl
'''

import sys
import os
import sqlalchemy
import sqlite3
from sqlalchemy import create_engine
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline


from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import warnings
warnings.filterwarnings('ignore')

import pickle

def load_data(database_filepath):
    ''' This function loads database from pathes specified by user:
    Parameters:
        database_filepath: path to database with messages
    Returns:
        X, Y, category_names: dataset, splited for training
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    table_name = os.path.basename(database_filepath).replace(".db","")
    df = pd.read_sql_table(table_name, engine)
    X = df.message
    Y = df[df.columns[4:]]
    category_names = Y.columns
    return X, Y, category_names 


def tokenize(text):
    ''' This function tokenizes text messages.
    Parameters:
        text: text of message
    Returns:
        tokens(list): tokenised text 
    '''
    message = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = []
    for i in message:
        token = lemmatizer.lemmatize(i).lower().strip()
        tokens.append(token)
    return tokens

def build_model():
    ''' This function creates a pipeline for a prediction model.
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    # It takes approximately 5 min for training model with these parameters. 
    parameters ={
        'clf__estimator__n_estimators': [5, 10],
        'clf__estimator__min_samples_split': [2, 3],
        'clf__estimator__criterion': ["gini", "entropy"],
        }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=4, verbose=2, cv=3)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    This function performs evaluation of model score
    '''
    prediction = model.predict(X_test)
    report = classification_report(Y_test, prediction, target_names=category_names)
    print(report)

def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


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
