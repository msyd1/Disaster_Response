''' This script creates a database based on new datasets specified by the user.
It performs loading and cleaning data.

Syntax:
python process_data.py path_to_messages path_to_categories path_to_database

Example:
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
'''

# Importing libraries:
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    ''' This function loads datasets from pathes specified by user:
    Parameters:
        messages_filepath: path to dataset with messages
        categories_filepath: path to dataset with categories
    Returns:
        df: dataframe with data
    '''
    #Read .csv files and store them in variables:
    df_messages = pd.read_csv(messages_filepath)
    df_categories = pd.read_csv(categories_filepath)

    #Merge the messages and categories datasets using the common id
    df = df_messages.merge(df_categories, on='id', how='inner')
    return df

def clean_data(df):
    '''
    This function cleans dataset "df".
    Parameters:
        df: mergad dataframe
    Returns:
        df: cleaned data frame
    '''
    # Create a dataframe with 36 individual category columns:
    categories = df["categories"].str.split(';', expand=True)
    # first row:
    row = categories.iloc[0,:]
    # The first row is the source for a list of category names:
    category_colnames = ['category_'+col.split('-')[0] for col in row]
    # Rename columns of "category" dataframe using the previous list:
    categories.columns = category_colnames
    # Go through columns of "category" dataframe;
    for column in categories.columns:
        # Slice all values to the last digit:
        categories[column] = categories[column].str[-1]
        # Convert to integer:
        categories[column] = categories[column].astype(np.int)

    ###
    # There are some non-bolean values (2) in these columns;
    # I assumed that it should be 1 instead of 2 and made a replacement:
    categories = categories.replace(2, 1)
    # The original "category" column should be dropped from the dataset:
    df = df.drop('categories',axis=1)
    # Concatenate df and new categories dataframe
    df = pd.concat([df,categories],axis=1)
    # Remove duplicates
    df = df.drop_duplicates()

    return df

def save_data(df, database_filename):
    '''
    This function saves dataframe into an sqlite database.
    Parameters:
        df: cleaned dataset
        database_filename: name of the sqlite database, provided by user
    Returns:
        Sqlite file with database
    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterResponse', engine,if_exists = 'replace', index=False)  

#Check if the user provided 4 arguments:
def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        # Print a hint to the user
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
