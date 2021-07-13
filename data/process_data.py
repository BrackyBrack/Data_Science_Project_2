import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Reads data from 2 csv files and returns a merged Pandas Dataframe using 'id' columns in both csv's

    Parameters:
    messages_filepath -- filepath of a csv containing messages
    categories_filepath -- filepath of a csv containing categories of the messages from messages_filepath
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return messages.merge(categories, how='outer', on='id')

def clean_data(df):
    """
    Splits the categories column into individual appropriately names columns

    Parameters:
    df - Pandas dataframe to be cleaned
    """

    # Replace any categorisation that have been marked as 2 with 1. 
    df['categories'] = df['categories'].str.replace('-2', '-1')

    # Create columns for each category and name them appropriately
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = [col[0:-2] for col in row]
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        df = pd.concat([df, categories], axis=1)

    # Replace existing categories column with created columns
    df.drop(columns='categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    
    return df.drop_duplicates()
    


def save_data(df, database_filename):
    """
    Saves a pandas dataframe to CategorisedMessages table on a SQLite database

    Parameters:
    df -- Pandas dataframe to be saved
    database_filename -- name of database file to be created
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('CategorisedMessages', engine, index=False, if_exists='replace')  


def main():
    """
    main function. Reads data from a CSV, cleans the data and stores to an SQLite database. Takes filepaths for 2 csv files and
    a database name as args
    """
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
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()