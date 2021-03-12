import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):

    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, on='id', how='left')

    return df

def clean_data(df):

    # create a dataframe of the 36 individual category columns
    categories = pd.DataFrame(columns=['' for x in range(36)])
    # select the first row of the categories dataframe
    row = df.loc[0, 'categories']
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = [x[:-2] for x in row.split(';')]
    # rename the columns of `categories`
    categories.columns = category_colnames

    for number, column in enumerate(categories):
        # set each value to be the last character of the string
        categories[column] = df.categories.astype(str).apply(\
                                            lambda x: x.split(';')[number][-1])
        # convert column from string to numeric
        categories[column] = categories[column].apply(int)

    # Some cleaning in case a value is differente from 0 or 1
    categories = (categories>0).astype(int)
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    # drop duplicates
    df = df.drop_duplicates()
    df = df.drop_duplicates('id', keep='last')

    return df

def save_data(df, database_filename):

    final_path = 'sqlite:///' + database_filename
    engine = create_engine(final_path)
    df.to_sql(database_filename[:-3], engine, index=False)


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
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
