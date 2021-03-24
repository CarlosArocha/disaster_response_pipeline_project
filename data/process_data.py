import sys
import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    This function return the new dataset from merging the message and category
    datasets, found in the addresses received.

    Function parameters:

        Required:

            messages_filepath : The address to find and access the messages
                                data file.

            categories_filepath : The address to find and access the categories
                                data file.

        Return:

            df : pd.Dataframe.
                the merged dataset with both files data.

    '''

    # loading messages dataset
    messages = pd.read_csv(messages_filepath)
    # loading categories dataset
    categories = pd.read_csv(categories_filepath)
    # merging datasets
    df = messages.merge(categories, on='id', how='left')

    return df

def Oversampling_Imbalanced_Data(data, target_col, factor=2, top_data=0.05,
                                                                replace=True):
    '''
    This function return the new dataset taking adding a sample of the existing
    undersampled data.

    Function parameters:

        Required:

            data : The original dataset to be oversampled.

            target_col : list ;
                the columns of the featured data or categories to be checked.

        Optional:

            factor : float; default = 2 (2x), example: 0.0 (0x) - 10.0 (10x)
                the factor that multiplies the number of rows to be repeated.

            top_data : float; default = 0.05 (5%), example: 0.0 (0%) - 0.1 (10%)
                the percentage of data represented that will include the
                features to be repeated.

            replace : True or False;
                This determine if wants to repeat some of the data already
                sampled. If you don't want to repeat data more than once then
                the factor can't be more than 1.

        Return:

            new_data : pd.Dataframe.
                the new dataset oversampled.

    '''

    new_data = data.copy()
    # Let's walk through every feature or category
    for col in target_col:
        # taking a sub dataframe where all rows with this column is '1'
        d_base = data[data[col] == 1]
        # If there is data and the number of rows is below the top_data value:
        if (len(d_base) > 0) and ((len(d_base)/len(data))<top_data):
            # Let's take the samples and add them to the new dataframe
            d_samples = d_base.sample(n = int(factor*len(d_base)), \
                                      replace=replace)
            new_data = pd.concat([new_data, d_samples], ignore_index=True)

    # Returning the new oversampled dataset.
    return new_data

def Undersampling_Imbalanced_Related_Data(data, target_col, factor=0.5):
    '''
    This function return the new dataset removing a sample of the existing data
    of features than only have values in their columns alone, and '0' in the
    others.

    Function parameters:

        Required:

            data : The original dataset to be undersampled.

            target_col : list ;
                the columns of the featured data or categories to be checked.

        Optional:

            factor : float; default = 0.5 (50%), example: 0.0 (0%) - 1.0 (100%)
                the factor that represent the percentage been elminated from the
                existing data.

        Return:

            new_data : pd.Dataframe.
                the new dataset undersampled.

    '''

    new_data = data.copy()
    # calculating the number of rows to delete in the rows that only have one
    # value. In this case they are only the related and not_related columns.
    l = int(len(new_data[new_data[target_col].sum(axis=1) == 1]) * factor)
    # Taking a sample of that data.
    d_samples = new_data[new_data[target_col].sum(axis=1) == 1].sample(n = l, \
                                                                  replace=False)
    # Finally we create new_data dataset from the data without these rows, and
    # the sample selected.
    new_data = pd.concat([new_data[new_data[target_col].sum(axis=1) > 1],
                          d_samples], ignore_index=True)

    return new_data

def clean_data(df):
    '''
    In this function we will be preparing the dataset received. Interpreting the
    categories data and loading the dummies correspondent columns, cleaning it
    from duplicates, and from non 0-1 values, and finally doing some
    oversampling and undersampling.

    Function parameters:

        Required:

            df : pd.Dataframe.
                The raw dataset to be prepared.

        Return:

            df : pd.Dataframe.
                the cleaned and prepared dataframe.

    '''
    # creating a dataframe of the 36 individual category columns
    categories = pd.DataFrame(columns=['' for x in range(36)])
    # now selecting the first row of the categories dataframe
    row = df.loc[0, 'categories']
    # then we use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = [x[:-2] for x in row.split(';')]
    # renaming the columns of `categories`
    categories.columns = category_colnames
    # Let's iterate for each column data:
    for number, column in enumerate(categories):
        # set each value to be the last character of the string
        categories[column] = df.categories.astype(str).apply(\
                                            lambda x: x.split(';')[number][-1])
        # convert column from string to numeric
        categories[column] = categories[column].apply(int)

    # Some cleaning in case a value is different from 0 or 1
    categories = (categories>0).astype(int)
    # Duplicate the 'related' column to avoid eliminate those rows with 0 values
    # the duplicated new column will have '1' where 'related' is '0'
    # that warantees some issues fixed in the ML step because we are avoiding
    # rows full of '0's.
    categories['not_related'] = categories['related'].apply(lambda x: [1,0][x])
    # For ML training purposes, let's erase the columns with '0' values, this
    # could bring issues in some ML algorithms.
    for col in categories.columns:
        if categories[col].sum() == 0:
            categories.drop(col, axis=1, inplace=True)
    # dropping the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    # now concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    # drop duplicates
    df = df.drop_duplicates()
    df = df.drop_duplicates('id', keep='last')
    # Let's do oversampling of the data that is below the 3,5% of the
    # representation, and just add 70% more of that data without repeating.
    df = Oversampling_Imbalanced_Data(df, categories.columns,
                                      factor=0.7, top_data=0.035,
                                      replace=False)
    # And finally some undersampling of the related features.
    # Let's reduce it 60%
    df = Undersampling_Imbalanced_Related_Data(df, categories.columns, 0.4)

    return df

def save_data(df, database_filename):
    '''
    This function just creates the SQLite database and table, from the dataframe
    parameter and saving it into the address received.

    Function parameters:

        Required:

            df : pd.Dataframe.
                    The dataframe to be transformed and saved in the location
                    specified.

            database_filepath : The location where the new table will be saved.

    '''

    # Declaring the final path for sqlite engine purposes
    final_path = 'sqlite:///' + database_filename
    # Creating the engine and saving the new table.
    engine = create_engine(final_path)
    df.to_sql('DisasterResponse_table', engine, index=False)

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
