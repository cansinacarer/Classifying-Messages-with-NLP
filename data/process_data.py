import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Parameters:
        messages_filepath: Path to the CSV file containing messages
        categories_filepath: Path to the CSV file containing categories
    Returns:
        df: read data in a df
    """
    # Extract: Read and merge the data files into a df
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on="id")
    return df


def clean_data(df):
    """
    Parameters:
        df: Combined data containing messages and categories
    Returns:
        df: clean data
    """
    # 1. Import libraries and load datasets
    # Transform: create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(pat=";", expand=True)

    # select the first row of the categories dataframe
    labels_row = categories.loc[0]

    # use this row to extract a list of new column names for categories.
    category_colnames = labels_row.apply(lambda x: x[:-2])

    # rename the columns of `categories`
    categories.columns = category_colnames

    # set each value to be the last character of the string
    categories = categories.applymap(lambda x: x[-1:])

    # convert column from string to numeric
    categories = categories.apply(pd.to_numeric)

    # drop the original categories column from `df`
    df.drop("categories", axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    # Drop the 188 rows that have 2 in the related column
    # This is assumed to be an error in data since we only expect 1s and 0s
    df = df[df.related != 2]

    return df


def save_data(df, database_filename):
    """
    Parameters:
        df: Cleaned data containing messages and categories
        database_filename: Path to SQLite file
    """
    engine = create_engine("sqlite:///" + database_filename)
    df.to_sql("Tweets", engine, index=False, if_exists="replace")


def main():
    if len(sys.argv) == 4:

        # Get passed arguments
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(
            f"Loading data...\n    MESSAGES: {messages_filepath}\n    CATEGORIES: {categories_filepath}"
        )
        df = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        df = clean_data(df)

        print(f"Saving data...\n    DATABASE: {database_filepath}")
        save_data(df, database_filepath)

        print("Cleaned data saved to database!")

    else:
        print(
            "Please provide the filepaths of the messages and categories "
            "datasets as the first and second argument respectively, as "
            "well as the filepath of the database to save the cleaned data "
            "to as the third argument. \n\nExample: python process_data.py "
            "disaster_messages.csv disaster_categories.csv "
            "DisasterResponse.db"
        )


if __name__ == "__main__":
    main()
