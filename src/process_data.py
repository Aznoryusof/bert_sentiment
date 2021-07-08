import pandas as pd
import os, sys


MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(MAIN_DIR, "data/")
sys.path.append(MAIN_DIR)

from config import *


def clean_data(df):
    df_clean = df.copy()
    df_clean["Text"] = df_clean["Text"].str.lower()
    #df_clean["Text"] = df_clean["Text"].str.replace("<br />", " ")

    return df_clean


def _save_preprocessed_data(df):
    files = [
        (
            df, "Train_processed.csv"
        )
    ]

    for data, filename in files:
        if not os.path.exists(os.path.join(DATA_DIR, filename)):
            print("Saving to", DATA_DIR)

            data.to_csv(
                os.path.join(DATA_DIR, filename),
                index = False
            )

        else:
            print("Processed data already saved in ", DATA_DIR)


def _sample(df, data_size):
    df_final = df.sample(frac=1).head(data_size)

    return df_final


def process_data(data_size):
    data = pd.read_csv("data/Train.csv")
    data_clean = clean_data(data)
    data_sampled = _sample(data_clean, data_size)
    _save_preprocessed_data(data_sampled)
    

if __name__ == "__main__":
    data = pd.read_csv("data/Train.csv")
    data_length = len(data)
    process_data(data_length)