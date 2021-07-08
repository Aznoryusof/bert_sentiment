import os, sys

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(MAIN_DIR, "data/")
sys.path.append(MAIN_DIR)

import pandas as pd
import numpy as np
import random
import torch
from config import START_LEN, END_LEN, seed

# Set seed 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def _custom_truncate(text, start_len, end_len):

    no_words = len(text.split())
    text_front = " ".join(text.split()[:3])
    text_end = " ".join(text.split()[(no_words-end_len):no_words + 1])
    
    text_truncated = text_front + " " + text_end
    
    return text_truncated


def clean_data(df):
    df_clean = df.copy()
    df_clean["Text"] = df_clean["Text"].str.lower()
    #df_clean["Text"] = df_clean["Text"].str.replace("<br />", " ")
    df_clean["Text"] = df_clean["Text"].apply(_custom_truncate, start_len=START_LEN, end_len=END_LEN)  

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