import os, sys

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(MAIN_DIR, "data/")
RESULT_DIR = os.path.join(MAIN_DIR, "result/")
sys.path.append(MAIN_DIR)

import time
import datetime
from transformers import BertForSequenceClassification, BertTokenizer


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def check_model_exist():
    if os.path.exists(RESULT_DIR):
        print("Loading model from {}".format(RESULT_DIR))
                
        return True
    else:
        return False
    

def load_model_artifacts(device):
    print("\nLoading model artifacts...")

    model = BertForSequenceClassification.from_pretrained(
        RESULT_DIR,
        output_hidden_states = True,
    )

    tokenizer = BertTokenizer.from_pretrained(RESULT_DIR)
    model.to(device)

    return {
        "model": model,
        "tokenizer": tokenizer
    }


def load_model_artifacts_cpu():
    print("\nLoading model artifacts...")

    model = BertForSequenceClassification.from_pretrained(
        RESULT_DIR,
        output_hidden_states = True,
    )

    tokenizer = BertTokenizer.from_pretrained(RESULT_DIR)

    return {
        "model": model,
        "tokenizer": tokenizer
    }
