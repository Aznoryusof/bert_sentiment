import os, sys

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULT_DIR = os.path.join(MAIN_DIR, "result/")
sys.path.append(MAIN_DIR)

import numpy as np
import torch
import textwrap
from transformers import BertForSequenceClassification, BertTokenizer
from src.utils.gpu_setup import gpu_setup

from config import MAX_LEN


class_names = {
    0: "Negative",
    1: "Positive"
}

wrapper = textwrap.TextWrapper(
    initial_indent="    ", 
    subsequent_indent="    ", 
    width = 80
)


def _check_model_exist():
    if os.path.exists(RESULT_DIR):
        print("Loading model from {}".format(RESULT_DIR))
                
        return True
    else:
        return False


def _load_model_artifacts(device):
    print("\nLoading model artifacts...")

    vecs = np.load('result/embeddings.npy')
    model = BertForSequenceClassification.from_pretrained(
        RESULT_DIR,
        output_hidden_states = True,
    )

    tokenizer = BertTokenizer.from_pretrained(RESULT_DIR)
    model.to(device)

    return {
        "model": model,
        "vecs": vecs,
        "tokenizer": tokenizer
    }


def _predict_string(string, model_dict, device):

    encoded_review = model_dict["tokenizer"].encode_plus(
        string,
        max_length=128,
        add_special_tokens=True,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoded_review['input_ids'].to(device)
    attention_mask = encoded_review['attention_mask'].to(device)
    output = model_dict["model"](input_ids, attention_mask)
    prediction = np.argmax(output[0].detach().cpu().numpy())
    sentiment = class_names[prediction]
    logit = output[0].detach().cpu().numpy()

    return {
        "sentiment": sentiment,
        "logit": logit
    }


def predict(string):
    model_available = _check_model_exist()
    if model_available:
        device = gpu_setup()
        model_dict = _load_model_artifacts(device)
        prediction = _predict_string(string, model_dict, device)
        
        return prediction

    else:
        print("Train model before running predictions.")


if __name__ == "__main__":
    test_string = "Ive probably stayed there 5 weeks in the last few months. Had to extend my latest stay by another 8 days and asked if I could get a late checkout due to a late flight. Basically refused and didn't offer anything in recognition of how much I've stayed there recently. the manager there is an idiot. Have moved to the Y Hotel. Not much more and a thousand times better Hotel. i won't be back X Hotel. You blew it!"
    prediction = predict(test_string)
    print("\nTest predictions on text:\n")
    print(wrapper.fill(test_string))
    print("\nSentiment: {}".format(prediction["sentiment"]))
    print("Logit: {}".format(prediction["logit"]))