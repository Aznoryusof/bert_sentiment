import os, sys

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULT_DIR = os.path.join(MAIN_DIR, "result/")
sys.path.append(MAIN_DIR)

import numpy as np
import torch
import textwrap
from transformers import BertForSequenceClassification, BertTokenizer
from src.utils.gpu_setup import gpu_setup
from config import use_gpu_predict, prediction_text, MAX_LEN

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


def _load_model_artifacts_cpu():
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


def _predict_string(string, model_dict, device):

    encoded_text = model_dict["tokenizer"].encode_plus(
        string,
        max_length=MAX_LEN,
        add_special_tokens=True,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoded_text['input_ids'].to(device)
    attention_mask = encoded_text['attention_mask'].to(device)
    output = model_dict["model"](input_ids, attention_mask)
    prediction = np.argmax(output[0].detach().cpu().numpy())
    sentiment = class_names[prediction]
    logit = output[0].detach().cpu().numpy()

    return {
        "sentiment": sentiment,
        "logit": logit
    }


def _predict_string_cpu(string, model_dict):

    encoded_text = model_dict["tokenizer"].encode_plus(
        string,
        max_length=MAX_LEN,
        add_special_tokens=True,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoded_text['input_ids']
    attention_mask = encoded_text['attention_mask']
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
        if use_gpu_predict & torch.cuda.is_available():
            device = gpu_setup()
            model_dict = _load_model_artifacts(device)
            prediction = _predict_string(string, model_dict, device)
            
            return prediction
        else:
            model_dict = _load_model_artifacts_cpu()
            prediction = _predict_string_cpu(string, model_dict)

            return prediction

    else:
        print("Train model before running predictions.")


if __name__ == "__main__":
    test_string = prediction_text
    prediction = predict(test_string)
    print("\nTest predictions on text:\n")
    print(wrapper.fill(test_string))
    print("\nSentiment: {}".format(prediction["sentiment"]))
    print("Logit: {}".format(prediction["logit"]))
