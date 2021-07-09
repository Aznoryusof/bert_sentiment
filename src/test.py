import os, sys, argparse

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(MAIN_DIR, "data/")
RESULT_DIR = os.path.join(MAIN_DIR, "result/")
sys.path.append(MAIN_DIR)

import time
import torch
import pandas as pd
import numpy as np
from tabulate import tabulate
from transformers import BertForSequenceClassification, BertTokenizer
from data_processing.process_data import clean_data
from data_processing.build_tensor_evaluate import build_tensor_evaluate
from src.utils.gpu_setup import gpu_setup
from utils.model_utilities import format_time, check_model_exist, load_model_artifacts, load_model_artifacts_cpu

from config import use_gpu_test, MAX_LEN

# Set seed 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


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

    return prediction


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

    return prediction


def _predict_test(path):
    # Convert to df to use data cleaning pipeline
    df_test = pd.DataFrame(
        {
            "Text": pd.read_csv(path, header=None, squeeze=True)
        }
    )

    # Process Text data
    test_list = clean_data(df_test)["Text"].tolist()
    
    # Load the model artifacts and perform predictions
    test_pred_list = []
    t0 = time.time()
    counter=0
    
    model_available = check_model_exist()
    if model_available:
        
        print()
        if use_gpu_test & torch.cuda.is_available():
            device = gpu_setup()
            model_dict = load_model_artifacts(device)
            
            # Perform the predictions using GPU
            for text in test_list:
                prediction = _predict_string(text, model_dict, device)
                test_pred_list.append(prediction)
                counter+=1
                
                # Progress update every 100 sample
                if counter % 100 == 0 and not counter == 0:
                    # Calculate elapsed time
                    elapsed = format_time(time.time() - t0)
                    # Report progress
                    print('  Text {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(counter, len(test_list), elapsed))
                
        else:
            model_dict = load_model_artifacts_cpu()
            
            # Perform the predictions using CPU
            for text in test_list:
                prediction = _predict_string_cpu(text, model_dict)
                test_pred_list.append(prediction)
                counter+=1
                
                # Progress update every 100 sample
                if counter % 100 == 0 and not counter == 0:
                    # Calculate elapsed time
                    elapsed = format_time(time.time() - t0)
                    # Report progress
                    print('  Text {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(counter, len(test_list), elapsed))
    
    return {
        "test_list": test_list,
        "test_pred_list": test_pred_list
    }


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Performs sentiment prediction on the input test file.")
    
    # Add the arguments for the parser
    parser.add_argument("file_path", metavar="path", type=str, help="path to the test file")
    
    # Execute the parse_args() method
    args = parser.parse_args()
    
    # Make the predictions
    print("\nPerforming predictions on test data: {}".format(args.file_path))
    test_pred_dict = _predict_test(args.file_path)

    # Save results
    df_pred = pd.DataFrame(
        {
            "Text": test_pred_dict["test_list"],
            "test_pred_list": test_pred_dict["test_pred_list"]
        }
    )
    df_pred.to_csv(os.path.splitext(args.file_path)[0] + "_pred.csv", header=False, index=False)
    
    
    