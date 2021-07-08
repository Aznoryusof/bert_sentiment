import os, sys, argparse

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(MAIN_DIR, "data/")
RESULT_DIR = os.path.join(MAIN_DIR, "result/")
sys.path.append(MAIN_DIR)

import time
import torch
import datetime
import pandas as pd
import numpy as np
from tabulate import tabulate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import BertForSequenceClassification, BertTokenizer
from process_data import clean_data
from data_processing.build_tensor_evaluate import build_tensor_evaluate
from src.utils.gpu_setup import gpu_setup

from config import MAX_LEN


def _format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def _predict(test_dataloader, model, device):
    # Put model in evaluation mode
    model.eval()

    # Tracking variables 
    predictions , true_labels = [], []

    # Measure elapsed time.
    t0 = time.time()

    # Predict 
    for (step, batch) in enumerate(test_dataloader):
        
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
    
        # Progress update every 20 batches.
        if step % 20 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = _format_time(time.time() - t0)
            
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(test_dataloader), elapsed))


        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
    
        # Telling the model not to compute or store gradients, saving memory and 
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids.long(), token_type_ids=None, 
                            attention_mask=b_input_mask.long())

        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
    
        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)

    print('    DONE.')

    predictions = np.concatenate(predictions, axis=0)
    prediction_labels = np.argmax(predictions, axis=1).flatten()
    true_labels = np.concatenate(true_labels, axis=0)   

    return {
        "prediction_labels": prediction_labels,
        "true_labels": true_labels
    }


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


def _validate(path):
    df_pred = pd.read_csv(path)
    
    # Process Text data
    df_clean = clean_data(df_pred) 
    
    # Load the model artifacts
    device = gpu_setup()
    model_dict = _load_model_artifacts(device)
    
    # Prepare Text data for modelling
    data_dict = {
        "test_inputs": df_clean["Text"],
        "test_labels": df_clean["Label"],
        "tokenizer": model_dict["tokenizer"]
    }
        
    # Build the tensors for validation
    data_test_dict = build_tensor_evaluate(data_dict)
    
    # Perform test on validation data
    evaluated_dict = _predict(data_test_dict["test_dataloader"], model_dict["model"], device)
    
    # negative sentiment used as positive label because for sentiment analysis, it is arguably more important to detect negative sentiments.
    accuracy = accuracy_score(evaluated_dict["true_labels"], evaluated_dict["prediction_labels"]) * 100
    precision = precision_score(evaluated_dict["true_labels"], evaluated_dict["prediction_labels"], pos_label=0) * 100
    recall = recall_score(evaluated_dict["true_labels"], evaluated_dict["prediction_labels"], pos_label=0) * 100
    f1 = f1_score(evaluated_dict["true_labels"], evaluated_dict["prediction_labels"], pos_label=0) * 100
    
    return {
        "accuracy": str(round(accuracy, 2)) + "%", 
        "precision": str(round(precision, 2)) + "%", 
        "recall": str(round(recall, 2)) + "%", 
        "f1": str(round(f1, 2)) + "%"
    }


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Performs sentiment prediction on the Text in the input validation file.")
    
    # Add the arguments for the parser
    parser.add_argument("file_path", metavar="path", type=str, help="path to the test file")
    
    # Execute the parse_args() method
    args = parser.parse_args()
    
    # Make the predictions
    print("\nPerforming validation for Text in file: {}".format(args.file_path))
    scores_dict = _validate(args.file_path)
       
    # Print the results
    results = [[scores_dict["accuracy"], scores_dict["precision"], scores_dict["recall"], scores_dict["f1"]]]
    print("{}".format("="*50))
    print(tabulate(results, headers=["accuracy", "precision-neg", "recall-neg", "f1-neg"])) 
    print("{}".format("="*50))