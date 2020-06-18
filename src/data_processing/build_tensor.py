import os, sys

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(MAIN_DIR, "data/")
sys.path.append(MAIN_DIR)


import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from config import seed, MAX_LEN, batch_size
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


def _train_test_split(df, seed):
    df_shuffled = df.sample(frac=1).reset_index(drop=True)
    train_inputs, test_inputs, train_labels, test_labels = train_test_split(
        df_shuffled["review"], df_shuffled["label"], random_state=seed, test_size=0.3
    )

    return train_inputs, test_inputs, train_labels, test_labels


def _tokenize(complete_train_inputs):
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    input_ids = []
    lengths = []

    print('Tokenizing comments...')

    for sen in complete_train_inputs:
        if ((len(input_ids) % 20000) == 0):
            print('  Read {:,} comments.'.format(len(input_ids)))

        encoded_sent = tokenizer.encode(
            sen,
            add_special_tokens = True
        )

        input_ids.append(encoded_sent)
        lengths.append(len(encoded_sent))

    print('DONE.')
    print('{:>10,} comments'.format(len(input_ids)))

    return {
        "input_ids": input_ids, 
        "lengths": lengths,
        "tokenizer": tokenizer
    }


def _pad_sequences(input_ids, tokenizer):
    print('\nPadding/truncating all sentences to %d values...' % MAX_LEN)
    print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))
    
    input_ids_padded = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", 
                          value=0, truncating="post", padding="post")

    print('\nDone.')

    return input_ids_padded


def _attention_masks(input_ids):
    attention_masks = []

    for sent in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)

    return attention_masks


def _train_valid_split(input_ids, complete_train_labels, attention_masks):
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(
        input_ids, complete_train_labels, random_state=seed, test_size=0.1
    )

    train_masks, validation_masks, _, _ = train_test_split(
        attention_masks, complete_train_labels, random_state=seed, test_size=0.1
    )

    return {
        "train_inputs": torch.tensor(train_inputs), 
        "validation_inputs": torch.tensor(validation_inputs), 
        "train_labels": torch.tensor(train_labels),
        "validation_labels": torch.tensor(validation_labels), 
        "train_masks": torch.tensor(train_masks), 
        "validation_masks": torch.tensor(validation_masks)
    }


def _iterator(processed_data_dict):
    train_data = TensorDataset(
        processed_data_dict["train_inputs"], 
        processed_data_dict["train_masks"], 
        processed_data_dict["train_labels"]
    )
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(
        processed_data_dict["validation_inputs"], 
        processed_data_dict["validation_masks"], 
        processed_data_dict["validation_labels"]
    )
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    return train_dataloader, validation_dataloader


def build_tensor(data_path):
    """This function converts data of comments and labels from a csv file
    to Pytorch tensors for fine-tuning BERT Model.   

    Args:
        data_path (str): The path of the csv file to be 
    """    
    data = pd.read_csv(os.path.join(DATA_DIR, data_path))
    complete_train_inputs, test_inputs, complete_train_labels, test_labels = _train_test_split(data, seed)
    complete_train_labels = complete_train_labels.to_numpy().astype(int)

    tokenized_dict = _tokenize(complete_train_inputs)
    input_ids = _pad_sequences(tokenized_dict["input_ids"], tokenized_dict["tokenizer"])
    attention_masks = _attention_masks(input_ids)
    processed_data_dict = _train_valid_split(
        input_ids, complete_train_labels, attention_masks
    )
    train_dataloader, validation_dataloader = _iterator(processed_data_dict)

    return {
        "train_dataloader": train_dataloader,
        "validation_dataloader": validation_dataloader,
        "test_inputs": test_inputs,
        "test_labels": test_labels
    }


if __name__ == "__main__":
    build_tensor("hotel_reviews_processed.csv")