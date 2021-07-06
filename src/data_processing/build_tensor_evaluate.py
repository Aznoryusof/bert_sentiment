import os, sys

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(MAIN_DIR, "data/")
sys.path.append(MAIN_DIR)

import pandas as pd
import numpy as np
import torch
from config import seed, MAX_LEN, batch_size
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


def build_tensor_evaluate(data_dict):
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    test_input_ids = []

    # For every sentence...
    for sen in data_dict["test_inputs"]:
        
        # Report progress.
        if ((len(test_input_ids) % 20000) == 0):
            print('  Read {:,} comments.'.format(len(test_input_ids)))
        
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_sent = data_dict["tokenizer"].encode(
                            sen,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = MAX_LEN,          # Truncate all sentences.                        
                    )
        
        # Add the encoded sentence to the list.
        test_input_ids.append(encoded_sent)

    print('DONE.')
    print('')
    print('{:>10,} test comments'.format(len(test_input_ids)))

    # Also retrieve the labels as a list.

    # Get the labels from the DataFrame, and convert from booleans to ints.
    test_labels = data_dict["test_labels"].to_numpy().astype(int)

    print('{:>10,} positive (contains attack)'.format(np.sum(test_labels)))
    print('{:>10,} negative (not an attack)'.format(len(test_labels) - np.sum(test_labels)))

    # Pad our input tokens
    test_input_ids = pad_sequences(test_input_ids, maxlen=MAX_LEN, 
                                dtype="long", truncating="post", padding="post")

    # Create attention masks
    test_attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in test_input_ids:
        seq_mask = [float(i>0) for i in seq]
        test_attention_masks.append(seq_mask) 

    # Convert to tensors.
    test_inputs = torch.tensor(test_input_ids)
    test_masks = torch.tensor(test_attention_masks)
    test_labels = torch.tensor(test_labels)

    # Set the batch size.  
    batch_size = batch_size

    # Create the DataLoader.
    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    return {
        "test_data": test_data,
        "test_sampler": test_sampler,
        "test_dataloader": test_dataloader
    }