import os, sys

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(MAIN_DIR)

import numpy as np
import time
import datetime
import random
import torch
from config import seed
from utils.model_utilities import format_time


def _flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def training_loop(
    epochs, 
    model, 
    train_dataloader, 
    validation_dataloader, 
    optimizer, 
    scheduler,
    device
):
    
    # Set seed 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # The average loss after each epoch
    loss_values = []

    # For each epoch...
    for epoch_i in range(0, epochs):
        
        # Train over one full pass over the training data
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Log initial time
        t0 = time.time()

        # Reset loss for current epoch
        total_loss = 0

        # Set model mode to train
        model.train()

        # Perform the training for each batch
        for step, batch in enumerate(train_dataloader):

            # Show update every 20 batches
            if step % 20 == 0 and not step == 0:
                
                # Format the elapse time
                elapsed_time = format_time(time.time() - t0)
                
                # Print out the training progress
                print('  Batch {:>5,}  of  {:>5,}   Elapsed: {:}.'.format(step, len(train_dataloader), elapsed_time))

            # Extract current training batch from the dataloader
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Clear previously calculated gradients
            model.zero_grad()        

            # Forward pass
            outputs = model(
                b_input_ids.long(),
                token_type_ids=None, 
                attention_mask=b_input_mask.long(), 
                labels=b_labels.long()
            )
            
            # Extract the loss
            loss = outputs[0]
            
            # Accumulate the loss for current batch
            total_loss += loss.item()

            # Backward pass
            loss.backward()

            # Perform gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update the parameters
            optimizer.step()

            # Update the learning rate
            scheduler.step()

        # Take the average of the training loss over the dataset
        avg_train_loss = total_loss / len(train_dataloader)            
        
        # Store the loss values
        loss_values.append(avg_train_loss)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(format_time(time.time() - t0)))
        
        # Test on the validation set after each epoch
        print("")
        print("Running Validation...")
        
        t0 = time.time()
        
        # Set model to evaluation mode
        model.eval()

        # Reset variables 
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data after current epoch
        for batch in validation_dataloader:
            
            # Set move batch to GPU
            batch = tuple(t.to(device) for t in batch)
            
            # Unpack the inputs from the dataloader
            b_input_ids, b_input_mask, b_labels = batch
            
            # No need to compute or store gradients for validation
            with torch.no_grad():
            
                # Make predictions
                outputs = model(b_input_ids.long(), 
                                token_type_ids=None, 
                                attention_mask=b_input_mask.long())
            
            # Extract the scores from the model outputs
            logits = outputs[0]
            
            # Evaluate accuracy
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
                       
            tmp_eval_accuracy = _flat_accuracy(logits, label_ids)
            
            # Accumulate the total accuracy
            eval_accuracy += tmp_eval_accuracy

            # Track the number of batches
            nb_eval_steps += 1
            
        # Report the final accuracy for this validation run.
        print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
        print("  Validation took: {:}".format(format_time(time.time() - t0)))

    print("")
    print("Training complete!")

    return {
        "loss_values": loss_values,
        "model_trained": model,
    }