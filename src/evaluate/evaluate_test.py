import os, sys
import numpy as np
import torch

from sklearn.metrics import roc_auc_score

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULT_DIR = os.path.join(MAIN_DIR, "result/")
sys.path.append(MAIN_DIR)


def evaluate(test_inputs):
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
    
        # Progress update every 100 batches.
        if step % 100 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(test_dataloader), elapsed))


        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
    
        # Telling the model not to compute or store gradients, saving memory and 
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None, 
                            attention_mask=b_input_mask)

        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
    
        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)

    print('    DONE.')

    predictions = np.concatenate(predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)

    print('The models output for the first 10 test samples are: {}'.format(predictions[0:10]))
    print('The corresponding correct labels are: {}'.format(true_labels[0:10]))

    # Use the model output for label 1 as our predictions.
    p1 = predictions[:,1]

    # Calculate the ROC AUC.
    auc = roc_auc_score(true_labels, p1)

    print('Test ROC AUC: %.3f' %auc)

    return {
        "predictions": predictions,
        "true_labels": true_labels,
        "auc": auc
    }
