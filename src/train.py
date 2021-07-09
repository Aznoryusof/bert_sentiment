import os, sys

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(MAIN_DIR, "data/")
RESULT_DIR = os.path.join(MAIN_DIR, "result/")
sys.path.append(MAIN_DIR)

from src.data_processing.process_data import process_data
from src.data_processing.build_tensor import *
from src.data_processing.build_tensor_evaluate import *
from src.model.bert_training_loop import *
from src.utils.plot_training import plot_training
from src.utils.gpu_setup import gpu_setup
from config import epochs, ATTENTION_PROB_DROPOUT_PROB, HIDDEN_DROPOUT_PROB

from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import torch


def _training_setup(data_dict):
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels = 2,
        output_attentions = False,
        output_hidden_states = False,
        attention_probs_dropout_prob=ATTENTION_PROB_DROPOUT_PROB,
        hidden_dropout_prob=HIDDEN_DROPOUT_PROB
    )

    model.cuda()

    optimizer = AdamW(
        model.parameters(),
        lr = 1e-5,
        eps = 1e-8
    )

    total_steps = len(data_dict["train_dataloader"]) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = 0,
        num_training_steps = total_steps
    )

    return {
        "model": model,
        "scheduler": scheduler,
        "optimizer": optimizer
    }


def _save_model(model, tokenizer):
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    print("Saving model to %s" % RESULT_DIR)

    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(RESULT_DIR)
    tokenizer.save_pretrained(RESULT_DIR)
    

def train():
    device = gpu_setup()
    df_train = pd.read_csv("data/Train.csv")
    data_length = len(df_train)
    df_processed = process_data(df_train, data_length)
    data_dict = build_tensor(df_processed)
    setup_dict = _training_setup(data_dict)
    results_dict = training_loop(
            epochs, setup_dict["model"], data_dict["train_dataloader"], 
            data_dict["validation_dataloader"], setup_dict["optimizer"], 
            setup_dict["scheduler"], device
        )
    plot_training(results_dict["loss_values"])
    _save_model(results_dict["model_trained"], data_dict["tokenizer"])
    

if __name__ == "__main__":
    train()