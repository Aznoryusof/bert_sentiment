import pandas as pd
import os, sys
import textwrap


MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(MAIN_DIR, "data/")
sys.path.append(MAIN_DIR)

from config import *


def _extract_positive(df, minimum_positive_reviewer_score, minimum_positive_word_count):
    pos_extracted_df = df[["Positive_Review", "Positive_Review_Word_Count", "Reviewer_Score"]]
    pos_extracted_df = pos_extracted_df[
        (pos_extracted_df["Reviewer_Score"] >= minimum_positive_reviewer_score) &
        (pos_extracted_df["Positive_Review_Word_Count"] > minimum_positive_word_count)
    ]

    pos_extracted_df["label"] = 1
    pos_extracted_df["review"] = pos_extracted_df["Positive_Review"] 

    return pos_extracted_df[["review", "label"]]


def _extract_negative(df, maximum_negative_reviewer_score, minimum_negative_word_count):
    neg_extracted_df = df[["Negative_Review", "Negative_Review_Word_Count", "Reviewer_Score"]]
    neg_extracted_df = neg_extracted_df[
        (neg_extracted_df["Reviewer_Score"] <= maximum_negative_reviewer_score) &
        (neg_extracted_df["Negative_Review_Word_Count"] > minimum_negative_word_count)
    ]

    neg_extracted_df["label"] = 0
    neg_extracted_df["review"] = neg_extracted_df["Negative_Review"] 

    return neg_extracted_df[["review", "label"]]


def _count_words(df, col_name):
    word_count_list = [len(comment.split()) for comment in df[col_name]]
    word_count_series = pd.Series(word_count_list) 

    return word_count_series


def _clean_data(df):
    keep_col_names = ["Positive_Review", "Negative_Review", "Reviewer_Score"]
    data_clean = df[keep_col_names]
    data_clean["Positive_Review_Word_Count"] = _count_words(data_clean, "Positive_Review")
    data_clean["Negative_Review_Word_Count"] = _count_words(data_clean, "Negative_Review")

    return data_clean


def _extract_pos_neg(df, minimum_positive_reviewer_score, minimum_positive_word_count, maximum_negative_reviewer_score, minimum_negative_word_count):
    pos_reviews_df = _extract_positive(df, minimum_positive_reviewer_score, minimum_positive_word_count)
    neg_reviews_df = _extract_negative(df, maximum_negative_reviewer_score, minimum_negative_word_count)
    data_extracted = pos_reviews_df.append(neg_reviews_df, ignore_index=True)

    return data_extracted


def _save_preprocessed_data(df):
    files = [
        (
            df, "hotel_reviews_processed.csv"
        )
    ]

    for data, filename in files:
        if not os.path.exists(os.path.join(DATA_DIR, filename)):
            print("Saving to", DATA_DIR)

            data.to_csv(
                os.path.join(DATA_DIR, filename),
                index = False
            )

        else:
            print("Processed data already saved in ", DATA_DIR)


def process_data():
    data = pd.read_csv("data/Hotel_Reviews.csv")
    data_clean = _clean_data(data)
    data_extracted = _extract_pos_neg(
        data_clean, minimum_positive_reviewer_score, minimum_positive_word_count,
        maximum_negative_reviewer_score, minimum_negative_word_count
    )

    _save_preprocessed_data(data_extracted)
    

if __name__ == "__main__":
    process_data()