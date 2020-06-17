import os, sys
import pandas as pd

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(MAIN_DIR)

from src.process_data import _extract_pos_neg


df = pd.DataFrame({
    "Positive_Review": ["This is sucha great room", "Delightful ambience"],
    "Negative_Review": ["Yucks, so dirty", "Can't wait for my holiday to end. Disgusting"],
    "Reviewer_Score": [5, 6],
    "Positive_Review_Word_Count": [5, 2],
    "Negative_Review_Word_Count": [3, 8]
})

Expected_df = pd.DataFrame({
    "review": ["This is sucha great room", "Can't wait for my holiday to end. Disgusting"],
    "label": [1, 0]
})

minimum_positive_reviewer_score = 1
minimum_positive_word_count = 3
maximum_negative_reviewer_score = 10 
minimum_negative_word_count = 5


def test_extract_pos_neg():
    result_df = _extract_pos_neg(
        df, minimum_positive_reviewer_score, minimum_positive_word_count, 
        maximum_negative_reviewer_score, minimum_negative_word_count
    )

    assert result_df.equals(Expected_df)
