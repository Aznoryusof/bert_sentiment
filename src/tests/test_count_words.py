import os, sys
import pandas as pd

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(MAIN_DIR)

from src.process_data import _count_words


df = pd.DataFrame({
    "Positive_Review": ["This is sucha great room", "Delightful ambience"],
    "Negative_Review": ["Yucks, so dirty", "Can't wait for my holiday to end. Disgusting"]
})


def test_count_words():
    assert _count_words(df, "Negative_Review").equals(pd.Series([3, 8]))
    assert _count_words(df, "Positive_Review").equals(pd.Series([5, 2]))
