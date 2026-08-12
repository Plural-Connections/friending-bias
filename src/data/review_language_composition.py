"""Language composition of GreatSchools reviews — what share are (partly) Spanish."""

import pandas as pd
from langdetect import detect_langs
from langdetect.detector_factory import DetectorFactory

DetectorFactory.seed = 0  # langdetect samples internally; fix the seed for reproducible results

reviews_path = "data/interim/gs_reviews_sorted_by_day_posted.csv"
TEXT_COL = "comments"


def detect_spanish_full_text(text):
    """
    Detect potential Spanish content using langdetect.
    Returns True if Spanish is detected with >5% probability.
    """
    if pd.isna(text) or not str(text).strip():
        return False

    try:
        languages = detect_langs(str(text))
        for lang in languages:
            if lang.lang == 'es' and lang.prob > 0.05:
                return True
    except:
        # If detection fails, return False
        return False

    return False


def run(reviews_path=reviews_path):
    df = pd.read_csv(reviews_path, usecols=[TEXT_COL])
    is_spanish = df[TEXT_COL].apply(detect_spanish_full_text)
    pct = is_spanish.mean() * 100
    print(f"n reviews = {len(df):,}")
    print(f"Spanish reviews = {is_spanish.sum():,} ({pct:.2f}%)")
    return pct


if __name__ == "__main__":
    run()
