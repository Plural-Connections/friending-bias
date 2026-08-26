"""CONCAT, FILTERED: Same as topics_distro_BERT_concat.py, but over the
nces-filtered concatenated reviews file
(data/processed/gs_reviews_concat_by_school_nces_filtered.csv) — only the
11,859 schools that have a matching nces_id (and therefore a friending-bias
score).

Run from the project root as a module:
    python -m src.bertopic.unsupervised_exploration.topics_distro_BERT_concat_filtered
"""

import os

import pandas as pd
from bertopic import BERTopic

concat_path = "data/processed/gs_reviews_concat_by_school_nces_filtered.csv"
model_dir = "outputs/models/bertopic/bertopic_model_50_min_cluster_filtered"
topic_summary_path = "outputs/bertopic/unsupervised_exploration/summary/topic_summary_50_min_cluster_filtered.csv"
output_path = "outputs/bertopic/unsupervised_exploration/distributions/topic_distributions_50_min_cluster_by_school_concat_filtered.csv"

TEXT_COL = "comments"
ID_COL = "universal-id"
NCES_ID_COL = "nces_id"
N_REVIEWS_COL = "n_reviews"
N_WORDS_COL = "n_words"

# window/stride/batch_size: same reasoning as topics_distro_BERT_concat.py.
WINDOW = 8
STRIDE = 4
BATCH_SIZE = 200


def load_school_docs(path=concat_path):
    """Read the school-concatenated review file and drop empty blobs."""
    df = pd.read_csv(
        path,
        usecols=[ID_COL, NCES_ID_COL, TEXT_COL, N_REVIEWS_COL, N_WORDS_COL],
        dtype={NCES_ID_COL: str},
    )
    df = df.dropna(subset=[TEXT_COL])
    df[TEXT_COL] = df[TEXT_COL].astype(str)
    df = df[df[TEXT_COL].str.strip().str.len() > 0]
    return df.reset_index(drop=True)


def all_topic_ids(path=topic_summary_path):
    """Every real topic (excludes the noise topic -1), sorted by size, so
    column order still reflects topic size even though nothing is dropped."""
    summary = pd.read_csv(path)
    summary = summary[summary["topic_id"] != -1].sort_values("size", ascending=False)
    return summary["topic_id"].tolist(), summary.set_index("topic_id")["keywords"]


def run():
    os.makedirs("outputs/bertopic/unsupervised_exploration/distributions", exist_ok=True)

    print(f"Loading fitted BERTopic model from {model_dir} ...")
    model = BERTopic.load(model_dir)

    meta = load_school_docs()
    docs = meta[TEXT_COL].tolist()
    print(f"n schools = {len(docs):,}")

    topic_ids, keywords = all_topic_ids()
    print(f"Using all {len(topic_ids)} topics (topic_id {min(topic_ids)}-{max(topic_ids)})")

    topic_distr, _ = model.approximate_distribution(
        docs,
        window=WINDOW,
        stride=STRIDE,
        batch_size=BATCH_SIZE,
    )

    dist_df = pd.DataFrame(
        topic_distr[:, topic_ids],
        columns=[f"topic_{t}" for t in topic_ids],
    )

    out = pd.concat(
        [meta[[ID_COL, NCES_ID_COL, N_REVIEWS_COL, N_WORDS_COL]], dist_df],
        axis=1,
    )
    out.to_csv(output_path, index=False)

    print(f"\nWrote {output_path}  ({out.shape[0]:,} schools x {len(topic_ids)} topics)")
    print("\nTop-10 keyword reference for the columns above:")
    for t in topic_ids[:10]:
        print(f"  topic_{t}: {keywords[t]}")


if __name__ == "__main__":
    run()
