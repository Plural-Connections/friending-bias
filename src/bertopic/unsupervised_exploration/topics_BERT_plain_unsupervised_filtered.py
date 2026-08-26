"""Unsupervised topic exploration of GreatSchools reviews (BERTopic), nces-filtered corpus.

Same as topics_BERT_plain_unsupervised.py, but fit only on the nces-filtered
review corpus (data/processed/gs_reviews_sorted_by_day_posted_filtered.csv) --
i.e. only reviews belonging to schools with a matching nces_id (and therefore
a friending-bias score) -- with MIN_TOPIC_SIZE=50 (see the note below on why
this differs from the unfiltered script's 200).

Unit of analysis is the INDIVIDUAL review, not the school. BERTopic assigns one
dominant topic per document, so a document should be about roughly one thing; a
single review usually is, whereas the school-concatenated file glues hundreds of
reviews into one blob and would wash the topic signal out. We therefore read the
per-review file and keep each review's `universal-id` so topics can be
aggregated back to the school level downstream (topics_distro_BERT*.py,
topics_BERT_classbased*.py).

Pipeline (the standard BERTopic pieces, chosen explicitly so the run is
reproducible and legible):
  1. embed    — Sentence-Transformers MiniLM turns each review into a vector.
  2. reduce   — UMAP compresses those vectors to 5 dims (clustering in ~400
                dims is unreliable; UMAP keeps neighborhood structure).
  3. cluster  — HDBSCAN finds dense groups and leaves genuine noise unclustered
                as topic -1 (no need to pre-declare how many topics exist).
  4. describe — a class-based TF-IDF (c-TF-IDF) over each cluster's reviews
                ranks the words that make a topic distinctive -> its keywords.

We FIT on the full nces-filtered corpus (69,238 reviews, the schools that have
a matching nces_id / friending-bias score) -- not the full ~618K-review corpus
the unfiltered script trains on.

Representative reviews are chosen here, not read off BERTopic's built-in list:
for each topic we take the reviews whose embeddings are closest (cosine) to the
topic centroid — its most prototypical members. Version-independent, and lets us
ask for exactly N.

`mean_words` is reported per topic so review-length variation is visible. Note
this is only a *visibility* hook: the actual length-confounder check (mean words
in group A vs. group B) belongs to the contrastive stages that create those
groups, not to this unsupervised pass.

MIN_TOPIC_SIZE=50, not 200: reusing the unfiltered script's min_cluster=200
threshold on this much smaller (nces-filtered, ~69K-review) corpus collapsed
everything into 2 degenerate topics (one alone covering 92.5% of reviews) --
see outputs/models/bertopic/bertopic_model_200_min_cluster_filtered/ and
outputs/bertopic/unsupervised_exploration/summary/topic_summary_200_min_cluster_filtered.csv
for that attempt, kept on disk as a record of why it was rejected. An
appropriate granularity threshold scales with corpus size, so we dropped back
to 50 here, which recovers 85 real topics. All outputs embed both
MIN_TOPIC_SIZE and "filtered" in their name so this run is never confused
with the unfiltered run (min_cluster=200) or with the rejected
min_cluster=200_filtered attempt.

Outputs:
    outputs/models/bertopic/bertopic_model_{MIN_TOPIC_SIZE}_min_cluster_filtered/ (the fitted model)
    outputs/bertopic/unsupervised_exploration/assignments/topic_assignments_{MIN_TOPIC_SIZE}_min_cluster_filtered.csv
        (universal-id, review id, topic)
    outputs/bertopic/unsupervised_exploration/summary/topic_summary_{MIN_TOPIC_SIZE}_min_cluster_filtered.csv
        (topic_id, size, share, keywords, mean_words)
    outputs/bertopic/unsupervised_exploration/representative_docs/topic_representative_docs_{MIN_TOPIC_SIZE}_min_cluster_filtered.csv
        (top-N reviews per topic, top TOP_N_TOPICS topics by size)

Run from the project root as a module:
    python -m src.bertopic.unsupervised_exploration.topics_BERT_plain_unsupervised_filtered
"""

import os

import numpy as np
import pandas as pd

reviews_path = "data/processed/gs_reviews_sorted_by_day_posted_filtered.csv"

TEXT_COL = "comments"
ID_COL = "universal-id"
NCES_ID_COL = "nces_id"
REVIEW_ID_COL = "id"
DATE_COL = "posted"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# UMAP is stochastic; fix its seed so topic assignments are reproducible.
RANDOM_STATE = 0
# HDBSCAN minimum cluster size == smallest topic we are willing to name
MIN_TOPIC_SIZE = 50
# Only pull representative reviews for the N largest topics.
TOP_N_TOPICS = 40
N_REPR_DOCS = 10
NOISE_TOPIC = -1

model_dir = f"outputs/models/bertopic/bertopic_model_{MIN_TOPIC_SIZE}_min_cluster_filtered"
table_dir = "outputs/bertopic/unsupervised_exploration"
assignments_path = f"{table_dir}/assignments/topic_assignments_{MIN_TOPIC_SIZE}_min_cluster_filtered.csv"
summary_path = f"{table_dir}/summary/topic_summary_{MIN_TOPIC_SIZE}_min_cluster_filtered.csv"
repr_docs_path = f"{table_dir}/representative_docs/topic_representative_docs_{MIN_TOPIC_SIZE}_min_cluster_filtered.csv"


def load_reviews(path=reviews_path, random_state=RANDOM_STATE):
    """Read the per-review file and drop empty reviews. Fits on the full
    nces-filtered corpus. Returns school id, NCES id, review id, date, text,
    words."""
    df = pd.read_csv(
        path, usecols=[ID_COL, NCES_ID_COL, REVIEW_ID_COL, DATE_COL, TEXT_COL],
        dtype={NCES_ID_COL: str},
    )
    df = df.dropna(subset=[TEXT_COL])
    df[TEXT_COL] = df[TEXT_COL].astype(str)
    df = df[df[TEXT_COL].str.strip().str.len() > 0]
    df = df.reset_index(drop=True)
    df["n_words"] = df[TEXT_COL].str.split().str.len()
    return df


def fit_topics(docs, embedding_model=EMBEDDING_MODEL, random_state=RANDOM_STATE,
               min_topic_size=MIN_TOPIC_SIZE, embeddings=None):
    """Fit BERTopic on a list of review strings.

    Returns (fitted_model, topic_per_doc, embeddings). Embeddings are computed
    once here and returned so callers can reuse them for centroid-based
    representative-doc selection without encoding twice.
    """
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    encoder = SentenceTransformer(embedding_model)
    if embeddings is None:
        embeddings = encoder.encode(
            docs, batch_size=64, show_progress_bar=True, convert_to_numpy=True,
        )

    umap_model = UMAP(
        n_neighbors=15, n_components=5, min_dist=0.0,
        metric="cosine", random_state=random_state,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size, metric="euclidean",
        cluster_selection_method="eom", prediction_data=True,
    )
    # c-TF-IDF vocabulary: drop English stopwords, keep uni- and bi-grams.
    # NB: BERTopic feeds this one concatenated document *per topic*, so a
    # min_df here would count topics, not reviews — hence none.
    vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 2))

    topic_model = BERTopic(
        embedding_model=encoder,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=False,
        verbose=True,
    )
    topics, _ = topic_model.fit_transform(docs, embeddings=embeddings)
    return topic_model, np.asarray(topics), embeddings


def topic_summary(model, topics, meta):
    """One row per topic: id, size, share, mean review length, top-10 keywords.

    `mean_words` is included so review-length variation across topics is visible
    (see the module docstring on why that is only a visibility hook here).
    """
    n = len(topics)
    words = meta["n_words"].to_numpy()
    rows = []
    for topic_id in sorted(set(topics)):
        idx = np.where(topics == topic_id)[0]
        terms = model.get_topic(topic_id) or []
        rows.append({
            "topic_id": int(topic_id),
            "size": len(idx),
            "share": round(len(idx) / n, 4),
            "mean_words": round(float(words[idx].mean()), 1),
            "keywords": ", ".join(w for w, _ in terms[:10]),
        })
    out = pd.DataFrame(rows)
    # Sort by size, but always push the noise topic (-1) to the bottom.
    out["_order"] = np.where(out["topic_id"] == NOISE_TOPIC, -1, out["size"])
    return out.sort_values("_order", ascending=False).drop(columns="_order").reset_index(drop=True)


def representative_reviews(topics, embeddings, meta, keep_topics, n=N_REPR_DOCS):
    """Top-`n` most prototypical reviews for each topic in `keep_topics`.

    Prototypicality = cosine similarity to the topic's mean (centroid) embedding.
    Returns a long DataFrame: topic_id, rank, similarity, <ids>, n_words, text.
    """
    unit = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    rows = []
    for topic_id in keep_topics:
        idx = np.where(topics == topic_id)[0]
        centroid = unit[idx].mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-12
        sims = unit[idx] @ centroid
        top = idx[np.argsort(-sims)[:n]]
        for rank, j in enumerate(top, start=1):
            row = meta.iloc[j]
            rows.append({
                "topic_id": int(topic_id),
                "rank": rank,
                "similarity": round(float(unit[j] @ centroid), 4),
                ID_COL: row[ID_COL],
                REVIEW_ID_COL: row[REVIEW_ID_COL],
                "n_words": int(row["n_words"]),
                "text": row[TEXT_COL],
            })
    return pd.DataFrame(rows)


def print_representative_reviews(reprs, summary):
    """Print the top representative reviews per topic to the console for review."""
    shares = summary.set_index("topic_id")["share"]
    keywords = summary.set_index("topic_id")["keywords"]
    for topic_id, group in reprs.groupby("topic_id"):
        print(f"\n{'='*100}\nTOPIC {topic_id}  |  {shares[topic_id]:.1%} of reviews"
              f"\nkeywords: {keywords[topic_id]}\n{'-'*100}")
        for r in group.sort_values("rank").itertuples():
            text = " ".join(str(r.text).split())
            print(f"  [{r.rank}] ({r.n_words}w, sim={r.similarity:.2f}) {text}")


def run():
    os.makedirs(model_dir, exist_ok=True)
    for sub in ("assignments", "summary", "representative_docs"):
        os.makedirs(f"{table_dir}/{sub}", exist_ok=True)

    meta = load_reviews()
    docs = meta[TEXT_COL].tolist()
    print(f"n reviews = {len(docs):,}  |  n schools = {meta[ID_COL].nunique():,}\n")

    model, topics, embeddings = fit_topics(docs)
    n_topics = len(set(topics) - {NOISE_TOPIC})
    print(f"\nFound {n_topics} topics; {(topics == NOISE_TOPIC).sum():,} reviews "
          f"left as noise (topic {NOISE_TOPIC})\n")

    summary = topic_summary(model, topics, meta)
    # `summary` is already sorted by size descending (noise pushed to the bottom).
    review_topics = summary.loc[
        summary["topic_id"] != NOISE_TOPIC, "topic_id"
    ].head(TOP_N_TOPICS).tolist()
    reprs = representative_reviews(topics, embeddings, meta, review_topics)

    assignments = meta[[ID_COL, REVIEW_ID_COL, DATE_COL, "n_words"]].copy()
    assignments["topic"] = topics

    assignments.to_csv(assignments_path, index=False)
    summary.to_csv(summary_path, index=False)
    reprs.to_csv(repr_docs_path, index=False)
    try:
        model.save(model_dir, serialization="safetensors",
                   save_ctfidf=True, save_embedding_model=EMBEDDING_MODEL)
    except Exception as exc:  # fall back to a single pickle if safetensors is unavailable
        print(f"safetensors save failed ({exc}); falling back to pickle")
        model.save(f"{model_dir}/model.pkl", serialization="pickle")

    print("=== Topics by size (excluding noise) ===")
    with pd.option_context("display.max_colwidth", 70, "display.width", 160):
        print(summary[summary["topic_id"] != NOISE_TOPIC].to_string(index=False))

    print(f"\n=== Top {N_REPR_DOCS} representative reviews per topic "
          f"(top {TOP_N_TOPICS} topics by size) ===")
    print_representative_reviews(reprs, summary)

    print(f"\nWrote model to {model_dir}/ and tables to {table_dir}/")


if __name__ == "__main__":
    run()
