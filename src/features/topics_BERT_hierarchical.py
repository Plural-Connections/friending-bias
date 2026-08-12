"""Stage 2b — hierarchical topic structure (BERTopic), on top of the flat
topics from topics_BERT_plain_unsupervised.py.

The plain/flat pass gives ~950 independent topics with no notion of which ones
are near-duplicates or minor variants of a broader theme. This stage adds that
structure: BERTopic's `hierarchical_topics()` computes the cosine distance
between every pair of topics' c-TF-IDF vectors, then runs hierarchical
(agglomerative) clustering over those distances — the same technique described
here:
https://maartengr.github.io/BERTopic/getting_started/hierarchicaltopics/hierarchicaltopics.html#linkage-functions

This does NOT change any topic or its assignments — it only adds a merge tree
on top of the topics that already exist, useful for:
  * spotting near-duplicate topics worth merging before Stage 3's concept list
  * picking a natural "collapse level" (how many topics to keep) instead of
    treating all ~950 flat topics as equally distinct
  * distinguishing a genuinely standalone topic (its nearest neighbor in the
    tree is still a large distance away) from one that's just a fragment of a
    bigger theme (merges in almost immediately)

Linkage function: BERTopic's default is Ward linkage (minimizes within-cluster
variance at each merge — good for compact, evenly-sized clusters). The doc page
above lists the scipy alternatives — single, complete, average, centroid,
median — each with a different merging bias (e.g. single-linkage chains
together anything with even one close pair, complete-linkage only merges
clusters that are close on their *farthest* members). LINKAGE_METHOD below is
a plain string knob so a different one can be tried without touching the
pipeline logic.

We reuse the ALREADY-fitted model from topics_BERT_plain_unsupervised.py
(outputs/models/bertopic_model/) when present, since hierarchical_topics()
only needs the fitted topics + each topic's c-TF-IDF vector, not the raw
embeddings — no need to re-embed/re-cluster the full corpus. Falls back to a
fresh fit if no saved model exists yet.

Outputs:
    outputs/tables/topic_hierarchy.csv   (hierarchical_topics() merge tree)
    outputs/tables/topic_tree.txt        (human-readable text tree)
    outputs/figures/topic_hierarchy.html (interactive dendrogram)

Run from the project root as a module:
    python -m src.features.topics_BERT_hierarchical
"""

import os

from scipy.cluster import hierarchy as sch

from .topics_BERT_plain_unsupervised import (
    NOISE_TOPIC,
    TEXT_COL,
    fit_topics,
    load_reviews,
    model_dir,
)

hierarchy_path = "outputs/tables/topic_hierarchy.csv"
tree_path = "outputs/tables/topic_tree.txt"
hierarchy_fig_path = "outputs/figures/topic_hierarchy.png"

# Ward is BERTopic's own default. Alternatives from the linked "Linkage
# functions" section: "single", "complete", "average", "centroid", "median".
LINKAGE_METHOD = "complete"


def load_or_fit_base_model(reviews_path_arg=None):
    """Reuse the saved flat-topic model + its docs if available (fast: no
    re-embedding/re-clustering needed), else fit fresh on the full corpus."""
    from bertopic import BERTopic

    meta = load_reviews(reviews_path_arg) if reviews_path_arg else load_reviews()
    docs = meta[TEXT_COL].tolist()

    try:
        model = BERTopic.load(model_dir)
        if len(model.topics_) != len(docs):
            raise ValueError(
                f"saved model has {len(model.topics_)} docs, "
                f"but load_reviews() now returns {len(docs)} — stale model, refitting"
            )
        topics = model.topics_
        print(f"Loaded existing model from {model_dir}/ ({len(docs):,} docs)")
    except Exception as exc:
        print(f"Could not reuse saved model ({exc}); fitting fresh on the full corpus.")
        model, topics, _ = fit_topics(docs)

    return model, docs, topics, meta


def build_hierarchy(model, docs, linkage_method=LINKAGE_METHOD):
    """Ward (or other scipy linkage) hierarchy over topics' c-TF-IDF distances."""
    linkage_function = lambda x: sch.linkage(x, linkage_method, optimal_ordering=True)
    return model.hierarchical_topics(docs, linkage_function=linkage_function)


def run():
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)

    model, docs, topics, meta = load_or_fit_base_model()
    n_topics = len(set(topics) - {NOISE_TOPIC})
    print(f"n reviews = {len(docs):,}  |  n flat topics (excl. noise) = {n_topics}\n")

    hier = build_hierarchy(model, docs)
    hier.to_csv(hierarchy_path, index=False)
    print(f"Wrote {hierarchy_path} ({len(hier)} merges, linkage={LINKAGE_METHOD!r})")

    tree_text = model.get_topic_tree(hier)
    with open(tree_path, "w") as f:
        f.write(tree_text)
    print(f"Wrote {tree_path}")

    # visualize
    fig = model.visualize_hierarchy(
        hierarchical_topics=hier, width=1400, height=max(600, 18 * n_topics)
    )
    fig.write_image(hierarchy_fig_path)
    print(f"Wrote {hierarchy_fig_path}")

    print("\n=== Topic hierarchy (text tree) ===")
    print(tree_text)

    return hier


if __name__ == "__main__":
    run()
