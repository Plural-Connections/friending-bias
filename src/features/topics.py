"""Stage 2 — unsupervised topic exploration via BERTopic."""

def fit_topics(docs: list, embedding_model: str = "all-MiniLM-L6-v2"):
    raise NotImplementedError

def export_topic_summary(model, docs):
    """Returns topic_id, keywords, size, and 5 representative docs each."""
    raise NotImplementedError
