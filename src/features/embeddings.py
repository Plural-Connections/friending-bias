"""Stage 4 — text embeddings (predictive ceiling)."""

def embed_frozen(texts: list, model_name: str = "all-mpnet-base-v2"):
    raise NotImplementedError

def finetune_encoder(train_ds, val_ds, base_model: str = "distilbert-base-uncased", target: str = "friending_bias"):
    raise NotImplementedError
