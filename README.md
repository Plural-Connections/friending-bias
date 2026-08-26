# friending-bias

Naming-convention glossary for `src/` and `outputs/`. If a script or filename
carries one of these suffixes, this is what it means.

## Pipeline stages (top-level groupings in both `src/` and `outputs/`)

- **`demographics_regression/`** -- predicts friending bias from school
  demographics (Stage 1 baseline).
- **`bertopic/`** -- topic-modeling pipeline: fits BERTopic on review text,
  then uses the topic mixtures as regression features. Split into
  `unsupervised_exploration/` (fitting + inspecting topics),
  `class_based/` (topic frequency by high-FB vs. low-FB class),
  `hierarchical_exploration/` (topic merge tree), and `topic_regression/`
  (topic-mixture -> friending-bias GBM).
- **`gabriel/`** -- LLM-based (GABRIEL) theme discovery and classification of
  review text, run as an alternative/complement to BERTopic.

## Corpus scope

- **`filtered`** -- restricted to the NCES-matched, review-availability
  subset of schools (11,859 schools / 69,238 reviews), vs. the full
  ~618K-review corpus when absent.
- **`min_cluster{N}`** -- HDBSCAN's minimum cluster size parameter used to
  fit that particular BERTopic model; different values are different model
  fits with different topic counts, not comparable head-to-head.
- **`top{N}`** -- how many of the fitted topics were kept as regression
  features (e.g. `top235` = all 235 topics from the min_cluster=200 fit,
  `top40`/`top300` = an earlier, since-superseded topic-count cutoff).

## Review aggregation method

- **`concat`** -- glue all of a school's reviews into one blob and score it
  once against the topic model. Has boundary-contamination and implicit
  word-count-weighting issues.
- **`per_review`** -- score each review individually, then average per
  school. Avoids both issues above; preferred method going forward.

## Class-based chart suffixes

- **`normalized`** -- y-axis is each topic's share of that class's reviews,
  not a raw review count (controls for the two classes having different
  total review counts).
- **`gap_sorted`** -- topics ordered by `|high_fb_share - low_fb_share|`
  (the gap between classes) instead of by overall frequency.

## Regression / importance outputs

- **`signed_importance`** -- permutation importance (magnitude-only) times
  `sign(correlation with target)`, so bars have a direction
  (positive = associated with higher friending bias).
- **`_gbm` / `_metrics` / `_importances`** -- the fitted model, its
  cross-validated scores, and its permutation-importance table,
  respectively, for a given regression run.

## GABRIEL merge/adjustment variants

- **`no_merge` / `merge_aggressive` / `merge_deliberate`** -- three
  instruction variants controlling how aggressively GABRIEL's theme
  -discovery step merges similar themes together.
- **`len_adjusted`** -- effect sizes recomputed after controlling for
  review length, to check a theme's association isn't just a length
  artifact.

## Archive

`outputs/archive_metric_tuning_explorations_for_bertopic/` holds superseded
BERTopic runs and the specification-search history that led to the current
pipeline -- see `ARCHIVE_README.md` in that folder for details. Nothing in
it is read or written by any current script.
