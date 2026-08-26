# demo_baseline

# CURRENT STATE (verified against code, 2026-08-25)

Everything below this point, starting at "# related data columns in processed
csv:", is the ORIGINAL write-up for this stage and is kept as-is for its
process notes and reasoning (in particular the GroupKFold-vs-KFold experiment
narrative in the "demographics only baseline" section) — but it is ARCHIVED:
some of it describes decisions in past tense from when they were still being
made, which reads ambiguously once the decision is settled. This section
above is the unambiguous current answer.

Confirmed directly against the code (`grep -rn "GroupKFold" src/` returns zero
matches across the entire pipeline): every regression script —
`demo_baseline.py`, `demo_baseline_filtered.py`, `topic_regression_per_review.py`,
`topic_regression_concat_filtered.py` — uses a single plain, shuffled `KFold`:

```python
N_SPLITS = 5
CV = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEEDS)
```

GroupKFold (keyed on NCES district — the first seven digits of `nces_id`,
resolving to 8,917 districts) was tried first, to guard against schools
within the same district leaking district-level similarity across folds. It
was dropped because it changed the out-of-sample scores by a negligible
amount (GBM: 0.368 ungrouped vs. 0.363 grouped; ElasticNet: 0.210 vs. 0.206 —
see the archived section below for the original context), and because
sharing a district doesn't reliably imply sharing resources, demographics, or
other features, so the leakage concern the grouping was meant to guard
against was likely overstated for this data. `src/eval/cv_splits.py`'s
GroupKFold helper and the `groups` argument that had been threaded through
every function were removed along with it, which also simplified the code.
This is not an open question — it's the final, current setup for every
regression script in the pipeline.

# related data columns in processed csv:

student-teacher-ratio,enrollment,percent-free-and-reduced-price-lunch,percent-economically-disadvantaged,percent-students-with-limited-english-proficiency,percentage-of-full-time-teachers-who-are-certified,student-counselor-ratio,percentage-female,percentage-male,ethnicity-Hispanic,ethnicity-Black,ethnicity-Two or more races,ethnicity-Asian or Pacific Islander,ethnicity-White,ethnicity-Native Hawaiian or Other Pacific Islander,ethnicity-Native American,high_school_name,zip,county,students_9_to_12,ec_own_ses_hs,ec_own_ses_se_hs,ec_parent_ses_hs,ec_parent_ses_se_hs,ec_high_own_ses_hs,ec_high_own_ses_se_hs,ec_high_parent_ses_hs,ec_high_parent_ses_se_hs,exposure_own_ses_hs,exposure_parent_ses_hs,bias_own_ses_hs,bias_parent_ses_hs,bias_high_own_ses_hs,bias_high_parent_ses_hs,clustering_hs,volunteering_rate_hs

# methodology for single factor correlation graphs taken from Social capital II: determinants of economic connectedness

Paper methology: "To construct the binned scatter plots, we divide the variable on the horizontal axis into ventiles (5 percentile point bins) and plot the mean of the vertical-axis variable against the mean of the horizontal-axis variable in each ventile. All binned scatter plots are weighted by the number of students in each high school as reported in the NCES data. As a visual guide to approximate the non-parametric relationships, the solid lines in each figure show lines of best fit from quadratic regressions estimated using OLS."

My methology: Starting from the school-level table gs_demos_with_social_capital.csv, I take friending bias by own socioeconomic status (bias_own_ses_hs) as the outcome and examine the following as predictors: the shares of White and Black students, the student–teacher ratio, the percentage of limited-English-proficiency students, own-SES exposure, parent-SES friending bias, and a Herfindahl–Hirschman index (HHI) of racial concentration constructed by renormalizing each school's ethnicity percentages to sum to one and summing their squares (so the index runs from low values for evenly mixed schools up to 1 for a single-group school). For each predictor I build a binned scatter plot following the Social Capital Atlas convention: the predictor is divided into ventiles (twenty bins of five percentile points each), and within every bin I compute the student-weighted mean — weighting by grades 9–12 enrollment (students_9_to_12), the population over which friending bias is defined — of both the predictor and friending bias, plotting those twenty points against each other. Alongside each panel I report the weighted quadratic R² (computed on the underlying school-level observations, not the twenty binned points), the p-value on the quadratic term as a test for curvature, and the sample size.

# demographics only baseline

Working from data/interim/gs_demos_with_social_capital.csv, I use the 13,958 schools that have a friending-bias value, predicting it from eighteen features: fifteen demographics — school size measured as grades-9–12 enrollment (students_9_to_12, taken from the Social Capital Atlas, not to be confused with enrollment from gs), the student–teacher and student–counselor ratios, the shares of students on free/reduced lunch and with limited English proficiency, the percentage of certified teachers, the sex breakdown, and the seven ethnicity shares — together with three review covariates (review count, total words, and mean words per review). I drop percent-economically-disadvantaged because it is missing for roughly 95% of schools.

To see whether the relationship is simple or complex, I fit two models and compare them: an ElasticNet regularized linear model (with median imputation and standardization) and a gradient-boosted tree model (LightGBM). Both are scored with five-fold KFold (shuffled, `random_state=0`) rather than a district-grouped split: I had originally used GroupKFold keyed on NCES district to guard against schools in the same district leaking across train/test, but school-level heterogeneity turned out to already be large enough that the grouping wasn't needed, so I dropped it in favor of a plain shuffled split — this also simplified the code, since `src/eval/cv_splits.py`'s GroupKFold helper and the `groups` argument threaded through every function could be removed. The same `CV` splitter object is reused for the GBM grid search, the ElasticNet's internal alpha/l1_ratio tuning, and the final out-of-sample scoring, so "best params" and "reported score" always come from the same folds.

The results show that demographics can explain some shares of friending bias. The ElasticNet model reaches an out-of-sample R² of 0.210 (RMSE 0.053), while the gradient-boosted model reaches 0.368 (RMSE 0.047, best params: `learning_rate=0.05, num_leaves=31, n_estimators=500`), so the best demographic baseline accounts for roughly 37% of the variation in bias. The most informative result is the gap between the two models: the tree model beats the linear model by three-quarters (0.368 vs. 0.210). That gap tells us the demographics–bias relationship is substantially nonlinear and involves interactions. This is perhaps what the inverted-U shapes in the earlier binned-scatter plots hinted at, since a straight line cannot bend to fit a hump. The GBM should be the baseline and not the linear model. (These numbers are essentially unchanged from the district-grouped CV version — 0.206/0.363 — which suggests the grouping wasn't materially affecting the score either way.)

Finally, inspecting the tree model's permutation importances gives the "known confounders" list to carry into Stage 6. School size (students_9_to_12) is by far the dominant predictor (importance ≈ 0.66), well ahead of the share of White students (≈ 0.23), the free/reduced-lunch rate as a poverty proxy (≈ 0.13), and the Black and Asian enrollment shares (≈ 0.12 and 0.08); the Hispanic share, limited-English share, the two ratios, and the remaining ethnicity shares follow. The three review covariates (mean_words_per_review ≈ 0.045, n_words ≈ 0.039, n_reviews ≈ 0.024) rank near the bottom, which means review volume and length barely predict bias and are therefore not a lurking confounder that the text stages might accidentally absorb. Keep in mind that ElasticNet's internal penalty tuning and the GBM grid search both use the same shuffled KFold as the reported out-of-sample scores.

After meeting with Nabeel: adding text features in doesn’t give you much: from earlier studies, discovered that whiter more affluent schools have more reviews, so it’s probably already taken into account of that; venn diagram thing, information captured in one feature (demographics) is already in another feature (text)
