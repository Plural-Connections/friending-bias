# stage3_concept_validation

# ran regression (similar to demographics) on topics

major caveat - only 10k schools made it into the final review!!

concat vs per review

Boundary contamination is real and unavoidable in the concat approach. I confirmed directly in BERTopic's source that approximate_distribution tokenizes a school's whole concatenated blob as one continuous stream and slides its window across it with no awareness of where one review ends and the next begins. Any window straddling that boundary blends unrelated text into a nonsense token set that still gets scored. Per-review scoring makes this structurally impossible — there's no boundary to cross within a single review.
The implicit word-count weighting in concat is a real, unexamined side effect, not a deliberate choice — a school's one 900-word review mechanically outweighs several 20-word reviews by ~45x in the final blend. Per-review + equal-weight mean makes the weighting scheme an explicit, defensible decision instead of an accident of how L1-normalization interacts with concatenation.

Low R^2 fix

Widening TOP_N_TOPICS only reused the already-fitted model (cheap: ~5 min for approximate_distribution). Fixing the fragmentation means changing how the topics themselves get formed, which means rerunning topics_BERT_plain_unsupervised.py from scratch: re-embed all 618K reviews with MiniLM, rerun UMAP, rerun HDBSCAN clustering. That script's own docstring calls out "expect a long run (GPU/MPS strongly preferred)" — this machine has MPS (Apple Silicon GPU) but not CUDA, so it'll help but I can't promise a tight time estimate; I haven't run this step myself yet this session.

The one parameter that actually matters here: MIN_TOPIC_SIZE (topics_BERT_plain_unsupervised.py:69), currently 50 — this feeds HDBSCAN's min_cluster_size, and it's the knob controlling topic granularity. Everything else in that script (embedding model, UMAP settings, vectorizer/ngram range) isn't implicated in the school-name-fragmentation problem, so I wouldn't touch those.

My reasoning for what to set it to:

The school-name clusters we saw (st theresa, theresa, teresa; walden, walden school, walden green) are almost certainly small clusters right around the current threshold — a single school's ~50-150 reviews clumping together because they all repeat that school's distinctive name, not because they share real thematic content. Raising the threshold should push those below the bar, forcing them either into a real topic or into noise (currently already 52.5%).
There's no formula that tells you the "right" topic count in advance — it's inherently a try it and inspect parameter. I can't guarantee a specific outcome.
I'd suggest MIN_TOPIC_SIZE = 200 (4x current) as a reasonable first experiment — enough to meaningfully cull the singleton/school-name clusters without being so aggressive we lose real smaller-but-legitimate themes (e.g. topic_118: drugs, topic_181: counselors were both useful and moderately sized).

Update: increasing top topics to 300 only increased to 3% R^2, might need to retrain model

1. Are reviews joined to the right school? Verified clean: nces_id ↔ universal-id is a 1:1 mapping with zero duplicates in the demos file, and the per-review file's universal-id is always present (it's the review table's own primary grouping key, unlike the sparse embedded nces_id). No mis-joins found.

2. Are individual topic assignments sensible? Spot-checked the keyword lists (topic_26 = "ap, ap classes, honors", topic_25 = "ib, ib program, baccalaureate", topic_8 = "worst school, worst, horrible") — these read as coherent, sane clusters, not garbage.

3. Concatenation problem — real, but not the dominant driver here. I compared how much of each school's total topic mass falls inside our 40 kept columns, old vs. new method:
   OLD (concatenated blob): mean 23.4%, median 18.1% of mass captured
   NEW (per-review, averaged): mean 18.1%, median 13.9% of mass captured
   Both are bad. Fixing the boundary-crossing bug didn't fix the real problem, because it wasn't the real problem. R^2 only improved by 0.4% after new method

4. Normalization / noise — this is the actual issue. The fitted BERTopic model has 952 real topics, not 40 — "top 40" is a tiny slice we chose for legibility, not the whole model. I checked the full distribution:

   52.5% of all 618,039 reviews are hard-clustering noise (Topic == -1) — over half the corpus never joined any named cluster at all.
   Even restricting to the soft/token-set approximate distribution (which never treats -1 as a candidate, as we established earlier), ~27% of reviews score below min_similarity=0.1 against every one of the 952 topics and get an all-zero distribution row, functionally noise again just represented differently.
   Of the reviews that do get a real distribution (which sums to exactly 1.0 across all 952 topics — normalization itself is fine), the topic-size distribution has a brutally long tail:
   top 40 topics → 44.9% of non-noise reviews
   top 100 topics → 60.8%
   top 200 topics → 72.6%
   top 500 topics → 88.6%
   top 952 topics → 100.0%
   So "top 40" — the feature set both scripts use — captures well under half of even the non-noise content, and that's before accounting for the 52.5% that's noise to begin with. Roughly 75-85% of each school's real topical signal is sitting in one of the other 912 topics we simply never expose to the GBM as a feature. That's a much bigger information loss than anything caused by concatenation or aggregation choice, and it's very plausibly why this comes in at R²=0.01 while Maddie's n-gram model — which keeps the entire vocabulary as features, no top-N cutoff — got 0.12.

5. GBM itself — not implicated. With this little retained signal, no amount of hyperparameter tuning recovers information that was never in the feature matrix.

Root cause
MIN_TOPIC_SIZE=50 in topics_BERT_plain_unsupervised.py:69, fit over 618K reviews, produced 952 mostly-tiny topics plus 52.5% noise. TOP_N_TOPICS=40 then keeps a small, arbitrary slice of that long tail as the entire feature set for both distribution scripts.

Options going forward
Widen the feature set — bump TOP_N_TOPICS way up (e.g. 300-500) so most non-noise mass is actually captured. Simple, no refit needed, but produces a much wider/sparser feature matrix.
Refit BERTopic coarser — raise min_topic_size well above 50 (or call .reduce_topics() on the existing model) so "top 40" or "top 100" actually covers most of the corpus instead of a sliver of it.
Some combination of both.
Want me to try option 1 first (cheap, reuses the existing fitted model) and rerun the regression to see how much of the R² gap closes?

# handlabeling (abandoned)

Handlabeling (~400 reviews), BERTopic:

Bullying: Reviews describe rampant bullying and often lack of institutional support

Teacher and Staff Quality: Reviews describe friendly, enthusiastic, great teachers and staff

Christian School: Reviews describe Christian schools embodying Christian values

Montessori School: Reviews describe the Montessori approach

Active PTA and parent involvement: Reviews describe active PTAs, PTOs, and involved parents

Spanish immersion program: Reviews describe spanish dual language and bilingula immersion programs

Music and arts program: Reviews describe, band, music, dance, and arts programs

Charter School: Reviews describe charter schools

Bad Principal: Reviews describe terrible principals and leadership

Racism: Reviews describe racial diversity and racism

Special Education and IEP: Reviews describe IEP, special needs, special education, and learning disablity

Test scores: Reviews describe test scores and standards

Catholic schools: Reviews describe catholic schools

Good Principal: Reviews describe great principals and leadership

Dresscode: Reviews discuss dress codes

Gifted program: Reviews describe gifted programs and students

AP/IB program: Reviews describe AP/IB programs

Sports and Athletics: Reviews describe sports and athletic programs

Covid Pandemic: Reviews describe experiences related to 2020 Covid 19 pandemic and remote learning

Small Class Size: Review describe small class sizes and specific, personal education

Interesting common themes (not topics): middle school, kindergarten, meta reviews about reviews

Complications: putting a word there to negate

# GABRIEL instruction log

INSTRUCTIONS = (
"Each text is ALL of the GreatSchools reviews for a single U.S. high "
"school, concatenated together (parent/student/teacher reviews mixed). "
"Identify what differentiates these two groups of SCHOOLS based on what is "
"actually happening inside the building and in the community around it — "
"concrete practices, policies, incidents, staff behavior, programs, "
"facilities, and outcomes that reviewers describe.\n\n"
"Do NOT propose themes about how the reviews are WRITTEN: ignore tone, "
"sentiment intensity, formality, enthusiasm, politeness, gratitude, "
"grammar, review length, vocabulary, or reviewer style/identity. A theme "
"like 'circle entries express more gratitude than square entries' or "
"'group A entries are written more informally' is NOT acceptable — it "
"describes the writing, not the school. A theme like 'group A entries "
"describe more responsive communication from teachers' or 'group B entries "
"describe more frequent bullying incidents' IS acceptable — it describes a "
"substantive, actionable characteristic of the school itself.\n\n"
"Also avoid themes built on proper nouns, school names, locations, or other "
"identifying details specific to one school rather than a generalizable "
"pattern across schools."
)

"Each text is ALL of the GreatSchools reviews for a single U.S. high "
"school, concatenated together (parent/student/teacher reviews mixed). "
"Identify what differentiates these two groups of SCHOOLS based on what is "
"actually happening inside the building and in the community around it — "
"concrete practices, policies, incidents, staff behavior, programs, "
"facilities, and outcomes that reviewers describe.\n\n"
"Do NOT propose themes about how the reviews are WRITTEN: ignore tone, "
"sentiment intensity, formality, enthusiasm, politeness, gratitude, "
"grammar, review length, vocabulary, or reviewer style/identity. A theme "
"like 'circle entries express more gratitude than square entries' or "
"'group A entries are written more informally' is NOT acceptable — it "
"describes the writing, not the school. A theme like 'group A entries "
"describe more responsive communication from teachers' or 'group B entries "
"describe more frequent bullying incidents' IS acceptable — it describes a "
"substantive, actionable characteristic of the school itself.\n\n"
"Also avoid themes built on proper nouns, school names, locations, or other "
"identifying details specific to one school rather than a generalizable "
"pattern across schools.\n\n"
"MERGE aggressively: before finalizing your theme list, check every pair of "
"themes against this test — 'would the same piece of review evidence "
"plausibly support both of these?' If yes, they are not two themes; fold "
"them into one and write a single definition that covers the union. Do not "
"split a topic into narrower sub-themes (depth vs. breadth, frequency vs. "
"severity, one specific example vs. the general pattern) unless the two "
"sub-themes would be supported by clearly different, non-overlapping "
"evidence. For example:\n"
" - 'more AP/IB course offerings' and 'more rigorous/advanced coursework' "
"-> merge into one 'academic program strength' theme.\n"
" - 'more bullying incidents' and 'less effective response to bullying' "
"-> merge into one 'bullying/school safety climate' theme.\n"
" - 'unresponsive teachers' and 'unresponsive administration' -> merge "
"into one 'poor staff communication/responsiveness' theme, unless the "
"review evidence clearly distinguishes teacher-level from admin-level "
"communication.\n"
" - 'outdated facilities' and 'crowded classrooms' -> these ARE distinct "
"(different evidence: building condition vs. enrollment/space), so keep "
"separate.\n"
"When you are unsure whether two candidate themes are really different, "
"default to merging them — a smaller set of broad, clearly-separated "
"themes is strongly preferred over a larger set of narrow, overlapping "
"ones. Each final theme should be describable in one sentence that could "
"NOT also describe any other theme in the set."

    INSTRUCTIONS = (
    "Each text is ALL of the GreatSchools reviews for a single U.S. high "
    "school, concatenated together (parent/student/teacher reviews mixed). "
    "Identify what differentiates these two groups of SCHOOLS based on what is "
    "actually happening inside the building and in the community around it — "
    "concrete practices, policies, incidents, staff behavior, programs, "
    "facilities, and outcomes that reviewers describe.\n\n"
    "Do NOT propose themes about how the reviews are WRITTEN: ignore tone, "
    "sentiment intensity, formality, enthusiasm, politeness, gratitude, "
    "grammar, review length, vocabulary, or reviewer style/identity. A theme "
    "like 'circle entries express more gratitude than square entries' or "
    "'group A entries are written more informally' is NOT acceptable — it "
    "describes the writing, not the school. A theme like 'group A entries "
    "describe more responsive communication from teachers' or 'group B entries "
    "describe more frequent bullying incidents' IS acceptable — it describes a "
    "substantive, actionable characteristic of the school itself.\n\n"
    "Also avoid themes built on proper nouns, school names, locations, or other "
    "identifying details specific to one school rather than a generalizable "
    "pattern across schools.\n\n"
    "MERGE, but only when two candidate themes restate the SAME underlying "
    "fact: before finalizing your theme list, check every pair against this "
    "test — 'if I deleted one of these, would the other one now just be "
    "saying the same thing in different words?' Only merge when the answer "
    "is yes. Do NOT merge just because the same reviews tend to mention both "
    "— reviews routinely praise or criticize several different things about "
    "a school in one breath, so co-occurring in the same text is NOT "
    "evidence that two themes are one construct. Do not split a topic into "
    "narrower sub-themes (depth vs. breadth, frequency vs. severity, one "
    "specific example vs. the general pattern) unless the two sub-themes "
    "would be supported by clearly different, non-overlapping evidence.\n\n"
    "Watch for one recurring failure mode in particular: folding a concrete, "
    "specific offering or practice (a particular kind of course, program, "
    "policy, or facility the school does or doesn't have) into a broad, "
    "holistic quality-of-experience theme (general teaching quality, "
    "engagement, morale, climate) just because schools praised for the "
    "specific offering are often praised generally too. What a school "
    "concretely provides and how the delivery/experience feels are "
    "DIFFERENT facts — keep them as separate themes even when the same "
    "review touches both. For example:\n"
    "  - 'more advanced/honors-level course offerings' and 'more rigorous "
    "coursework' -> merge into one 'academic program strength' theme (same "
    "underlying fact), but do NOT further merge that into a broader "
    "'teaching quality' or 'instructional experience' theme — course "
    "availability/rigor and how well classes are taught are different "
    "facts, even if the same review praises both.\n"
    "  - 'more bullying incidents' and 'less effective response to bullying' "
    "-> merge into one 'bullying/school safety climate' theme.\n"
    "  - 'unresponsive teachers' and 'unresponsive administration' -> merge "
    "into one 'poor staff communication/responsiveness' theme, unless the "
    "review evidence clearly distinguishes teacher-level from admin-level "
    "communication.\n"
    "  - 'outdated facilities' and 'crowded classrooms' -> these ARE distinct "
    "(different evidence: building condition vs. enrollment/space), so keep "
    "separate.\n"
    "When you are unsure whether two candidate themes restate the same "
    "fact, default to keeping them separate, not merging — a theme list "
    "where every entry still names a specific, concrete practice is "
    "strongly preferred over a smaller set of vaguer, catch-all themes. If "
    "a merged definition becomes so broad it no longer names a specific "
    "practice, policy, or offering (e.g. a vague 'instructional experience' "
    "or 'overall quality' label), that is a sign of over-merging — split it "
    "back into the more specific themes it absorbed. Each final theme "
    "should be describable in one sentence that could NOT also describe any "
    "other theme in the set."

)
