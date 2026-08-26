# stage0_data_summary

# Notes on research process:

raw data sizes:
data/raw/gs_reviews.csv: 764442 rows total (`comments` column), of which 701157 are
non-empty; non-empty reviews total 58200527 words (word-split, no other cleaning)

review aggregation:
Sorted all reviews by chronological order down to date (not the hour), dropped all reviews after 2022 (2022 reviews are not dropped)
No built in mininum review count; concatenated all reviews and dropped rows with blank reviews; left review count and word count

joining via nces_ids:
Atlas schools: 17525, demos schools: 133912
Matched schools: 17311
Matched demos schools: 17311 (12.9%)
Matched atlas schools: 17311 (98.8%)
Dropped demos schools: 116601 (87.1%) (elementary/middle schools dropped)
Dropped atlas schools: 214 (1.2%) (private schools, charter schools, with overlaps such that the name appears somewhere else under different nces id)
Dropped 3353 matched schools with no bias_own_ses_hs (can't be modeled without the target)
Wrote 13958 rows to data/interim/gs_demos_with_social_capital.csv
