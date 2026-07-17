# stage0_data_summary

# Notes on research process:

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
Wrote 17311 rows to data/interim/gs_demos_with_social_capital.csv
