# stage3_concept_validation

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
