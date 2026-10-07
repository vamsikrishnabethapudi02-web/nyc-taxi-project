# Challenges faced so far (data and cleaning)

Owner: Member 1. Numbers in **[brackets]** come from `data/processed/cleaning_summary.json`
after running `01_cleaning.ipynb`. Delete any challenge that did not actually happen to you, and add real ones.

## 1. File size and memory
The four files hold **47.2 million rows**. Our first run loaded all four on a free Colab T4 GPU and failed with a CUDA out-of-memory error when combining the files (earlier, plain RAM ran out too).
- **What we did:** read only the columns we need; the notebook uses GPU-accelerated cuDF when available and falls back to pandas; keep a random **25%** of every file right after loading it (**11.8 million** rows), so memory stays small but all months and days are still represented; take a 200,000-row sample for plots.
- **Measured:** the final cleaning run took **521 seconds** on **pandas (CPU)**.

## 2. Missing values were not the main problem
Across 11.8 million rows only **one** value was missing (one `improvement_surcharge`), per Figure 2. The data is "clean" in the null sense but full of **invalid values** (zero or negative fares, 0-passenger trips, GPS at (0, 0), dropoff before pickup). A plain `dropna()` would have changed nothing.
- **What we did:** 11 explicit validity rules in an ordered "row funnel", so each removal is counted and explainable.

## 3. Deciding what counts as an outlier
Taxi data is naturally skewed: a few airport or out-of-town trips are real and expensive. Statistical trimming (e.g. IQR) would delete legitimate long trips.
- **What we did:** used hard, domain-based limits (duration over 3 hours, distance over 100 miles, fare over $500, speed over 80 mph) and **kept** skewed-but-possible values. EDA uses log scales instead.
- **Trade-off:** thresholds are judgment calls. We removed **315,764 rows (2.67%)** in total.

## 4. Impossible trips that look valid column by column
Some rows pass every single-column check but fail when combined (e.g. 40 miles in 4 minutes).
- **What we did:** derived `duration_min` and `speed_mph` and added a speed rule (removed **749** rows after the earlier rules; **9,591** rows break it on their own).

## 5. GPS errors
Missing GPS fixes appear as longitude/latitude (0, 0) and stray points far from New York (Figure 4).
- **What we did:** kept only pickups and dropoffs inside a bounding box around the five boroughs (removed **184,898** rows, **1.56%** of all rows, the largest single step).

## 6. Tips are not recorded for cash trips
Per the data dictionary, `tip_amount` is not captured for cash payments, so it is 0 for nearly all of them. Using all rows would make tipping look much lower than it really is.
- **What we did:** `tip_pct` is only computed for credit-card trips (`payment_type == 1`). Tip analysis must filter the same way.

## 7. Column names differ between files (a silent bug we caught)
The January 2015 file calls the column `RateCodeID` but the 2016 files call it `RatecodeID`. Our first version matched only one spelling, so every 2015 row got an empty rate code and was removed by the rate-code rule. That single rule removed about **26% of all rows (3.1 million)**, and the clean data contained **no January 2015 trips at all**.
- **How we noticed:** the row funnel showed one rule removing far more than any other, and the per-file check showed a file missing.
- **What we did:** column names are now matched ignoring capitalisation, the notebook prints which names it harmonised, and it warns if any step removes more than 5% of rows or if a source file disappears. After the fix, **97.3%** of rows are kept and all four files are present (about 3.09M, 2.66M, 2.77M and 2.98M clean rows per file).
- **Also:** the notebook reported no other column differences between the files.

## 8. Sharing a very large dataset in Git
GitHub rejects files over 100 MB.
- **What we did:** raw and full processed files are git-ignored; `data/raw/README.md` explains how to download; we commit the cleaning notebook, summary JSON, figures, and a 50,000-row clean sample.

## Still open / risks
- Thresholds (3 h, 100 mi, $500, 80 mph) could be tuned if EDA suggests we cut too much or too little.
- **[Add anything the team is unsure about.]**
