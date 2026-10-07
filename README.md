# NYC Yellow Taxi: Data Visualization Group Project (DATA-230)

> Team: Shashidhar · Shivanand · Amruth · Vamsi

**Project question:** *How do time of day, day of week and pickup location shape NYC yellow taxi fares and tips, and can we predict whether a trip will be high-fare using only what is known at pickup?*

Details, definitions and the planned visuals: see `PROJECT_QUESTION.md`.

---

## 1. Dataset *(Member 1)*

| | |
|---|---|
| **Source** | NYC Taxi & Limousine Commission (TLC) Trip Record Data, via Kaggle: [NYC Yellow Taxi Trip Data](https://www.kaggle.com/datasets/elemento/nyc-yellow-taxi-trip-data) |
| **Format / size** | 4 monthly CSV files, **47.2 million trips** in total (12.7M + 10.9M + 11.4M + 12.2M). Rows used: **11.8 million** (random 25% of each file) · Clean rows: **11.5 million** (97.3% kept) *(from `data/processed/cleaning_summary.json`)* |
| **Period** | January 2015 and January to March 2016 |
| **Granularity** | One row per taxi trip |

**Features (19 raw columns)**

| Group | Columns |
|---|---|
| Time | `tpep_pickup_datetime`, `tpep_dropoff_datetime` |
| Location (GPS) | `pickup_longitude`, `pickup_latitude`, `dropoff_longitude`, `dropoff_latitude` |
| Trip | `trip_distance`, `passenger_count`, `RatecodeID`, `store_and_fwd_flag`, `VendorID` |
| Money | `fare_amount`, `extra`, `mta_tax`, `tip_amount`, `tolls_amount`, `improvement_surcharge`, `total_amount` |
| Payment | `payment_type` |

**Why this dataset**
- *Relevance:* fares, tips, time and location are exactly the variables in our project question, and the data supports the ML direction the assignment asks for: classifying high-fare trips (an imbalanced classification problem).
- *Rich for visualization:* timestamps (hour/weekday patterns), GPS coordinates (maps), money (distributions, relationships), categorical codes (payment, rate).

**Why it is complex**
- **Large:** 47.2 million rows across four files. Ordinary pandas workflows are slow and memory-hungry, and a free GPU runs out of memory if all rows are loaded at once, so we keep a random 25% sample of every file. The notebook uses cuDF when a GPU is available and pandas otherwise (our final cleaning run: pandas on CPU, about 9 minutes).
- **Multi-feature:** time, geospatial, monetary and categorical variables in one table.
- **Messy:** raw taxi meter data contains impossible values, and we remove them with 11 documented rules (below).

**Cleaning summary**

| Step | Rows removed | % of raw |
|---|---:|---:|
| Exact duplicates | 61 | 0.001% |
| Missing key field | 0 | 0.000% |
| Pickup outside study period | 0 | 0.000% |
| Dropoff not after pickup | 12,989 | 0.110% |
| Duration < 1 min or > 3 h | 97,907 | 0.829% |
| Passengers not in 1-6 | 1,706 | 0.014% |
| Distance <= 0 or > 100 mi | 13,443 | 0.114% |
| Fare <= 0 or > $500 | 3,882 | 0.033% |
| Total amount <= 0 | 1 | 0.000% |
| GPS outside NYC | 184,898 | 1.565% |
| Rate code / payment type invalid | 128 | 0.001% |
| Speed > 80 mph | 749 | 0.006% |
| **Clean rows kept** | **11,496,446** | **97.33%** |

Rules, rationale, and before/after figures: `notebooks/01_cleaning.ipynb`, `figures/cleaning_*.png`. Problems met along the way: `challenges.md`.

**Processed data:** `data/processed/yellow_taxi_clean.parquet` (not committed; regenerate with the notebook). A 50,000-row sample, `yellow_taxi_clean_sample.csv`, is committed for quick exploration.

**Columns added during cleaning:** `duration_min`, `speed_mph`, `pickup_hour`, `pickup_dow`, `pickup_month`, `is_weekend`, `tip_pct` (credit-card trips only).

## 2. Repository layout

```
data/raw/            download instructions only (files are too large for GitHub)
data/processed/      cleaned sample + cleaning_summary.json
notebooks/           01_cleaning · 02_eda_part1 · 03_eda_part2
figures/             cleaning_*.png · eda_1-6.png
presentation/        slides
```

## 3. How to run
1. `pip install -r requirements.txt` (optional: RAPIDS cuDF for GPU)
2. Download the raw data as described in `data/raw/README.md`
3. Run `notebooks/01_cleaning.ipynb`, then the EDA notebooks

## 4. EDA · 5. ML direction · 6. Dashboard
*(owned by Members 2, 3 and 4; add your sections here)*
