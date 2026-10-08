# NYC Yellow Taxi: Data Visualization Group Project (DATA-230)

> Team (Group 9): Shashidhar Kumar Chintagunta (Member 1) · Sai Shivananda Gunda (Member 2) · Geethaamruth V. Tirumalasetty (Member 3) · Vamsi Krishna Bethapudi (Member 4)
> Dashboard: [`NYC_Taxi_Dashboard.twbx`](NYC_Taxi_Dashboard.twbx) (Tableau packaged workbook) · Presentation: [`NYC_Taxi_Group9_Slides.pdf`](NYC_Taxi_Group9_Slides.pdf)

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
data/processed/      cleaned sample + cleaning_summary.json + eda_part2_summary.json
data/tableau/        small aggregated tables for the Tableau visuals (tip_by_hour, high_fare_by_area, daily_jan_compare)
notebooks/           01_cleaning · 02_eda_part1 · 03_eda_part2
figures/             cleaning_*.png · eda_1-6.png
tableau/             Tableau packaged workbooks (.twbx) for visuals 4-6
ml_direction.md      proposed ML task and its justification from the EDA
presentation/        slides
```

*Note: for the mid-presentation upload, the files sit in the repository root (notebooks, figures, CSVs and workbooks side by side). The layout above is the intended structure.*

## 3. How to run
1. `pip install -r requirements.txt` (optional: RAPIDS cuDF for GPU)
2. Download the raw data as described in `data/raw/README.md`
3. Run `notebooks/01_cleaning.ipynb`, then the EDA notebooks
4. For `notebooks/03_eda_part2.ipynb`, use a GPU runtime (Colab: Runtime → Change runtime type → T4 GPU). The first cell turns on RAPIDS `cudf.pandas`; without a GPU it falls back to plain pandas. On Colab, put `yellow_taxi_clean.parquet` and `cleaning_summary.json` in **My Drive → nyc-taxi** and the notebook copies them in.

## 4. Exploratory data analysis

### 4a. EDA part A *(Member 2)*
Notebook: `eda_part1.ipynb` · figures: `eda1.png`, `eda2.png`, `eda3.png` · Tableau: `TableauEDA1.twb` · decisions: `eda_part1_insights.md`

| # | Visual | Insight | Decision |
|---|---|---|---|
| 1 | Pickups by weekday and hour (heatmap) | Busiest: Friday 19:00 (6,743 pickups on average). Quietest: Tuesday 03:00 (427). | Keep pickup hour, weekday and a weekend flag; busy periods are not necessarily expensive. |
| 2 | Pickups by taxi zone (map) | Manhattan holds 92.8% of mapped pickups; the biggest zone is Upper East Side South (431,383). | Use pickup zone and an airport flag as features; keep valid airport trips. |
| 3 | Trip length by passenger group (grouped bars) | Long-trip share is 13.4% (1 passenger), 15.0% (2), 14.2% (3-6): no steady rise with group size. | Keep passenger count as a candidate, but do not expect it to be strong. |

Full write-up: `eda_part1_insights.md`.

### 4b. EDA part B *(Member 3)* · `notebooks/03_eda_part2.ipynb`

**GPU acceleration (RAPIDS cuDF).** The notebook loads `cudf.pandas`, so ordinary pandas code runs on the GPU. It loads all **11,496,446** clean trips and runs the filtering, pickup-area assignment and every group-by on a Colab T4 GPU: **15.4 seconds** in total. The results are written to small tables in `data/tableau/` (24-62 rows each), and the visuals are built in Tableau from those tables. The heavy work happens once on the GPU, and every number in Tableau and on the slides comes from the same run (`data/processed/eda_part2_summary.json`).

**High-fare definition (used in Visual 5 and the ML task).** A trip is *high-fare* if `fare_amount` is above the 80th percentile, **$16.00**. That makes **19.0%** of trips high-fare (an imbalance of about 1 : 4.3).

**Pickup areas.** The raw data has GPS coordinates but no zone IDs, so the notebook assigns approximate areas: boxes around JFK and LaGuardia, an outline of Manhattan split at 14th St and 59th St, and simple boundaries for the other boroughs.

| # | Visual | Data (`data/tableau/`) | Tableau workbook (`tableau/`) | Figure |
|---|---|---|---|---|
| 4 | Card tip % by hour, weekday vs weekend | `tip_by_hour.csv` | `member3_visual4_tips_by_hour.twbx` | `figures/eda_4.png` |
| 5 | Share of high-fare trips by pickup area | `high_fare_by_area.csv` | `member3_visual5_high_fare_by_area.twbx` | `figures/eda_5.png` |
| 6 | January 2015 vs January 2016: trips per day and average fare | `daily_jan_compare.csv` | `member3_visual6_jan_2015_vs_2016.twbx` | `figures/eda_6.png` |

**Visual 4: Card tips stay near 20-22% all day; the weekday 4-8 pm bump matches the rush-hour surcharge**
- *Insight:* on credit-card trips (65.7% of all trips) the median tip is **21.9%** of the fare. On weekdays the average ranges only from **20.1% (06:00)** to **22.4% (16:00)**. The one clear change is a step up at exactly 16:00-19:00 on weekdays, which is not seen on weekends. That matches NYC's $1 weekday 4-8 pm surcharge: card tip suggestions are a percentage of the *total*, so the tip as a share of the *fare* rises. Only 3.5% of card trips leave no tip.
- *Decision:* tipping barely changes with time, and the tip is only known after the trip, so the model predicts the **fare**, not the tip. Tip analysis uses card trips only, because cash tips are not recorded.

**Visual 5: Airport pickups are over 93% high-fare; Midtown and Uptown only 13-14%**
- *Insight:* airports are only **4.3%** of pickups but **21.6%** of all high-fare trips: JFK **96.3%**, LaGuardia **93.3%**. Queens, Brooklyn and the Bronx are 28-30%, Downtown Manhattan 20.4%, and Midtown and Uptown Manhattan only **13.8% / 13.4%**, even though Manhattan is over 90% of all pickups.
- *Decision:* pickup location is the strongest signal known at pickup, so it becomes a model feature. Airport trips are real trips, not outliers, so they stay in the data.

**Visual 6: 14% fewer trips per day in January 2016, but fares 4.6% higher**
- *Insight:* January 2016 had **14.2% fewer trips per day** than January 2015, while the average fare rose **4.6%** ($11.84 → $12.39). Each January has one collapse day, **27 Jan 2015 (-68%)** and **23 Jan 2016 (-78%)**, both blizzards with an NYC travel ban.
- *Decision:* year/month go into the model, and the two years are not treated as identical. Storm days are real behaviour, so they are kept and annotated, and weather is a possible future feature.

## 5. ML direction *(Member 3)* · `ml_direction.md`

**Task:** binary classification. *Will this trip be high-fare (fare > $16.00)?* The model uses only information known **at pickup**. No model results are reported for the mid-presentation.

**Why this task, from the EDA**
- **Imbalanced target:** only 19.0% of trips are positive, so a model that always says "no" would be 81% accurate and useless. The plan is to use **class weights or SMOTE**, and to judge models by **F1 and PR-AUC**, not accuracy.
- **Location separates the classes** (Visuals 3 and 5): airports are 93-96% high-fare, Midtown and Uptown about 13%.
- **Time matters** (Visual 1): the high-fare share doubles from 15.7% at 19:00 to 31.8% at 04:00, one of the quietest hours.
- **Year matters** (Visual 6): trip volume and fares both shifted between January 2015 and January 2016.
- **Why not regression:** the distance, which drives most of the fare, is unknown at pickup, so predicting the exact dollar amount is unrealistic. A yes/no "expensive trip" question is the realistic one.

**Features (known at pickup):** hour, weekday, weekend flag, month/year, pickup taxi zone (or pickup area), passenger count (expected to be weak, see Visual 2), vendor.

**Excluded to avoid data leakage** (only known after the trip): `trip_distance`, `duration_min`, `speed_mph`, `dropoff_*`, `tip_amount`, `total_amount`, `extra`, `mta_tax`, `tolls_amount`, `payment_type`.

**Open decision, rate code:** `RatecodeID` is set at pickup, but codes 2-5 (JFK flat fare, Newark, out-of-town, negotiated) are only 2.1% of trips and are 93-100% high-fare, so they nearly give the answer away. We will train **with and without** rate code and report both. The harder and more useful problem is the standard-rate trips (17.2% high-fare).

**Limitations:** Visual 5 uses approximate GPS areas; the final model will use the official taxi zones that Visual 3 already uses. Weather is not in the data.

Full details, including a summary of all six visuals and the decision each one led to: `ml_direction.md`.

## 6. Dashboard *(Member 4)*
**Tool:** Tableau (class tool) · **File:** [`NYC_Taxi_Dashboard.twbx`](NYC_Taxi_Dashboard.twbx), a packaged workbook, so it opens in Tableau with the data inside · **Write-up:** [`dashboard_insights.md`](dashboard_insights.md) (also as [PDF](dashboard_insights.pdf))

One screen, five views, built on the same cleaned 25% sample as the rest of the project (11.5M trips):

| View | What it shows | Where it comes from |
|---|---|---|
| Pickups by day and hour (heatmap) | When taxis are busy | EDA visual 1 |
| High-fare share by pickup area | Where expensive trips start | EDA visual 5 |
| Card tip % by hour | How tipping changes through the day | EDA visual 4 |
| **Average fare by pickup area** | What a ride costs by area | **New, dashboard only** |
| **No-tip trips by hour** | How often riders leave no tip | **New, dashboard only** |

A *Day type* filter (weekday / weekend) controls the hour charts, every mark has a hover tooltip, and clicking an area highlights it in both area charts.

**What we found**
1. **Airport rides cost about 4x more.** JFK averages $46, LaGuardia $31, Midtown $11. Airports are 4.3% of pickups but 21.6% of high-fare trips, and 93-96% of airport rides are high-fare.
2. **Late-night riders skip the tip most.** On card trips, about 8% leave no tip at 3-5 am, against about 3% in the daytime. We see the pattern but did not test the reason.
3. **Busy does not mean expensive.** The busiest hours (weekday evenings, weekend nights after midnight) are mostly short Manhattan trips. The expensive trips start at the airports, at any hour.

**Link to the project question:** time and day mostly change *how many* trips there are; pickup location mostly changes *how much* a trip costs. That is why pickup area is the main input of our ML plan.

**Presentation:** [`NYC_Taxi_Group9_Slides.pdf`](NYC_Taxi_Group9_Slides.pdf), 15 slides. The last slide lists each member's files and commits.

**Member 4 contribution:** `NYC_Taxi_Dashboard.twbx`, `dashboard_insights.md` / `.pdf`, `NYC_Taxi_Group9_Slides.pdf`, and the README header and section 6.
