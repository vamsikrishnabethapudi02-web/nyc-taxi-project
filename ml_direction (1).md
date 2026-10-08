# ML direction (Member 3)

> Mid-presentation: **direction and justification only, no model results.**
> Numbers come from `notebooks/03_eda_part2.ipynb` (11,496,446 clean trips, a 25% random sample of all four files), run on RAPIDS cuDF (GPU) in 15.4 s.

## What the six EDA visuals tell us (all decisions)
| # | Visual (owner) | Finding | Decision for the model |
|---|---|---|---|
| 1 | Pickups by weekday x hour heatmap (M2) | Weekdays peak 18:00-21:00 (up to ~6,700 pickups/hour on Friday 19:00); weekend nights stay busy after midnight (~5,500 pickups at 00:00 vs ~1,900 on Mondays); 04:00-05:00 is the quietest time | Use **hour and weekday together** (weekday x hour), not hour alone |
| 2 | Trip length by passenger group (M2) | Long trips (5+ mi) are 13.4% / 15.0% / 14.2% for 1 / 2 / 3-6 passengers: almost no difference | `passenger_count` is expected to be a **weak feature**; keep it, but low priority |
| 3 | Pickups by taxi zone map (M2) | Pickups concentrate in Midtown/Lower Manhattan, with airport hotspots | Pickup **location** is central; use the official **taxi zones** as the location feature |
| 4 | Card tip % by hour (M3) | Tips stay at 20-22% all day; the only step is weekday 16:00-20:00, matching the $1 rush-hour surcharge | Predict the **fare**, not the tip (tipping barely varies, and the tip is only known after the trip) |
| 5 | High-fare share by pickup area (M3) | Airports: 4.3% of pickups but 21.6% of high-fare trips (JFK 96.3%); Midtown/Uptown only 13-14% | Pickup area is the **strongest signal**; airport trips are real, not outliers |
| 6 | January 2015 vs 2016 (M3) | 14.2% fewer trips/day in 2016, average fare +4.6%; blizzard days -68% / -78% | Add **year/month**; keep storm days, note weather as a future feature |

**Combined insight:** the busiest places and times (Midtown, weekday evenings) have the *lowest* high-fare share, while rare situations (airports, 04:00) are mostly high-fare. A model must learn these rare cases, which is another reason to handle the class imbalance.

## Proposed task
**Binary classification:** predict whether a trip will be **high-fare**, using only information known **at pickup**.

## Definition of "high-fare"
A trip is high-fare if `fare_amount` is **above $16.00**, the 80th percentile of all fares.
- **Why a percentile:** fares are strongly right-skewed (long airport trips), so a fixed round number would be arbitrary. The percentile adapts to the data and is easy to explain.
- **Result:** **19.0%** of trips are positive (slightly under 20%, because many fares sit exactly at $16.00), so the classes are **imbalanced, about 1 : 4.3**.

## Why classification with resampling (from the EDA)
1. **The target is rare.** A model that always answers "not high-fare" would be 81% accurate and useless. We will use **class weights or SMOTE** and judge models by **F1 and PR-AUC**, not accuracy.
2. **Pickup area separates the classes (Visual 5).** Airports are only **4.3%** of pickups but **21.6%** of all high-fare trips.

| Pickup area | High-fare share |
|---|---:|
| JFK Airport | 96.3% |
| LaGuardia Airport | 93.3% |
| Queens | 29.7% |
| Brooklyn | 27.9% |
| Bronx | 27.7% |
| Other (Staten Is., NJ, edges) | 21.6% |
| Manhattan: below 14th St | 20.4% |
| Manhattan: 14th to 59th St | 13.8% |
| Manhattan: above 59th St | 13.4% |
| **All trips** | **19.0%** |

3. **Time of day matters (Visuals 1 and 5).** The high-fare share doubles from **15.7% at 19:00** to **31.8% at 04:00**, and 04:00 is also one of the quietest hours in the heatmap. Late-night trips are likely longer (to the airports and outer boroughs), though we have not tested this yet. So hour, weekday and weekend are features.
4. **Year matters (Visual 6).** January 2016 had **14.2% fewer trips per day** than January 2015, while the **average fare rose 4.6%** ($11.84 → $12.39). Year/month is a feature, and the two years should not be treated as identical.
5. **Why not regression:** at pickup we do not know the distance, which drives most of the fare, so predicting the exact dollar amount is unrealistic. "Will this be an expensive trip?" is the realistic yes/no question at pickup. Clustering of pickup areas is an optional extra.

## Features (known at pickup) and leakage rules
**Use:** `pickup_hour`, `pickup_dow`, `is_weekend`, `pickup_month`/year, pickup taxi zone (or `pickup_area` / pickup lat/long), `passenger_count` (expected weak, see Visual 2), `vendorid`.

**Do not use** (only known after the trip): `trip_distance`, `duration_min`, `speed_mph`, `dropoff_*`, `tip_amount`, `total_amount`, `extra`, `mta_tax`, `tolls_amount`, `payment_type` (card or cash is only known when the trip ends).

## Open decision: rate code
`RatecodeID` is set when the trip starts, but the special codes almost *are* the answer:

| Rate code | Meaning | Trips | High-fare share |
|---|---|---:|---:|
| 1 | Standard | 11,249,775 | 17.2% |
| 2 | JFK flat fare | 215,777 | 100.0% |
| 3 | Newark | 16,621 | 100.0% |
| 4 | Nassau / Westchester | 3,128 | 94.6% |
| 5 | Negotiated | 11,119 | 93.2% |
| 6 | Group ride | 26 | 11.5% |

Codes 2 to 5 are only 2.1% of trips but are almost all high-fare. **Plan:** train **with and without** rate code and report both, so the model's skill is not just "this is a JFK flat-fare trip". The harder and more useful problem is the **standard-rate trips** (17.2% positive).

## Validation plan (final presentation)
Stratified train/test split, or a time split (train on January 2015 to February 2016, test on March 2016). Baseline: logistic regression with class weights. Then a tree model (random forest or gradient boosting). Compare SMOTE against class weights on PR-AUC and F1.

## Limitations
- Visual 5 uses approximate GPS areas. The team already maps pickups to official taxi zones (Visual 3), so the final model will use those zones instead.
- Weather causes large one-day drops (see Visual 6) and is not in the data. Weather data is a possible extra feature.
