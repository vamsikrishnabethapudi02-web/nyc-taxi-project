# Project question

> **How do time of day, day of week and pickup location shape NYC yellow taxi fares and tips, and can we predict whether a trip will be high-fare using only what is known at pickup?**

## What this means
- **Fares and tips:** how fare amount and tip percentage vary with hour, weekday, month and pickup area.
- **Prediction (preliminary ML direction):** classify each trip as high-fare or not. High-fare is rare, so the classes are imbalanced, which suggests **classification with resampling** (for example SMOTE or class weights). No results are required for the mid-presentation, only the justification from EDA.
- **Definition of "high-fare":** decide from the EDA (for example the top 20% of fares, or above a fare threshold seen in the fare distribution). Write the chosen definition and the reason in `ml_direction.md`.

## Avoid data leakage
"Known at pickup" means these may be used as inputs: pickup time (hour, weekday, month), pickup location, passenger count, vendor, rate code.
These must **not** be inputs, because they are only known after the trip: `trip_distance`, `duration_min`, `speed_mph`, `dropoff_*`, `tip_amount`, `total_amount`, `extra`, `mta_tax`, `tolls_amount`.

## Tips: credit-card trips only
Cash tips are not recorded, so `tip_amount` is 0 for almost all cash trips. For any tip analysis filter `payment_type == 1` and use `tip_pct`.

## Planned visuals (6 total)
Member 2 (EDA A)
1. Average fare by hour and weekday (heatmap)
2. Fare against distance (hexbin or log scale), split by January 2015 and 2016
3. Pickup map coloured by average fare

Member 3 (EDA B)
4. Tip percentage by hour (credit-card trips only)
5. Share of high-fare trips by pickup area
6. January 2015 against January 2016: trip volume and average fare

Each visual needs one written insight and the decision it led to (for example "fare grows roughly linearly with distance but with a strong airport cluster, so keep airport trips and add a pickup-zone feature").

## Data everyone uses
`yellow_taxi_clean_1.parquet`: a random 25% sample of each of the four files, 11,496,446 rows, 27 columns. Use the same file so all numbers match.
