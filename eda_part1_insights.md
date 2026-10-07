# Member 2: EDA insights and decisions

Author: Sai Shivananda Gunda

Recreated cleaned ~25% Kaggle sample | Jan 2015 and Jan–Mar 2016 | 11,495,890 in-scope trips

## 1. Heatmap

Question: When are pickups busiest?

Insight: The highest average is 6,743 pickups on Friday at 19:00. The lowest is 427 on Tuesday at 03:00.

Decision: Retain pickup hour, weekday and a weekend flag as pickup-time candidates. Compare fare and card-tip summaries for the same cells; busy periods need not be expensive.

## 2. Choropleth

Question: Which taxi zones have the most recorded pickups?

Insight: The largest count is in Upper East Side South: 431,383 pickups. Manhattan contains 92.8% of mapped pickups. 4,402 rows are unmapped.

Decision: Evaluate pickup zone and an airport flag as pickup-time features. Compare their fare summaries directly before judging predictive strength; preserve valid airport trips.

## 3. Grouped bars

Question: Do larger passenger groups take longer trips?

Insight: Long-trip shares are 1 passenger: 13.42%, 2 passengers: 14.96%, 3–6 passengers: 14.15%. There is no steady rise in long-trip share with group size.

Decision: Evaluate passenger count alongside pickup time and location. This chart alone does not establish its ML importance.

## Supporting EDA linked to fares and tips

EXECUTED: fare and group summaries; recorded card tips where available

Among cells with at least 100 valid fares, the busiest is Friday 19:00 (median fare $9.00); the highest median fare is $12.00 at Monday 04:00. If there are tied maxima, this reports the first cell in the table.
Among mapped zones with at least 100 valid fares, Baisley Park has the highest median fare ($52.00); ties report the first zone. This is a descriptive comparison, not model importance.

## RAPIDS/cuDF

NOT EXECUTED: RUN_GPU=False (CPU preview)

GPU work has not run. No timing or speedup claim can be made.

Scope: historical recorded yellow taxi trips in the supplied cleaned sample.
The grouped-bar comparison is descriptive, not a statistical equivalence test.