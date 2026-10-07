# Dashboard Insights

**Made by:** Vamsi Krishna Bethapudi (Member 4)
**Tool:** Tableau (file: `NYC_Taxi_Dashboard.twbx`)
**Data:** the same cleaned 25% sample the whole team used (about 11.5 million trips)

## What's on the dashboard

We put 5 charts on one screen:

1. Pickups by day and hour (heatmap)
2. Share of high-fare trips by pickup area
3. Tip % by hour
4. Average fare by pickup area (new, only on the dashboard)
5. Trips with no tip, by hour (new, only on the dashboard)

You can switch between weekdays and weekends, hover over anything to see the numbers, and click an area to highlight it in both area charts.

## What we found

### 1. Airport rides cost way more

A ride from JFK costs about $46 on average. From LaGuardia it's about $31. From Midtown Manhattan it's only about $11.

So an airport pickup costs roughly 4 times as much as a normal city ride. Airports are only about 4% of all pickups, but almost every airport ride (93–96%) ends up in our "high-fare" group (fare over $16).

**What this means for us:** where the ride starts is the best clue we have for guessing whether it will be expensive.

### 2. People skip the tip more late at night

We only looked at card payments, because cash tips aren't recorded.

During the day, about 3 out of 100 card riders leave no tip. Around 3–5 am, that goes up to about 8 out of 100. That's nearly 3 times as many.

We don't know why yet. We only see the pattern in the data.

### 3. Busy doesn't mean expensive

The busiest times are weekday evenings (6–9 pm) and weekend nights after midnight. Most of those rides are short trips inside Manhattan.

The expensive rides come from the airports, and they happen at all hours.

**In short:** the time of day tells you how busy it will be. The pickup place tells you how much the ride will cost.

## How this connects to our project question

Our question was: how do time, day, and pickup location affect fares and tips, and can we predict an expensive trip at pickup?

- **Time and day** mostly change *how many* rides there are, and a little bit *how people tip*.
- **Pickup location** mostly changes *how much* a ride costs.

That's why our model plan uses pickup area as the main input to predict whether a trip will be high-fare.
