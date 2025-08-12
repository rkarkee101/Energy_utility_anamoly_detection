import pandas as pd
from sklearn.ensemble import IsolationForest

# 1. Load data from CSV (general utility bill data for one school or filtered by school ID)
df = pd.read_csv('utility_data.csv')  # replace with actual file name
# If multiple schools are present, filter by a specific School ID (e.g., SchoolID == 1)
# df = df[df['SchoolID'] == 1]

# Parse dates and sort chronologically
df['Date'] = pd.to_datetime(df['billing date'])  # assuming column name is 'billing date'
df = df.sort_values('Date')

# 2. Feature engineering: create month feature for seasonality (as numeric or one-hot)
df['Month'] = df['Date'].dt.month

# We will focus on electricity usage (kWh). Normalize by square footage if needed for consistency.
df['kWh_per_sqft'] = df['energy use in kWh'] / df['square footage']

# 3. Apply anomaly detection model (IsolationForest)
# Use kWh_per_sqft and month as features to incorporate seasonal context
features = df[['Month', 'kWh_per_sqft']].values
model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
model.fit(features)

# 4. Identify point anomalies
df['anomaly_score'] = model.decision_function(features)   # higher score = more normal
df['anomaly_label'] = model.predict(features)             # 1 = normal, -1 = anomaly
# Points predicted as -1 are anomalies
point_anomalies = df[df['anomaly_label'] == -1]

print("Detected point anomalies:")
for date, usage in zip(point_anomalies['Date'], point_anomalies['energy use in kWh']):
    print(f"  {date.date()} -> {usage:.2f} kWh (outlier)")

# 5. Detect small trend anomalies (collective anomalies)
# We check for any 3 consecutive months with unusually high usage.
# One simple approach: compute a rolling 3-month average and see if it's significantly above historical average.
df['rolling3_avg'] = df['kWh_per_sqft'].rolling(window=3, min_periods=3).mean()
overall_mean = df['kWh_per_sqft'].mean()
overall_std  = df['kWh_per_sqft'].std()

collective_anomalies = []
for i in range(len(df) - 2):
    window_avg = df.iloc[i:i+3]['kWh_per_sqft'].mean()
    # If the 3-month window average is more than 1 std above overall mean (moderate threshold),
    # and none of these points were already flagged as point anomalies, flag this window as a trend anomaly.
    if window_avg > overall_mean + overall_std:
        window_dates = df.iloc[i:i+3]['Date'].dt.strftime('%Y-%m-%d').tolist()
        if not set(window_dates).issubset(set(point_anomalies['Date'].dt.strftime('%Y-%m-%d'))):
            collective_anomalies.append(window_dates)

# Remove duplicate sequences (overlapping windows) for clarity
# (If a long trend exists, it will appear in overlapping windows; we can merge them)
merged_trends = []
for seq in collective_anomalies:
    if not merged_trends or set(seq).isdisjoint(set(merged_trends[-1])):
        merged_trends.append(seq)
# merged_trends now holds distinct collective anomaly sequences

print("\nDetected collective anomalies (anomalous trends):")
for seq in merged_trends:
    print("  Anomalous period:", " to ".join([seq[0], seq[-1]]))
