import pandas as pd
from sklearn.ensemble import IsolationForest

# 1. Load pre- and post-retrofit data
pre_df = pd.read_csv('preretrofit.csv')
post_df = pd.read_csv('postretrofit.csv')

# Ensure date parsing if needed and sort (though not strictly necessary for this part)
pre_df['Date'] = pd.to_datetime(pre_df['billing date'])
post_df['Date'] = pd.to_datetime(post_df['billing date'])
pre_df = pre_df.sort_values(['School ID', 'Date'])
post_df = post_df.sort_values(['School ID', 'Date'])

# 2. Compute energy use per square foot (EUI) for each record
pre_df['Electricity_EUI'] = pre_df['energy use in kWh'] / pre_df['square footage']
post_df['Electricity_EUI'] = post_df['energy use in kWh'] / post_df['square footage']
# (If gas usage is significant, you might do similar for gas or convert to a common unit and combine into a total EUI.)

# 3. Calculate annual EUI and percentage change for each school
pre_agg = pre_df.groupby('School ID').agg({
    'Electricity_EUI': 'mean',  # average monthly EUI (assuming same number of months)
    'energy use in kWh': 'sum',
    'square footage': 'first'   # square footage is constant per school
}).reset_index().rename(columns={'Electricity_EUI':'Pre_avgEUI','energy use in kWh':'Pre_total_kWh'})
post_agg = post_df.groupby('School ID').agg({
    'Electricity_EUI': 'mean',
    'energy use in kWh': 'sum',
    'square footage': 'first'
}).reset_index().rename(columns={'Electricity_EUI':'Post_avgEUI','energy use in kWh':'Post_total_kWh'})

# Merge pre and post info on School ID
performance = pd.merge(pre_agg, post_agg, on=['School ID', 'square footage'])
# Compute annual EUI (if data covers one year each, total_kWh/year per sqft is another way to get annual EUI)
performance['Pre_annualEUI'] = performance['Pre_total_kWh'] / performance['square footage']
performance['Post_annualEUI'] = performance['Post_total_kWh'] / performance['square footage']
# Percent change in EUI (positive means increase in usage - bad, negative means usage reduced)
performance['EUI_percent_change'] = (performance['Post_annualEUI'] - performance['Pre_annualEUI']) / performance['Pre_annualEUI'] * 100

# 4. Detect underperforming building(s) using anomaly detection on percent changes
X = performance[['EUI_percent_change']].values
model = IsolationForest(contamination=0.2, random_state=0)  # assume maybe 1 in 5 could be an outlier
model.fit(X)
performance['anomaly_label'] = model.predict(X)  # -1 = anomaly (underperformer)
underperformers = performance[performance['anomaly_label'] == -1]

if not underperformers.empty:
    print("Underperforming building(s) detected (post-retrofit anomalies):")
    for sid, change in zip(underperformers['School ID'], underperformers['EUI_percent_change']):
        print(f"  School ID {sid}: EUI change = {change:.1f}% (anomalous)")
else:
    print("No underperforming buildings detected (all retrofitted buildings within normal performance range).")

# 5. (Optional) Detailed analysis for flagged buildings
for sid in underperformers['School ID']:
    school_post = post_df[post_df['School ID'] == sid].copy()
    # Apply the Part 1 anomaly detection on this school's post-retrofit data
    # (For brevity, we reuse a simplified approach: flag months above pre-retrofit mean + threshold)
    baseline_eui = performance[performance['School ID']==sid]['Pre_annualEUI'].iloc[0]
    threshold = baseline_eui * 1.1  # e.g., 10% above pre-retrofit annual EUI
    anomalies = school_post[school_post['Electricity_EUI'] > threshold]
    if not anomalies.empty:
        print(f"\nDetailed anomalies for School ID {sid}:")
        for date, eui in zip(anomalies['Date'], anomalies['Electricity_EUI']):
            print(f"  {date.date()} -> EUI {eui:.4f} kWh/sqft (above expected)")
