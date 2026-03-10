import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────
print("Loading data...")
orders    = pd.read_csv('Data/processed/orders_cleaned.csv')
customers = pd.read_csv('Data/processed/customers_cleaned.csv')
payments  = pd.read_csv('Data/processed/payments_order_level.csv')

# ─────────────────────────────────────────
# 2. MERGE
# ─────────────────────────────────────────
print("Merging datasets...")
df = orders.merge(customers, on='customer_id', how='left') \
           .merge(payments,  on='order_id',    how='left')

# ─────────────────────────────────────────
# 3. CONVERT DATE COLUMNS
# ─────────────────────────────────────────
date_cols = [
    'order_purchase_timestamp',
    'order_approved_at',
    'order_delivered_carrier_date',
    'order_delivered_customer_date',
    'order_estimated_delivery_date'
]
df[date_cols] = df[date_cols].apply(pd.to_datetime)
df['late_delivery'] = df['late_delivery'].astype(bool)
df['num_payments']  = df['num_payments'].astype('Int64')

# ─────────────────────────────────────────
# 4. FILTER: DELIVERED ORDERS ONLY
# ─────────────────────────────────────────
df = df[df['order_status'] == 'delivered'].copy()
print(f"Delivered orders: {len(df)}")

# ─────────────────────────────────────────
# 5. KEEP ONLY RFM-RELEVANT COLUMNS
# ─────────────────────────────────────────
df = df[[
    'customer_unique_id',
    'order_id',
    'order_purchase_timestamp',
    'total_value'
]]

# ─────────────────────────────────────────
# 6. FIX MISSING VALUE IN total_value
# ─────────────────────────────────────────
missing = df['total_value'].isnull().sum()
print(f"Missing total_value rows: {missing} → dropping them")
df = df.dropna(subset=['total_value']).reset_index(drop=True)

# ─────────────────────────────────────────
# 7. RFM FEATURE ENGINEERING
# ─────────────────────────────────────────
print("Building RFM features...")
snapshot_date = df['order_purchase_timestamp'].max() + pd.Timedelta(days=1)
print(f"Snapshot date: {snapshot_date}")

rfm = df.groupby('customer_unique_id').agg(
    Recency   = ('order_purchase_timestamp', lambda x: (snapshot_date - x.max()).days),
    Frequency = ('order_id', 'nunique'),
    Monetary  = ('total_value', 'sum')
).reset_index()

print(f"RFM table shape: {rfm.shape}")
print(rfm.describe().round(2))

# ─────────────────────────────────────────
# 8. SCALING (Log + StandardScaler)
# ─────────────────────────────────────────
print("\nScaling RFM features...")
rfm_log = rfm[['Recency', 'Frequency', 'Monetary']].copy()
rfm_log['Recency']   = np.log1p(rfm_log['Recency'])
rfm_log['Frequency'] = np.log1p(rfm_log['Frequency'])
rfm_log['Monetary']  = np.log1p(rfm_log['Monetary'])

scaler     = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_log)
rfm_scaled = pd.DataFrame(rfm_scaled, columns=['Recency', 'Frequency', 'Monetary'])

# ─────────────────────────────────────────
# 9. K-MEANS CLUSTERING  (K=4)
# ─────────────────────────────────────────
print("\nRunning K-Means (K=4)...")
K      = 4
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

print("Cluster sizes:")
print(rfm['Cluster'].value_counts().sort_index())

# ─────────────────────────────────────────
# 10. AUTO-LABEL SEGMENTS
# ─────────────────────────────────────────
profile = rfm.groupby('Cluster').agg(
    R=('Recency',   'mean'),
    F=('Frequency', 'mean'),
    M=('Monetary',  'mean')
)
# Score: low Recency = good, high F and M = good
profile['score']   = -profile['R'] + profile['F'] * 50 + profile['M'] / 10
rank_map           = profile['score'].rank(ascending=False).astype(int)
label_map          = {1: 'Champions', 2: 'Promising', 3: 'At Risk', 4: 'Hibernating'}
segment_map        = rank_map.map(label_map).to_dict()
rfm['Segment']     = rfm['Cluster'].map(segment_map)

print("\nSegment counts:")
print(rfm['Segment'].value_counts())

# ─────────────────────────────────────────
# 11. ADD EXTRA COLUMNS USEFUL FOR POWER BI
# ─────────────────────────────────────────
# Recency bucket (for slicers)
rfm['Recency_Band'] = pd.cut(
    rfm['Recency'],
    bins=[0, 30, 90, 180, 365, 9999],
    labels=['0-30d', '31-90d', '91-180d', '181-365d', '365d+']
)

# Frequency bucket
rfm['Frequency_Band'] = pd.cut(
    rfm['Frequency'],
    bins=[0, 1, 2, 5, 9999],
    labels=['1 order', '2 orders', '3-5 orders', '6+ orders']
)

# Monetary tier
rfm['Monetary_Tier'] = pd.cut(
    rfm['Monetary'],
    bins=[0, 100, 300, 600, 9999999],
    labels=['Low (<100)', 'Mid (100-300)', 'High (300-600)', 'VIP (600+)']
)

# ─────────────────────────────────────────
# 12. EXPORT CSV FOR POWER BI
# ─────────────────────────────────────────
os.makedirs('Data/processed', exist_ok=True)
output_path = 'Data/processed/rfm_segments.csv'
rfm.to_csv(output_path, index=False)

print(f"\n✅ Saved to: {output_path}")
print(f"   Rows    : {len(rfm)}")
print(f"   Columns : {list(rfm.columns)}")
print("\nPreview:")
print(rfm.head(10).to_string(index=False))