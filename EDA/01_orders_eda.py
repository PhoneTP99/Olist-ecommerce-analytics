# =============================================================================
# 01_orders_eda.py
# Exploratory Data Analysis — olist_orders_dataset.csv
# Dataset: Brazilian E-Commerce Public Dataset (Kaggle)
# =============================================================================

# =============================================================================
# 1. IMPORTS
# =============================================================================
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 2. LOAD DATA
# =============================================================================

# Find the directory where this file is saved
notebook_dir = os.path.dirname(os.path.abspath(__file__))

# Find the parent directory (main project folder)
project_root = os.path.dirname(notebook_dir)

# Build the full path to the CSV
order_file_path = os.path.join(project_root, 'Data', 'raw', 'Olist_ecom_data', 'olist_orders_dataset.csv')

# Load the dataset
order = pd.read_csv(order_file_path)

# =============================================================================
# 3. DATA SETUP — Convert date columns to datetime
# =============================================================================

order['order_purchase_timestamp']      = pd.to_datetime(order['order_purchase_timestamp'])
order['order_approved_at']             = pd.to_datetime(order['order_approved_at'])
order['order_delivered_carrier_date']  = pd.to_datetime(order['order_delivered_carrier_date'])
order['order_delivered_customer_date'] = pd.to_datetime(order['order_delivered_customer_date'])
order['order_estimated_delivery_date'] = pd.to_datetime(order['order_estimated_delivery_date'])

# Correct day order for plots
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# =============================================================================
# 4. DATA CLEANING — Check & drop problematic delivered rows
# =============================================================================

# Check missing values for delivered orders across all date columns
delivered_missing = order[order['order_status'] == 'delivered'][[
    'order_approved_at',
    'order_delivered_carrier_date',
    'order_delivered_customer_date',
    'order_estimated_delivery_date'
]].isnull().sum()

print("Missing values in delivered orders:")
print(delivered_missing)

# Drop delivered orders missing ANY key date column
order = order[~(
    (order['order_status'] == 'delivered') &
    (
        order['order_delivered_customer_date'].isnull() |
        order['order_delivered_carrier_date'].isnull()  |
        order['order_approved_at'].isnull()
    )
)]

print(f"\nRows after cleaning: {len(order)}")

# =============================================================================
# 5. DELIVERY PERFORMANCE — Delivery time distribution
# =============================================================================

# Filter only delivered orders
delivered_df = order[order['order_status'] == 'delivered'].copy()

# Calculate delivery time (carrier → customer)
delivered_df['delivery_time'] = (
    delivered_df['order_delivered_customer_date'] -
    delivered_df['order_delivered_carrier_date']
).dt.days

print(f"\nMissing delivery_time values: {delivered_df['delivery_time'].isnull().sum()}")

# Plot delivery time distribution
plt.figure(figsize=(10, 6))
sns.histplot(delivered_df['delivery_time'], bins=50, kde=True, color='purple')
plt.title('How many days does delivery take?')
plt.xlabel('Days')
plt.xlim(0, 30)
plt.tight_layout()
plt.savefig('delivery_time_distribution.png')
plt.show()

# =============================================================================
# 6. LATE DELIVERY RATE
# =============================================================================

# Calculate late deliveries
delivered_df['late_delivery'] = (
    delivered_df['order_delivered_customer_date'] >
    delivered_df['order_estimated_delivery_date']
)

# Calculate and print the late delivery rate
late_rate = delivered_df['late_delivery'].mean() * 100
print(f"\nLate Delivery Rate: {late_rate:.2f}%")

# =============================================================================
# 7. SALE PULSE — Orders by day of the week
# =============================================================================

order['weekday'] = order['order_purchase_timestamp'].dt.day_name()

plt.figure(figsize=(10, 5))
sns.countplot(data=order, x='weekday', order=days_order, palette='magma')
plt.title('Orders by Day of the Week')
plt.xlabel('Day')
plt.ylabel('Number of Orders')
plt.tight_layout()
plt.savefig('orders_by_weekday.png')
plt.show()

# =============================================================================
# 8. ORDER STATUS BREAKDOWN
# =============================================================================

status_counts = order['order_status'].value_counts()

plt.figure(figsize=(10, 6))
sns.barplot(x=status_counts.values, y=status_counts.index, palette='rocket')
plt.title('Current Status of All Orders')
plt.xlabel('Number of Orders')
plt.tight_layout()
plt.savefig('order_status_breakdown.png')
plt.show()

# =============================================================================
# 9. ORDERS BY HOUR OF THE DAY
# =============================================================================

order['hour'] = order['order_purchase_timestamp'].dt.hour

plt.figure(figsize=(10, 5))
sns.countplot(data=order, x='hour', palette='viridis')
plt.title('Orders by Hour of the Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Orders')
plt.tight_layout()
plt.savefig('orders_by_hour.png')
plt.show()

# =============================================================================
# 10. MONTHLY ORDER TREND (Growth Over Time)
# =============================================================================

# Extract year+month
order['month'] = order['order_purchase_timestamp'].dt.to_period('M')

# Group by month and count orders
monthly_orders = order.groupby('month').size().reset_index(name='orders')

# Convert Period to string for plotting
monthly_orders['month'] = monthly_orders['month'].astype(str)

plt.figure(figsize=(12, 5))
sns.lineplot(data=monthly_orders, x='month', y='orders', marker='o')
plt.title('Monthly Order Trend')
plt.xlabel('Month')
plt.ylabel('Total Orders')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('monthly_order_trend.png')
plt.show()

# =============================================================================
# 11. WEEKDAY VS HOUR HEATMAP
# =============================================================================

# Create pivot table
heatmap_data = order.pivot_table(
    index=order['order_purchase_timestamp'].dt.day_name(),
    columns=order['order_purchase_timestamp'].dt.hour,
    aggfunc='size',
    fill_value=0
)

# Reorder rows by correct day order
heatmap_data = heatmap_data.reindex(days_order)

plt.figure(figsize=(14, 6))
sns.heatmap(heatmap_data, cmap='magma')
plt.title('Order Frequency by Weekday and Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Day of Week')
plt.tight_layout()
plt.savefig('weekday_hour_heatmap.png')
plt.show()

# =============================================================================
# 12. FINAL INFO CHECK
# =============================================================================

print("\n--- order DataFrame ---")
order.info()

print("\n--- delivered_df DataFrame ---")
delivered_df.info()