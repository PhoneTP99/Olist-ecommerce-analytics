# =============================================================================
# 03_payment_eda.py
# Exploratory Data Analysis — olist_order_payments_dataset.csv
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

notebook_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(notebook_dir)

payment_file_path = os.path.join(project_root, 'Data', 'raw', 'Olist_ecom_data', 'olist_order_payments_dataset.csv')
payment = pd.read_csv(payment_file_path)

# =============================================================================
# 3. INITIAL DATA CHECK
# =============================================================================

print("=== Dataset Info ===")
payment.info()

print("\n=== First 10 Rows ===")
print(payment.head(10))

print("\n=== Missing Values ===")
print(payment.isnull().sum())

print("\n=== Basic Stats ===")
print(payment.describe())

# =============================================================================
# 4. DATA QUALITY CHECKS
# =============================================================================

print("\n=== Data Quality Checks ===")

# Missing values
missing_values = payment.isnull().sum()
print(f"Missing values:\n{missing_values}")

# Orders with 0 payment value
zero_payment_orders = payment[payment['payment_value'] == 0]
print(f"\nOrders with 0 payment value: {zero_payment_orders.shape[0]}")
print(payment[payment['payment_value'] == 0]['payment_type'].value_counts())

# Orders with multiple payments (duplicate order_ids)
duplicate_order_ids = payment[payment.duplicated(subset='order_id', keep=False)]
num_duplicate_order_ids = duplicate_order_ids['order_id'].nunique()
print(f"\nOrders with multiple payments: {num_duplicate_order_ids}")

# All payment types before cleaning
print(f"\nPayment types before cleaning:\n{payment['payment_type'].value_counts()}")

# =============================================================================
# 5. DATA CLEANING
# NOTE: Cleaning done BEFORE KPI calculations to ensure accurate results
# =============================================================================

# Drop undefined + 0 value rows (genuine data errors)
payment = payment[~(
    (payment['payment_type'] == 'not_defined') &
    (payment['payment_value'] == 0)
)]

print(f"\nRows after cleaning: {len(payment)}")
print(f"Payment types after cleaning:\n{payment['payment_type'].value_counts()}")

# =============================================================================
# 6. REVENUE KPIs — Row level (per payment transaction)
# =============================================================================

print("\n=== Revenue KPIs (Transaction Level) ===")

total_revenue = payment['payment_value'].sum()
print(f"Total Revenue:             R$ {total_revenue:,.2f}")

aov_transaction = payment['payment_value'].mean()
print(f"Avg Transaction Value:     R$ {aov_transaction:.2f}")

median_transaction = payment['payment_value'].median()
print(f"Median Transaction Value:  R$ {median_transaction:.2f}")

max_transaction = payment['payment_value'].max()
print(f"Max Transaction Value:     R$ {max_transaction:.2f}")

min_transaction = payment['payment_value'][payment['payment_value'] > 0].min()
print(f"Min Transaction Value:     R$ {min_transaction:.2f}")

# =============================================================================
# 7. REVENUE KPIs — Order level (grouped by order_id)
# NOTE: More accurate AOV — accounts for multi-payment orders
# =============================================================================

print("\n=== Revenue KPIs (Order Level) ===")

order_level = payment.groupby('order_id').agg(
    total_value=('payment_value', 'sum'),
    num_payments=('payment_type', 'count'),
    payment_types=('payment_type', lambda x: ','.join(x.unique()))
).reset_index()

aov_order = order_level['total_value'].mean()
print(f"True AOV (per order):      R$ {aov_order:.2f}")

median_order = order_level['total_value'].median()
print(f"Median Order Value:        R$ {median_order:.2f}")

print(f"Total Unique Orders:       {order_level['order_id'].nunique()}")

# =============================================================================
# 8. PAYMENT METHOD ANALYSIS
# =============================================================================

print("\n=== Payment Method Analysis ===")

# Count by payment type
payment_count = payment['payment_type'].value_counts().reset_index()
payment_count.columns = ['payment_type', 'count']
payment_count['percentage'] = (payment_count['count'] / len(payment) * 100).round(2)
print(payment_count)

# Revenue by payment type
revenue_by_type = payment.groupby('payment_type')['payment_value'].sum().reset_index()
revenue_by_type.columns = ['payment_type', 'total_revenue']
revenue_by_type['revenue_pct'] = (revenue_by_type['total_revenue'] / total_revenue * 100).round(2)
revenue_by_type = revenue_by_type.sort_values('total_revenue', ascending=False)
print(f"\nRevenue by payment type:\n{revenue_by_type}")

# Average order value by payment type
avg_by_type = payment.groupby('payment_type')['payment_value'].mean().reset_index()
avg_by_type.columns = ['payment_type', 'avg_value']
avg_by_type = avg_by_type.sort_values('avg_value', ascending=False)
print(f"\nAvg order value by payment type:\n{avg_by_type}")

# High value orders (>1000) by payment type
high_value_orders = payment[payment['payment_value'] > 1000]
total_high_value = high_value_orders.shape[0]
high_value_by_type = high_value_orders['payment_type'].value_counts().reset_index()
high_value_by_type.columns = ['payment_type', 'count']
high_value_by_type['pct_of_high_value'] = (high_value_by_type['count'] / total_high_value * 100).round(2)
print(f"\nHigh value orders (>R$1000): {total_high_value} ({total_high_value/len(payment)*100:.2f}% of all orders)")
print(high_value_by_type)

# =============================================================================
# 9. PAYMENT METHOD VISUALIZATIONS
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Bar chart — count
sns.barplot(data=payment_count, x='count', y='payment_type', palette='magma', ax=axes[0])
axes[0].set_title('Payment Type Usage Count')
axes[0].set_xlabel('Number of Transactions')
axes[0].set_ylabel('Payment Type')

# Bar chart — revenue
sns.barplot(data=revenue_by_type, x='total_revenue', y='payment_type', palette='rocket', ax=axes[1])
axes[1].set_title('Revenue by Payment Type')
axes[1].set_xlabel('Total Revenue (R$)')
axes[1].set_ylabel('Payment Type')

plt.tight_layout()
plt.savefig('payment_type_analysis.png')
plt.show()

# =============================================================================
# 10. INSTALLMENT ANALYSIS
# =============================================================================

print("\n=== Installment Analysis ===")

# How many pay in installments
installment_payments = payment[payment['payment_installments'] > 1]
num_installment = installment_payments.shape[0]
installment_rate = (num_installment / len(payment)) * 100
print(f"Orders with installments:  {num_installment}")
print(f"Installment Rate:          {installment_rate:.2f}%")
print(f"Single payment Rate:       {100 - installment_rate:.2f}%")

# Average installments
avg_installments = payment['payment_installments'].mean()
print(f"Avg Number of Installments: {avg_installments:.2f}")

# Installments among credit card only (most relevant)
cc_payments = payment[payment['payment_type'] == 'credit_card']
avg_cc_installments = cc_payments['payment_installments'].mean()
print(f"Avg Installments (Credit Card only): {avg_cc_installments:.2f}")

# Correlation between installments and order value
correlation = payment['payment_installments'].corr(payment['payment_value'])
print(f"Correlation installments vs order value: {correlation:.2f}")

# Installment distribution
print(f"\nInstallment distribution:\n{payment['payment_installments'].value_counts().sort_index().head(15)}")

# =============================================================================
# 11. INSTALLMENT VISUALIZATIONS
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Installment count distribution
installment_dist = payment['payment_installments'].value_counts().sort_index().head(15).reset_index()
installment_dist.columns = ['installments', 'count']
sns.barplot(data=installment_dist, x='installments', y='count', palette='viridis', ax=axes[0])
axes[0].set_title('Distribution of Payment Installments')
axes[0].set_xlabel('Number of Installments')
axes[0].set_ylabel('Count')

# Avg order value by installment count
avg_value_by_installment = payment.groupby('payment_installments')['payment_value'].mean().reset_index().head(15)
sns.lineplot(data=avg_value_by_installment, x='payment_installments', y='payment_value', marker='o', ax=axes[1])
axes[1].set_title('Avg Order Value by Installment Count')
axes[1].set_xlabel('Number of Installments')
axes[1].set_ylabel('Avg Order Value (R$)')

plt.tight_layout()
plt.savefig('installment_analysis.png')
plt.show()

# =============================================================================
# 12. ORDER VALUE DISTRIBUTION
# =============================================================================

plt.figure(figsize=(12, 5))
sns.histplot(payment[payment['payment_value'] <= 1000]['payment_value'], bins=50, kde=True, color='purple')
plt.title('Order Value Distribution (capped at R$1000)')
plt.xlabel('Payment Value (R$)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('order_value_distribution.png')
plt.show()

# =============================================================================
# 13. EXPORT DATA FOR POWER BI
# =============================================================================

output_path = os.path.join(project_root, 'Data', 'processed')
os.makedirs(output_path, exist_ok=True)

# Export raw cleaned payment (row level)
payment.to_csv(os.path.join(output_path, 'payments_cleaned.csv'), index=False)

# Export order-level aggregated payment (for RFM monetary value)
order_level.to_csv(os.path.join(output_path, 'payments_order_level.csv'), index=False)

print("\n✅ Exported payments_cleaned.csv to Data/processed/")
print("✅ Exported payments_order_level.csv to Data/processed/")