# =============================================================================
# 02_customers_eda.py
# Exploratory Data Analysis — olist_customers_dataset.csv
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
customer_file_path = os.path.join(project_root, 'Data', 'raw', 'Olist_ecom_data', 'olist_customers_dataset.csv')

# Load the dataset
customer = pd.read_csv(customer_file_path)

# =============================================================================
# 3. INITIAL DATA CHECK
# =============================================================================

print("=== Dataset Info ===")
customer.info()

print("\n=== First 5 Rows ===")
print(customer.head())

print("\n=== Missing Values ===")
print(customer.isnull().sum())

# =============================================================================
# 4. UNIQUE VS REPEAT CUSTOMERS
# =============================================================================

print("\n=== Unique vs Repeat Customers ===")
print(f"Total rows:               {len(customer)}")
print(f"Unique customer_id:       {customer['customer_id'].nunique()}")
print(f"Unique customer_unique_id:{customer['customer_unique_id'].nunique()}")

# Total orders
total_orders = len(customer)

# Unique customers (actual people)
unique_customers = customer['customer_unique_id'].nunique()

# Repeat customers
repeat_customers = total_orders - unique_customers

# Repeat rate
repeat_rate = (repeat_customers / total_orders) * 100

print(f"\nRepeat Customers: {repeat_customers}")
print(f"Repeat Rate:      {repeat_rate:.2f}%")
print(f"One-time Rate:    {100 - repeat_rate:.2f}%")

# =============================================================================
# 5. GEOGRAPHIC DISTRIBUTION — Top 10 States
# =============================================================================

print("\n=== Top 10 States by Customer Count ===")
top_states = customer['customer_state'].value_counts().head(10).reset_index()
top_states.columns = ['state', 'count']
print(top_states)

# Visualize
plt.figure(figsize=(10, 6))
sns.barplot(data=top_states, x='count', y='state', palette='magma')
plt.title('Top 10 States by Customer Count')
plt.xlabel('Number of Customers')
plt.ylabel('State')
plt.tight_layout()
plt.savefig('top_states.png')
plt.show()

# =============================================================================
# 6. GEOGRAPHIC DISTRIBUTION — Top 10 Cities
# =============================================================================

print("\n=== Top 10 Cities by Customer Count ===")
top_cities = customer['customer_city'].value_counts().head(10).reset_index()
top_cities.columns = ['city', 'count']
print(top_cities)

# Visualize
plt.figure(figsize=(10, 6))
sns.barplot(data=top_cities, x='count', y='city', palette='magma')
plt.title('Top 10 Cities by Customer Count')
plt.xlabel('Number of Customers')
plt.ylabel('City')
plt.tight_layout()
plt.savefig('top_cities.png')
plt.show()

# =============================================================================
# 7. CONCENTRATION KPIs
# =============================================================================

print("\n=== Concentration KPIs ===")

# Top 3 states concentration
total_customers = len(customer)
top3_states_count = customer['customer_state'].value_counts().head(3).sum()
top3_concentration = (top3_states_count / total_customers) * 100

print(f"Total Customers:        {total_customers}")
print(f"Top 3 States Customers: {top3_states_count}")
print(f"Top 3 Concentration:    {top3_concentration:.2f}%")

# Average customers per state
avg_per_state = customer['customer_state'].value_counts().mean()
print(f"\nAvg Customers per State: {avg_per_state:.0f}")

# Average customers per city
avg_per_city = customer['customer_city'].value_counts().mean()
print(f"Avg Customers per City:  {avg_per_city:.0f}")

# Underserved states (below average)
state_counts = customer['customer_state'].value_counts()
underserved = state_counts[state_counts < state_counts.mean()]
print(f"\nUnderserved States: {len(underserved)} out of {len(state_counts)}")

# Unique states and cities
print(f"\nUnique States: {customer['customer_state'].nunique()}")
print(f"Unique Cities: {customer['customer_city'].nunique()}")

# SP market share
sp_share = (customer[customer['customer_state'] == 'SP'].shape[0] / len(customer)) * 100
print(f"\nSP Market Share: {sp_share:.2f}%")

# =============================================================================
# 8. STATE vs CITY RELATIONSHIP
# =============================================================================

print("\n=== Cities per State ===")
cities_per_state = customer.groupby('customer_state')['customer_city'].nunique().sort_values(ascending=False)
print(cities_per_state)

print("\n=== Top City per State ===")
top_city_per_state = customer.groupby('customer_state').apply(
    lambda x: x['customer_city'].value_counts().index[0]
).reset_index()
top_city_per_state.columns = ['state', 'top_city']
print(top_city_per_state)

# =============================================================================
# 9. EXPORT DATA FOR POWER BI
# =============================================================================

output_path = os.path.join(project_root, 'Data', 'processed')
os.makedirs(output_path, exist_ok=True)

customer.to_csv(os.path.join(output_path, 'customers_cleaned.csv'), index=False)

print("\n✅ Exported customers_cleaned.csv to Data/processed/")