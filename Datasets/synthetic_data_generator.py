import pandas as pd
import random
from faker import Faker
from datetime import datetime, timedelta

fake = Faker()

# Load original dataset
df = pd.read_csv("synthetic_supply_chain_data.csv")

# ========== E-COMMERCE ENRICHMENT ==========

product_categories = ['Electronics', 'Clothing',
                      'Home Appliances', 'Books', 'Toys', 'Groceries', 'Furniture']
payment_methods = ['Credit Card', 'Debit Card',
                   'UPI', 'Net Banking', 'Cash on Delivery']
customer_types = ['New', 'Returning', 'Guest']
delivery_modes = ['Standard', 'Express', 'Same Day']
promo_used = ['Yes', 'No']
order_statuses = ['Processing', 'Shipped',
                  'Delivered', 'Cancelled', 'Returned']
return_reasons = ['Damaged', 'Expired',
                  'Incorrect Delivery', 'Customer Rejection', 'None']

# Add synthetic e-commerce columns
df['order_id'] = [fake.uuid4() for _ in range(len(df))]
df['product_category'] = [random.choice(
    product_categories) for _ in range(len(df))]
df['order_value'] = [round(random.uniform(200, 5000), 2)
                     for _ in range(len(df))]
df['discount_applied'] = [round(random.uniform(
    0, 0.5) * value, 2) for value in df['order_value']]
df['final_price'] = df['order_value'] - df['discount_applied']
df['payment_method'] = [random.choice(payment_methods) for _ in range(len(df))]
df['promo_code_used'] = [random.choice(promo_used) for _ in range(len(df))]
df['delivery_mode'] = [random.choice(delivery_modes) for _ in range(len(df))]
df['order_status'] = [random.choice(order_statuses) for _ in range(len(df))]
df['customer_type'] = [random.choice(customer_types) for _ in range(len(df))]
df['customer_rating'] = [
    round(random.uniform(1.0, 5.0), 1) if status == 'Delivered' else None
    for status in df['order_status']
]

# Ensure datetime fields are in datetime format
df['departuretime'] = pd.to_datetime(df['departuretime'], errors='coerce')
df['arrivaltime'] = pd.to_datetime(df['arrivaltime'], errors='coerce')

# Clean return reason
df['if_return_reason'] = df.apply(
    lambda row: random.choice(
        return_reasons[:-1]) if row['return_status'] == 'Yes' else 'None',
    axis=1
)

# ========== BOTTLENECK SIMULATION ==========

# Select 15% of rows to simulate bottlenecks
bottleneck_rows = df.sample(frac=0.15).index

for i in bottleneck_rows:
    df.loc[i, 'resource_usage'] = round(random.uniform(95, 100), 2)
    df.loc[i, 'utility_allocation'] = round(random.uniform(1, 10), 2)
    df.loc[i, 'status'] = 'Delayed'
    df.loc[i, 'priority'] = 5
    if pd.notnull(df.loc[i, 'departuretime']):
        df.loc[i, 'arrivaltime'] = df.loc[i, 'departuretime'] + \
            timedelta(hours=random.randint(48, 96))
        df.loc[i, 'deadline'] = (
            df.loc[i, 'departuretime'] + timedelta(days=1)).strftime('%Y-%m-%d')
    df.loc[i, 'department'] = random.choice(['Logistics', 'Inventory'])

# Add a flag column for ML training
df['bottleneck_flag'] = df.index.isin(bottleneck_rows).astype(int)

# ========== EXPORT ==========
df.to_csv("ecommerce_supply_chain_data.csv", index=False)
print("✅ Enriched and bottleneck-enabled dataset saved as 'ecommerce_supply_chain_data.csv'")
