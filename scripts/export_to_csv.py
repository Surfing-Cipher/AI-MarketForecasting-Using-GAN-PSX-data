import pandas as pd
from sqlalchemy import create_engine
import os

# 1. Connect to your local database
db_path = "../data/database/nexus.db"
if not os.path.exists(db_path):
    print("Error: nexus.db not found!")
    exit()

print(f"Found database at: {db_path}")
engine = create_engine(f"sqlite:///{db_path}")

# 2. Query the data
query = "SELECT date, open, high, low, close, volume FROM market_data WHERE ticker='OGDC' ORDER BY date ASC"
df = pd.read_sql(query, engine)

# 3. Save to CSV
csv_path = "../data/raw/ogdc_data.csv"
df.to_csv(csv_path, index=False)

print(f"Success! Exported {len(df)} rows with columns {list(df.columns)} to '{csv_path}'.")
print("Now upload this 'ogdc_data.csv' to your Colab environment.")