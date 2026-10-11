
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

# Find the dataset in the same folder as this Python file
BASE_DIR = Path(__file__).resolve().parent
csv_files = list(BASE_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        "No CSV dataset found. Place your TNEB CSV file "
        "in the same folder as train_model.py."
    )

DATA_FILE = csv_files[0]
print(f"Loading dataset: {DATA_FILE.name}")

df = pd.read_csv(DATA_FILE)

# Features used by both prediction models
features = [
    "Temperature_C",
    "Humidity_Percent",
    "Rainfall_mm",
    "Fan_Usage_Hours",
    "Washing_Machine_Usage_Hours",
    "Refrigerator_Hours",
    "TV_Usage_Hours",
    "AC_Usage_Hours",
    "Previous_Month_Usage_kWh"
]

# Check that the required columns exist
required_columns = features + [
    "Electricity_Usage_kWh",
    "Electricity_Bill",
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing dataset columns: {missing}")

# Prepare inputs and prediction targets
X = df[features]
y_usage = df["Electricity_Usage_kWh"]
y_bill = df["Electricity_Bill"]

# Train the electricity usage model
usage_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
usage_model.fit(X, y_usage)

# Train the electricity bill model
bill_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
bill_model.fit(X, y_bill)

# Save both models beside main.py
joblib.dump(usage_model, BASE_DIR / "usage_model.pkl")
joblib.dump(bill_model, BASE_DIR / "bill_model.pkl")

print("\nTraining completed successfully!")
print("Created: usage_model.pkl")
print("Created: bill_model.pkl")
print(f"Training records: {len(df)}")
