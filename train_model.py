import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import os

# ========================================
# STEP 1: Generate Realistic Indian Housing Data
# ========================================
print("Generating realistic housing dataset...")

np.random.seed(42)

n_samples = 12000

cities = ['Mumbai', 'Delhi', 'Bangalore', 'Pune', 'Hyderabad', 'Chennai', 'Nagpur']
city_growth_multiplier = {
    'Mumbai': 1.8, 'Delhi': 1.6, 'Bangalore': 1.7,
    'Pune': 1.2, 'Hyderabad': 1.3, 'Chennai': 1.1, 'Nagpur': 0.9
}

data = []

for city in cities:
    n = n_samples // len(cities) + (1000 if city in ['Mumbai', 'Delhi', 'Bangalore'] else 0)
    
    for _ in range(n):
        area = np.random.randint(500, 3500)  # sq ft
        
        # Realistic BHK logic
        if area < 800:
            bhk = np.random.choice([1, 2], p=[0.7, 0.3])
        elif area < 1500:
            bhk = np.random.choice([2, 3], p=[0.6, 0.4])
        else:
            bhk = np.random.choice([3, 4, 5], p=[0.5, 0.3, 0.2])
        
        bathrooms = bhk if np.random.rand() < 0.7 else bhk + 1
        balcony = np.random.randint(0, 4)
        
        # Base price per sq ft by city + area tier
        base_psf = {
            'Mumbai': 22000, 'Delhi': 18000, 'Bangalore': 12000,
            'Pune': 8500, 'Hyderabad': 7800, 'Chennai': 7200, 'Nagpur': 4800
        }[city]
        
        # Premium for larger homes
        area_factor = 1.0
        if area > 2000: area_factor = 1.4
        elif area > 1500: area_factor = 1.15
        
        # Random noise
        noise = np.random.normal(1, 0.25)
        
        price_per_sqft = base_psf * city_growth_multiplier[city] * area_factor * noise
        price = int(area * price_per_sqft)
        
        # Round to nearest 5000
        price = round(price / 5000) * 5000
        
        data.append({
            'area': area,
            'bedrooms': bhk,
            'bathrooms': bathrooms,
            'balcony': balcony,
            'location': city,
            'price': price
        })

df = pd.DataFrame(data)
print(f"Dataset created: {len(df)} rows")

# Save raw data (optional)
os.makedirs("data", exist_ok=True)
df.to_csv("data/housing_large.csv", index=False)
print("Dataset saved to data/housing_large.csv")

# ========================================
# STEP 2: Preprocessing & One-Hot Encoding
# ========================================
print("\nPreprocessing data...")

# One-hot encode location
df_encoded = pd.get_dummies(df, columns=['location'], prefix='loc')

X = df_encoded.drop('price', axis=1)
y = df_encoded['price']

# ========================================
# STEP 3: Train-Test Split & Scaling
# ========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========================================
# STEP 4: Train Random Forest Model
# ========================================
print("\nTraining RandomForestRegressor...")

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

# ========================================
# STEP 5: Evaluate Model
# ========================================
y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"   Mean Absolute Error : ₹{mae:,.0f}")
print(f"   R² Score            : {r2:.4f} ({r2*100:.2f}%)")

# Example prediction
sample = X_test_scaled[0:1]
pred_price = int(model.predict(sample)[0])
actual_price = int(y_test.iloc[0])
print(f"\nSample Prediction:")
print(f"   Predicted : ₹{pred_price:,}")
print(f"   Actual    : ₹{actual_price:,}")

# ========================================
# STEP 6: Save Model & Scaler
# ========================================
os.makedirs("model", exist_ok=True)

pickle.dump(model, open("model/regression_model.pkl", "wb"))
pickle.dump(scaler, open("model/scaler.pkl", "wb"))

# Save column names for inference
with open("model/columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("\nModel, scaler, and columns saved successfully!")
print("Your Flask app is now ready to make predictions!")