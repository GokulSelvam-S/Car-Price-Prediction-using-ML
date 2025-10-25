import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

# Load Dataset
df = pd.read_csv("E:\Python Programs\cardekho.csv")
print("Dataset Shape:", df.shape)
print(df.head())
print(df.tail())
print(df.info())
print(df.describe())
print(df.shape)
print(df.size)

# Clean Dataset
df.dropna(inplace=True)

df['year'] = pd.to_numeric(df['year'], errors='coerce')
df['km_driven'] = pd.to_numeric(df['km_driven'], errors='coerce')
df.dropna(inplace=True)

# Visualizations
plt.figure(figsize=(8,5))
sns.histplot(df['selling_price'], bins=50, kde=True)
plt.title("Car Selling Price Distribution")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x='fuel', y='selling_price', data=df)
plt.title("Fuel Type vs Selling Price")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x='transmission', y='selling_price', data=df)
plt.title("Transmission vs Selling Price")
plt.show()

plt.figure(figsize=(10,6))
sns.scatterplot(x='year', y='selling_price', data=df, alpha=0.6)
plt.title("Car Price vs Year of Manufacture")
plt.show()

plt.figure(figsize=(10,6))
sns.scatterplot(x='km_driven', y='selling_price', data=df, alpha=0.6)
plt.title("Car Price vs Kilometers Driven")
plt.show()

# Using Model
df['car_model'] = df['name'].str.split().str[1]

car_fuel = LabelEncoder()
df['fuel'] = car_fuel.fit_transform(df['fuel'])

car_seller = LabelEncoder()
df['seller_type'] = car_seller.fit_transform(df['seller_type'])

car_trans = LabelEncoder()
df['transmission'] = car_trans.fit_transform(df['transmission'])

car_owner = LabelEncoder()
df['owner'] = car_owner.fit_transform(df['owner'])

car_model = LabelEncoder()
df['car_model'] = car_model.fit_transform(df['car_model'].astype(str))

# Step 5: Prepare Data
X = df[['year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'car_model']]
y = df['selling_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train Models
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)
lr_score = r2_score(y_test, y_pred)

rf_reg = RandomForestRegressor(random_state=42, n_estimators=200)
rf_reg.fit(X_train, y_train)
y_pred = rf_reg.predict(X_test)
rf_score = r2_score(y_test, y_pred)

print("Linear Regression R² Score:", round(lr_score, 3))
print("Random Forest R² Score:", round(rf_score, 3))

# Prediction Function 
def predict_price(year, kms, fuel, seller, transmission, owner, car_model, model=rf_reg):
    prediction = pd.DataFrame({
        'year': [year],
        'km_driven': [kms],
        'fuel': [car_fuel.transform([fuel])[0]],
        'seller_type': [car_seller.transform([seller])[0]],
        'transmission': [car_trans.transform([transmission])[0]],
        'owner': [car_owner.transform([owner])[0]],
        'car_model': [car_model.transform([car_model])[0]]
    })
    price = model.predict(prediction)
    return int(price[0])

print("Predicted Price:", predict_price(2018, 25000, 'Petrol', 'Individual', 'Manual', 'First Owner', 'Swift'))
print("Predicted Price:", predict_price(2015, 40000, 'Diesel', 'Dealer', 'Manual', 'Second Owner', 'i20'))