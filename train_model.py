import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# load the data
print("Loading data...")
df = pd.read_csv('car_data.csv')

print(f"Dataset shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())

print("\nBasic info:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

# clean up any missing data
print("\n--- Cleaning data ---")
df = df.dropna()

# feature engineering - depreciation stuff
current_year = datetime.now().year
df['car_age'] = current_year - df['year']
df['age_squared'] = df['car_age'] ** 2  # cars dont depreciate linearly

# mileage features
df['mileage_per_year'] = df['mileage'] / (df['car_age'] + 1)  # avoid div by 0
df['high_mileage'] = (df['mileage'] > 100000).astype(int)

print("\nNew features created:")
print(df[['year', 'car_age', 'mileage', 'mileage_per_year', 'price']].head())

# encode categorical stuff
label_encoders = {}
categorical_cols = ['make', 'model', 'fuel_type', 'transmission']

for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# setup features for training
feature_cols = ['year', 'mileage', 'car_age', 'age_squared', 
                'mileage_per_year', 'high_mileage']

# add the encoded ones
if 'make' in df.columns:
    feature_cols.append('make_encoded')
if 'model' in df.columns:
    feature_cols.append('model_encoded')
if 'fuel_type' in df.columns:
    feature_cols.append('fuel_type_encoded')
if 'transmission' in df.columns:
    feature_cols.append('transmission_encoded')

X = df[feature_cols]
y = df['price']

print(f"\nFeatures used: {feature_cols}")
print(f"Training samples: {len(X)}")
# print(X.head())  # uncomment if you need to debug

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTrain set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# gonna try a few different models
models = {
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'Ridge': Ridge(alpha=10)
    # tried XGBoost too but didn't have it installed
}

results = {}

print("\n--- Training models ---")
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'model': model
    }
    
    print(f"MAE: ${mae:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"R2 Score: {r2:.4f}")

# pick the best one
best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
best_model = results[best_model_name]['model']

print(f"\n--- Best model: {best_model_name} ---")
print(f"R2 Score: {results[best_model_name]['r2']:.4f}")

# save everything
print("\nSaving model...")
joblib.dump(best_model, 'car_price_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(feature_cols, 'feature_cols.pkl')

# save some info about the model
metadata = {
    'training_year': current_year,
    'best_model': best_model_name,
    'r2_score': results[best_model_name]['r2'],
    'mae': results[best_model_name]['mae']
}
joblib.dump(metadata, 'model_metadata.pkl')

print("\nDone!")
print("Files saved: car_price_model.pkl, label_encoders.pkl, feature_cols.pkl, model_metadata.pkl")

# check feature importance if possible
if best_model_name in ['RandomForest', 'GradientBoosting']:
    importances = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("\nTop features:")
    print(feature_importance_df.head(10))
    
    # quick plot
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['feature'][:10], feature_importance_df['importance'][:10])
    plt.xlabel('Importance')
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("\nSaved feature_importance.png")

# actual vs predicted
plt.figure(figsize=(10, 6))
y_pred = best_model.predict(X_test)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Predictions vs Actual')
plt.tight_layout()
plt.savefig('predictions_plot.png')
print("Saved predictions_plot.png")

print("\nReady to run the app!")
