import pandas as pd
import numpy as np
from datetime import datetime

np.random.seed(42)

# cars to include in dataset
car_data = {
    'Toyota': ['Camry', 'Corolla', 'RAV4', 'Highlander', 'Tacoma'],
    'Honda': ['Civic', 'Accord', 'CR-V', 'Pilot'],
    'Ford': ['F-150', 'Mustang', 'Explorer', 'Escape'],
    'Chevrolet': ['Silverado', 'Malibu', 'Equinox', 'Tahoe'],
    'BMW': ['3 Series', '5 Series', 'X3', 'X5'],
    'Mercedes-Benz': ['C-Class', 'E-Class', 'GLC', 'GLE'],
    'Tesla': ['Model 3', 'Model Y', 'Model S'],
    'Nissan': ['Altima', 'Rogue', 'Sentra'],
    'Hyundai': ['Elantra', 'Tucson', 'Santa Fe'],
    'Mazda': ['Mazda3', 'CX-5', 'CX-9']
}

# generate 3000 records
data = []
current_year = datetime.now().year

for _ in range(3000):
    make = np.random.choice(list(car_data.keys()))
    model = np.random.choice(car_data[make])
    year = np.random.randint(2010, current_year + 1)
    
    # mileage based on age
    car_age = current_year - year
    avg_miles_per_year = np.random.normal(12000, 3000)
    mileage = max(0, int(car_age * avg_miles_per_year + np.random.normal(0, 5000)))
    
    # fuel type
    if make == 'Tesla':
        fuel_type = 'Electric'
    else:
        fuel_type = np.random.choice(['Gasoline', 'Diesel', 'Hybrid'], 
                                     p=[0.7, 0.15, 0.15])
    
    transmission = np.random.choice(['Automatic', 'Manual'], p=[0.85, 0.15])
    
    # base prices (rough estimates)
    base_prices = {
        'Toyota': 28000, 'Honda': 27000, 'Ford': 30000,
        'Chevrolet': 29000, 'BMW': 45000, 'Mercedes-Benz': 50000,
        'Tesla': 48000, 'Nissan': 26000, 'Hyundai': 24000,
        'Mazda': 26000
    }
    
    base_price = base_prices[make]
    
    # apply depreciation
    depreciated_price = base_price
    if car_age >= 1:
        depreciated_price *= 0.80  # first year hit
        if car_age > 1:
            depreciated_price *= (0.85 ** (car_age - 1))
    
    # mileage adjustments
    if mileage > 100000:
        depreciated_price *= 0.85
    elif mileage > 75000:
        depreciated_price *= 0.92
    
    # add randomness
    price = depreciated_price * np.random.uniform(0.85, 1.15)
    
    # luxury brands hold value better
    if make in ['BMW', 'Mercedes-Benz', 'Tesla']:
        price *= 1.1
    
    data.append({
        'make': make,
        'model': model,
        'year': year,
        'mileage': mileage,
        'fuel_type': fuel_type,
        'transmission': transmission,
        'price': round(price, 2)
    })

df = pd.DataFrame(data)

print("Generated sample data")
print(f"Total: {len(df)} cars")
print("\nFirst few:")
print(df.head(10))

print("\nStats:")
print(df.describe())

print("\nSaving to car_data.csv...")
df.to_csv('car_data.csv', index=False)
print("Done! Run train_model.py next")
