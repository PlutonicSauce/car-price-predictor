# Car Price Predictor

Web app that predicts used car prices using machine learning. Built this to learn more about ML deployment and feature engineering.

## What it does

Predicts car prices based on:
- Make and model
- Year and mileage  
- Fuel type and transmission
- Depreciation patterns

The model accounts for how cars lose value over time, which was the most interesting part to figure out.

## Built with

- Python
- scikit-learn for the ML models
- Streamlit for the web interface
- pandas/numpy for data stuff

## How to run it

### Setup
```bash
git clone <your-repo-url>
cd car-price-predictor

# create virtual environment
python -m venv venv
source venv/bin/activate  # on windows: venv\Scripts\activate

# install stuff
pip install -r requirements.txt
```

### Generate data and train model
```bash
python generate_data.py
python train_model.py
```

### Run the app
```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

## Using real data

I used generated data but you can swap in real datasets from Kaggle:
- [Vehicle Dataset 2024](https://www.kaggle.com/datasets/kanchana1990/vehicle-dataset-2024)
- [Used Car Prices](https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset)

Just rename the CSV to `car_data.csv` and make sure it has these columns: `make`, `model`, `year`, `mileage`, `fuel_type`, `transmission`, `price`

## Files

```
car-price-predictor/
├── app.py                  # streamlit app
├── train_model.py          # trains the model
├── generate_data.py        # creates sample data
├── requirements.txt        
└── README.md              
```

## How it works

The model uses these features:
- Car age (current year - manufacture year)
- Mileage and mileage per year
- Make, model, fuel type, transmission

For depreciation, I applied different rates:
- First year: ~20% value loss
- Each year after: ~10-15%
- High mileage (>100k): additional reduction

The app also adjusts predictions based on how old the training data is, since cars depreciate constantly.

## Deployment

I deployed this on Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo
4. Hit deploy

You can also use Render or Heroku if you want.

## What I learned

- Feature engineering is huge and car_age helped capture non-linear depreciation
- Random Forest worked better than linear models for this
- Deploying ML models is different from training them. You need to handle edge cases
- Depreciation is pretty tricky because it changes based on brand, mileage, and time

## Future ideas

Things I might add:
- Car condition as a feature
- Location based pricing (cars cost different amounts by state)
- More brands and models
- Comparison with actual KBB values


