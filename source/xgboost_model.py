import pandas as pd
import numpy as np
import re
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error

df = pd.read_csv('data/processed_data_from_atl.csv')
df = df.drop(columns=['startingAirport', 'destinationAirport', 'fareBasisCode'])
df = df.dropna(subset=['totalTravelDistance'])


def duration_to_minutes(duration):
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?", duration)
    return int(match.group(1) or 0) * 60 + int(match.group(2) or 0) if match else 0


df['travelDurationMinutes'] = df['travelDuration'].apply(duration_to_minutes)
df['searchDate'] = pd.to_datetime(df['searchDate'])
df['flightDate'] = pd.to_datetime(df['flightDate'])
df['daysBeforeFlight'] = (df['flightDate'] - df['searchDate']).dt.days
df = df.drop(columns=['searchDate', 'flightDate', 'travelDuration'])

label_enc = LabelEncoder()
df['segmentsAirlineName'] = label_enc.fit_transform(df['segmentsAirlineName'])

scaler = StandardScaler()
df[['totalTravelDistance', 'travelDurationMinutes', 'daysBeforeFlight']] = scaler.fit_transform(
    df[['totalTravelDistance', 'travelDurationMinutes', 'daysBeforeFlight']]
)

X = df.drop(columns=['totalFare'])
y = df['totalFare']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [500, 1000, 2000],
    'learning_rate': [0.01, 0.05, 0.2, 0.5],
    'max_depth': [3, 5, 7],
    'min_child_weight': [3, 5, 10],
    'subsample': [0.5, 0.8, 0.99],
    'colsample_bytree': [0.7, 0.8, 0.99]
}

xgb = XGBRegressor(objective='reg:squarederror', random_state=42)

random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid,
    n_iter=20,
    cv=5,
    scoring='neg_mean_squared_error',
    random_state=42,
    verbose=2
)

random_search.fit(X_train, y_train)

best_xgb = XGBRegressor(**random_search.best_params_)
best_xgb.fit(X_train, y_train)
y_pred = best_xgb.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"XGBoost RMSE: {rmse}")

results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results.head(50))

results.to_csv('xgboost_results.csv', index=False)
