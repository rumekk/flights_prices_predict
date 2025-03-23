import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn
import re
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error

df = pd.read_csv('data/processed_data_from_atl.csv')
df = df.drop(columns=['startingAirport',
             'destinationAirport', 'fareBasisCode'])

# print(df.head())
# print(df.info())
# print(df.describe())
# print(df.isnull().sum())

df = df.dropna(subset=['totalTravelDistance'])
# df.to_csv('processed_data_from_atl_okrojony.csv', index=False)
# print(df.isnull().sum())
# print(df.shape)

# seaborn.histplot(df['totalFare'], bins=50, kde=True)
# plt.pyplot.show()

# seaborn.scatterplot(df, x='totalTravelDistance', y='totalFare')
# plt.pyplot.show()

# seaborn.boxplot(df, x='segmentsAirlineName', y='totalFare')
# plt.pyplot.show()


def duration_to_minutes(duration):
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?", duration)
    return int(match.group(1) or 0) * 60 + int(match.group(2) or 0) if match else 0


df['travelDurationMinutes'] = df['travelDuration'].apply(duration_to_minutes)

df['searchDate'] = pd.to_datetime(df['searchDate'])
df['flightDate'] = pd.to_datetime(df['flightDate'])
df['daysBeforeFlight'] = (df['flightDate'] - df['searchDate']).dt.days
df = df.drop(columns=['searchDate', 'flightDate', 'travelDuration'])
print(df.head())

label_enc = LabelEncoder()
for col in ['segmentsAirlineName']:
    df[col] = label_enc.fit_transform(df[col])

scaler = StandardScaler()
df[['totalTravelDistance', 'travelDurationMinutes', 'daysBeforeFlight']] = scaler.fit_transform(
    df[['totalTravelDistance', 'travelDurationMinutes', 'daysBeforeFlight']]
)

X = df.drop(columns=['totalFare'])
y = df['totalFare']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

random_search = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring='neg_mean_squared_error',
    random_state=42,
    verbose=2
)

random_search.fit(X_train, y_train)

model = random_search.best_estimator_

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")

results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results.head(50))
results.to_csv('random_forest_results.csv', index=False)

# seaborn.scatterplot(x='Actual', y='Predicted', data=results)
# plt.pyplot.plot([0, results['Actual'].max()], [0, results['Actual'].max()], color='red')
# plt.pyplot.show()
