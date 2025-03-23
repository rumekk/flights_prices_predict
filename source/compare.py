import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

rf_results = pd.read_csv('random_forest_results.csv')
xgb_results = pd.read_csv('xgboost_results.csv')

rf_rmse = np.sqrt(((rf_results['Actual'] - rf_results['Predicted'])**2).mean())
xgb_rmse = np.sqrt(((xgb_results['Actual'] - xgb_results['Predicted'])**2).mean())

print(f"Random Forest RMSE: {rf_rmse}")
print(f"XGBoost RMSE: {xgb_rmse}")

plt.figure(figsize=(10, 6))

plt.scatter(rf_results['Actual'], rf_results['Predicted'], color='blue', label='Random Forest', alpha=0.5)

plt.scatter(xgb_results['Actual'], xgb_results['Predicted'], color='green', label='XGBoost', alpha=0.5)

plt.plot([min(rf_results['Actual'].min(), xgb_results['Actual'].min()),
          max(rf_results['Actual'].max(), xgb_results['Actual'].max())],
         [min(rf_results['Actual'].min(), xgb_results['Actual'].min()),
          max(rf_results['Actual'].max(), xgb_results['Actual'].max())],
         color='red', linestyle='--')

plt.xlabel('Rzeczywiste wartości (totalFare)')
plt.ylabel('Przewidywane wartości')
plt.title('Porównanie wyników: Random Forest vs XGBoost')
plt.legend()
plt.show()
