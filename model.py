import pandas as pd 
import numpy as np 
import xgboost as xgb
import lightgbm as lgb
import  pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
data = pd.read_csv('hprice.csv')
X = data.drop(columns=['variable_dependiente'])
y = data['variable_dependiente']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf_grid_search = GridSearchCV(RandomForestRegressor(), param_grid=rf_param_grid, scoring='neg_root_mean_squared_error', cv=5)
rf_grid_search.fit(X_train, y_train)
best_rf_model = rf_grid_search.best_estimator_


models = [best_rf_model] 
best_model = None
best_rmse = float('inf')


for model in models:
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    if rmse < best_rmse:
        best_model = model
        best_rmse = rmse


print("Mejor modelo seleccionado:", best_model)
print("RMSE del mejor modelo en conjunto de testeo:", best_rmse)

with open("model.pickle", "wb") as f:
    pickle.dump(best_model, f)


