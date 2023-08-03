# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 10:28:04 2023

@author: rezas
"""

from EDA import EDA_process
from Visualization import plot_all_visualizations
from preprocessing import preprocess
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

def main():
    file_path = './train.csv'
    df = EDA_process(file_path)
    plot_all_visualizations(df)
    X_train, X_test, y_train, y_test = preprocess(df)
    
    
    # # MLP ---------------------------------------------------------------------------- 0.04!
    # # Define a simple MLP model
    # mlp = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=500, random_state=42)
    
    # # Train the model
    # mlp.fit(X_train, y_train)
    
    # # Make predictions
    # y_pred = mlp.predict(X_test)
    
    
    # # XGBoost ---------------------------------------------------------------------------- 0.6129!
    # #training phase xgboost
    # xgbr = XGBRegressor()
    # # Define the grid of hyperparameters to search
    # parameter_grid = {
    #     'n_estimators': [65],
    #     'learning_rate': [0.08],
    #     'max_depth': [3],
    #     'colsample_bytree': [0.7],
    #     'gamma': [0]
    # }

    # # Set up the grid search with 4-fold cross validation
    # grid_cv = GridSearchCV(estimator=xgbr, param_grid=parameter_grid, cv=5, scoring='r2', verbose=2, n_jobs=-1)

    # grid_cv.fit(X_train, y_train)
    
    # print("Best parameters found: ", grid_cv.best_params_)
    # print("Highest R^2 found: ", grid_cv.best_score_)
    
    # y_pred = grid_cv.predict(X_test)
    
    # # RandomForest ---------------------------------------------------------------------------- 0.6117!
    rfr = RandomForestRegressor(random_state=42)

    # Define the grid of hyperparameters to search
    parameter_grid = {
        'n_estimators': [400],
        'max_depth': [5],
        'min_samples_split': [2],
        'min_samples_leaf': [2],
        'max_features': ['auto']
    }

    # Set up the grid search with 4-fold cross validation
    grid_cv = GridSearchCV(estimator=rfr, param_grid=parameter_grid, cv=3, scoring='r2', verbose=2, n_jobs=-1)

    grid_cv.fit(X_train, y_train)
    
    print("Best parameters found: ", grid_cv.best_params_)
    print("Highest R^2 found: ", grid_cv.best_score_)
    
    y_pred = grid_cv.predict(X_test)
    
    
    
    
    
    print(r2_score(y_test, y_pred))
    
    
    
if __name__ == "__main__":
    main()

