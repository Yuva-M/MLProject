from dataclasses import dataclass
import os
import sys

from src.logger import logging
from src.exception import CustomException

from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_trainer(self, train_array,test_array):
        try:
            logging.info("splitting the data into train and test input data ")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1], # take out the last column and feed it into x_train
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models ={
                    "Random Forest": RandomForestRegressor(),
                    "Decision Tree": DecisionTreeRegressor(),
                    "Gradient Boosting": GradientBoostingRegressor(),
                    "Linear Regression": LinearRegression(),
                    "K-Neighbors Regressor": KNeighborsRegressor(),  
                    "CatBoosting Regressor": CatBoostRegressor(verbose=False), 
                    "AdaBoost Regressor": AdaBoostRegressor(),  
                    "XGBRegressor": XGBRegressor(),  
                }

            params = {
                    "Decision Tree": {
                        'criterion': ['squared_error','friedman_mse','absolute_error','poisson'],
                    },
                    "Random Forest":{
                        'n_estimators': [8,16,32,64,128,256]
                    },
                    "Gradient Boosting":{
                        'learning_rate':[.1,.01,.05,.001],
                        'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                        'n_estimators': [8,16,32,64,128,256]
                    },
                    "Linear Regression":{},
                    "K-Neighbors Regressor":{  # Corrected key
                        'n_neighbors':[5,7,9,11],
                    },
                    "XGBRegressor":{  # Corrected key
                        'learning_rate':[.1,.01,.05,.001],
                        'n_estimators': [8,16,32,64,128,256]
                    },
                    "CatBoosting Regressor":{  # Corrected key
                        'depth': [6,8,10],
                        'learning_rate': [0.01, 0.05, 0.1],
                        'iterations': [30, 50, 100]
                    },
                    "AdaBoost Regressor":{  # Corrected key
                        'learning_rate':[.1,.01,0.5,.001],
                        'n_estimators': [8,16,32,64,128,256]
                    }
                }
            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
                ]
            
            best_model = models[best_model_name]

            if best_model_score < .6:
                raise CustomException("No model found")
            logging.info("Best model found on both training and testing dataset")

            save_object(
                file_path =self.model_trainer_config.trained_model_file,
                obj = best_model
            ) 

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test,predicted)
            return r2_square
        
        except Exception as e:
            raise CustomException(e,sys)