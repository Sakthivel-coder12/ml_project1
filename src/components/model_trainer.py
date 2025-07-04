import pandas as pd
import numpy as np
from dataclasses import dataclass
import os 
import sys
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from src.exception import CustomException
from src.utils import save_object,evaluate_model
from src.logger import logging
from catboost import CatBoost
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor


@dataclass
class modeltrainerconfig:
    trained_model_file_path = os.path.join('artifacts',"model.pkl")
class modeltrainer:
    def __init__(self):
        self.model_trainer_config = modeltrainerconfig()
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info(f"Spliting the train and test input")
            x_train,x_test,y_train,y_test = (
                train_array[:,:-1],
                test_array[:,:-1],
                train_array[:,-1],
                test_array[:,-1]
            )
            models = {
                'random_forest': RandomForestRegressor(),
                'DecisionTreeRegressor':DecisionTreeRegressor(),
                'GradientBoostingRegressor': GradientBoostingRegressor(),
                'LinearRegression' : LinearRegression(),
                'knn': KNeighborsRegressor(),
                'XGBRegressor' : XGBRegressor(),
                'AdaBoostRegressor' : AdaBoostRegressor(),
            }
            model_report:dict  = evaluate_model(x_train = x_train,y_train = y_train,x_test=x_test,y_test=y_test,
                                                models=models)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]          
            best_model = models[best_model_name]
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )  
            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test,predicted)
            return r2_square
        except Exception as e:
            raise CustomException(e,sys)