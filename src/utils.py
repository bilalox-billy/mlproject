import os
import sys
import numpy as np
import pandas as pd

from src.exception import CustomException
import dill 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)  
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train,y_train,X_test,y_test, models,param):
    try:
    
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)
            
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train) # train the model

            

            y_train_pred = model.predict(X_train) # predict on train data
            y_test_pred = model.predict(X_test) # predict on test data

            train_model_score = r2_score(y_train, y_train_pred) # evaluate the model on train data
            test_model_score = r2_score(y_test, y_test_pred) # evaluate the model

            report[list(models.keys())[i]] = {
                'train_score': train_model_score,
                'test_score': test_model_score
            }
        
        return report
        
    except Exception as e:
        raise CustomException(e, sys)