import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging



from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    # use as input to the following function, the output of data_transformer
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1], 
                train_array[:, -1], 
                test_array[:, :-1], 
                test_array[:, -1]
            )

            models = {
                'Linear Regression': LinearRegression(),
                'Decision Tree': DecisionTreeRegressor(),
                'Random Forest': RandomForestRegressor(),
                'Gradient Boosting ': GradientBoostingRegressor(),
                'XGBoost Regressor': XGBRegressor(),
                'K-Neighbors Regressor': KNeighborsRegressor(),
                'AdaBoost Classifier': AdaBoostRegressor(),
                'CatBoost Classifier': CatBoostClassifier(verbose=0)
            }

            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test , models=models)
            logging.info(f"Models trained: {model_report}")

            ## To get best model score from dict (using test score for comparison)
            test_scores = {model_name: scores['test_score'] for model_name, scores in model_report.items()}
            best_model_score = max(test_scores.values())
            ## To get best model name from dict
            best_model_name = [model_name for model_name, score in test_scores.items() if score == best_model_score][0]

            best_model = models[best_model_name]
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")
            

            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)


            predicted = best_model.predict(X_test)
            r2_sequare = r2_score(y_test, predicted)
            return r2_sequare

        except Exception as e:
            raise CustomException(e, sys)
