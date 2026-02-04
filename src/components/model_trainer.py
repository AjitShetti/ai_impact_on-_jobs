import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """
        Train multiple models and return the best one.
        
        Args:
            train_array (np.array): Transformed training data with target in last column
            test_array (np.array): Transformed testing data with target in last column
            
        Returns:
            float: R2 score of the best model on test data
        """
        try:
            logging.info("Splitting training and test input data")
            
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            
            logging.info(f"Training data shape: {X_train.shape}, Target shape: {y_train.shape}")
            logging.info(f"Testing data shape: {X_test.shape}, Target shape: {y_test.shape}")
            
            models = {
                'Linear Regression': LinearRegression(),
                'Lasso': Lasso(random_state=42),
                'Ridge': Ridge(random_state=42),
                'K-Neighbors Regressor': KNeighborsRegressor(n_neighbors=5),
                'Decision Tree': DecisionTreeRegressor(random_state=42),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'XGBRegressor': XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
                'CatBoost Regressor': CatBoostRegressor(verbose=False, random_state=42),
                'AdaBoost Regressor': AdaBoostRegressor(random_state=42)
            }
            
            logging.info("Models initialized")
            
            # Evaluate all models
            model_report = evaluate_models(X_train, y_train, X_test, y_test, models)
            
            # Get the best model
            best_model_name = max(model_report, key=lambda x: model_report[x]['test_r2'])
            best_model = model_report[best_model_name]['model']
            best_test_r2 = model_report[best_model_name]['test_r2']
            
            logging.info(f"Best Model: {best_model_name} with Test R2 Score: {best_test_r2:.4f}")
            
            # Check if model performance is acceptable
            if best_test_r2 < 0.6:
                logging.warning(f"Best model R2 score ({best_test_r2:.4f}) is below 0.6")
            
            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            logging.info(f"Best model saved at {self.model_trainer_config.trained_model_file_path}")
            
            # Print model report
            logging.info("=" * 60)
            logging.info("MODEL EVALUATION REPORT")
            logging.info("=" * 60)
            
            for model_name, metrics in model_report.items():
                logging.info(f"\n{model_name}:")
                logging.info(f"  Train R2: {metrics['train_r2']:.4f}, Test R2: {metrics['test_r2']:.4f}")
                logging.info(f"  Train RMSE: {metrics['train_rmse']:.4f}, Test RMSE: {metrics['test_rmse']:.4f}")
                logging.info(f"  Train MAE: {metrics['train_mae']:.4f}, Test MAE: {metrics['test_mae']:.4f}")
            
            logging.info("=" * 60)
            
            return best_test_r2
            
        except Exception as e:
            raise CustomException(e, sys)
