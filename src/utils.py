import os
import pickle
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    """
    Save a Python object to a pickle file.
    
    Args:
        file_path (str): Path where the object will be saved
        obj: The object to be saved
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        
        logging.info(f"Object saved successfully at {file_path}")
        
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Load a Python object from a pickle file.
    
    Args:
        file_path (str): Path to the pickle file
        
    Returns:
        object: The loaded object
    """
    try:
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
        
        logging.info(f"Object loaded successfully from {file_path}")
        return obj
        
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Evaluate multiple models and return their performance metrics.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Testing features
        y_test: Testing target
        models (dict): Dictionary of model names and model objects
        
    Returns:
        dict: Dictionary containing model performance metrics
    """
    try:
        report = {}
        
        for name, model in models.items():
            logging.info(f"Training model: {name}")
            model.fit(X_train, y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Evaluate metrics
            train_mse = mean_squared_error(y_train, y_train_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)
            
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            report[name] = {
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_rmse': np.sqrt(train_mse),
                'test_rmse': np.sqrt(test_mse),
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'model': model
            }
            
            logging.info(f"Model {name} - Test R2 Score: {test_r2:.4f}")
        
        return report
        
    except Exception as e:
        raise CustomException(e, sys)
