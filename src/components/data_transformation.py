import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Create and return a preprocessor object for data transformation.
        
        Returns:
            ColumnTransformer: Preprocessor pipeline for scaling and encoding
        """
        try:
            logging.info("Data Transformation: Starting to create preprocessor object")
            
            # Note: These will be determined dynamically in initiate_data_transformation
            numerical_cols = []
            categorical_cols = []
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_cols),
                    ('cat', OneHotEncoder(
                        handle_unknown="ignore",
                        min_frequency=10,
                        sparse_output=False
                    ), categorical_cols)
                ]
            )
            
            logging.info("Preprocessor object created successfully")
            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Perform data transformation on train and test datasets.
        
        Args:
            train_path (str): Path to training dataset
            test_path (str): Path to testing dataset
            
        Returns:
            tuple: (transformed_train_array, transformed_test_array, preprocessor_path)
        """
        try:
            logging.info("Entered the data transformation method")
            
            # Read the datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test data completed")
            
            # Create target variable (log transformation of salary)
            train_df["log_salary"] = np.log1p(train_df["salary_usd"])
            test_df["log_salary"] = np.log1p(test_df["salary_usd"])
            
            logging.info("Log salary created")
            
            # Separate features and target
            input_feature_train_df = train_df.drop(columns=["salary_usd", "log_salary", "job_id"], axis=1)
            target_feature_train_df = train_df["log_salary"]
            
            input_feature_test_df = test_df.drop(columns=["salary_usd", "log_salary", "job_id"], axis=1)
            target_feature_test_df = test_df["log_salary"]
            
            logging.info("Separated features and target")
            
            # Identify numerical and categorical columns
            numerical_cols = input_feature_train_df.select_dtypes(
                include=['int64', 'float64']
            ).columns.to_list()
            categorical_cols = input_feature_train_df.select_dtypes(
                include=['object']
            ).columns.to_list()
            
            logging.info(f"Numerical columns: {numerical_cols}")
            logging.info(f"Categorical columns: {categorical_cols}")
            
            # Create preprocessor with identified columns
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_cols),
                    ('cat', OneHotEncoder(
                        handle_unknown="ignore",
                        min_frequency=10,
                        sparse_output=False
                    ), categorical_cols)
                ]
            )
            
            # Fit and transform training data
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            
            logging.info("Data transformation completed")
            
            # Combine features with target
            train_arr = np.c_[
                input_feature_train_arr, 
                np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, 
                np.array(target_feature_test_df)
            ]
            
            # Save the preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )
            
            logging.info("Preprocessor saved successfully")
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)
