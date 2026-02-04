import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        self.preprocessor = None
        self.model = None

    def load_models(self, preprocessor_path, model_path):
        """
        Load the preprocessor and trained model.
        
        Args:
            preprocessor_path (str): Path to the preprocessor pickle file
            model_path (str): Path to the trained model pickle file
        """
        try:
            logging.info("Loading preprocessor and model")
            self.preprocessor = load_object(file_path=preprocessor_path)
            self.model = load_object(file_path=model_path)
            logging.info("Models loaded successfully")
            
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features_df):
        """
        Make predictions on new data.
        
        Args:
            features_df (pd.DataFrame): Input features for prediction
            
        Returns:
            np.array: Predicted values (in log scale)
        """
        try:
            if self.preprocessor is None or self.model is None:
                raise CustomException("Models not loaded. Call load_models() first.", sys)
            
            logging.info("Starting prediction")
            
            # Transform features using the preprocessor
            data_scaled = self.preprocessor.transform(features_df)
            
            # Make predictions
            preds = self.model.predict(data_scaled)
            
            logging.info(f"Predictions made for {len(preds)} samples")
            
            return preds
            
        except Exception as e:
            raise CustomException(e, sys)

    def predict_salary(self, features_df):
        """
        Make predictions and convert from log scale back to actual salary.
        
        Args:
            features_df (pd.DataFrame): Input features for prediction
            
        Returns:
            np.array: Predicted salary values (in original scale)
        """
        try:
            log_predictions = self.predict(features_df)
            # Inverse log transformation: exp(log_salary) - 1
            salary_predictions = np.expm1(log_predictions)
            
            logging.info("Salary predictions converted from log scale")
            
            return salary_predictions
            
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    This class maps web/UI input data to dataframe format.
    Ensures data types match the training data structure.
    """
    
    def __init__(self,
                 posting_year: int,
                 country: str,
                 region: str,
                 city: str,
                 company_name: str,
                 company_size: str,
                 industry: str,
                 job_title: str,
                 seniority_level: str,
                 ai_mentioned: bool,
                 ai_keywords: str,
                 ai_intensity_score: float,
                 core_skills: str,
                 ai_skills: str,
                 salary_usd: int,
                 salary_change_vs_prev_year_percent: float,
                 automation_risk_score: float,
                 reskilling_required: bool,
                 ai_job_displacement_risk: str,
                 job_description_embedding_cluster: int,
                 industry_ai_adoption_stage: str):
        
        self.posting_year = posting_year
        self.country = country
        self.region = region
        self.city = city
        self.company_name = company_name
        self.company_size = company_size
        self.industry = industry
        self.job_title = job_title
        self.seniority_level = seniority_level
        self.ai_mentioned = ai_mentioned
        self.ai_keywords = ai_keywords
        self.ai_intensity_score = ai_intensity_score
        self.core_skills = core_skills
        self.ai_skills = ai_skills
        self.salary_usd = salary_usd
        self.salary_change_vs_prev_year_percent = salary_change_vs_prev_year_percent
        self.automation_risk_score = automation_risk_score
        self.reskilling_required = reskilling_required
        self.ai_job_displacement_risk = ai_job_displacement_risk
        self.job_description_embedding_cluster = job_description_embedding_cluster
        self.industry_ai_adoption_stage = industry_ai_adoption_stage

    def get_data_as_dataframe(self):
        """
        Convert the custom data to a pandas DataFrame with correct dtypes matching training data.
        Matches the exact structure used during model training.
        
        Returns:
            pd.DataFrame: DataFrame with a single row of data with proper dtypes
        """
        try:
            # Create dictionary with correct column order and values
            custom_data_input_dict = {
                'job_id': [0],  # Dummy value, will be excluded during preprocessing
                'posting_year': [self.posting_year],
                'country': [self.country],
                'region': [self.region],
                'city': [self.city],
                'company_name': [self.company_name],
                'company_size': [self.company_size],
                'industry': [self.industry],
                'job_title': [self.job_title],
                'seniority_level': [self.seniority_level],
                'ai_mentioned': [self.ai_mentioned],
                'ai_keywords': [self.ai_keywords],
                'ai_intensity_score': [self.ai_intensity_score],
                'core_skills': [self.core_skills],
                'ai_skills': [self.ai_skills],
                'salary_usd': [self.salary_usd],
                'salary_change_vs_prev_year_percent': [self.salary_change_vs_prev_year_percent],
                'automation_risk_score': [self.automation_risk_score],
                'reskilling_required': [self.reskilling_required],
                'ai_job_displacement_risk': [self.ai_job_displacement_risk],
                'job_description_embedding_cluster': [self.job_description_embedding_cluster],
                'industry_ai_adoption_stage': [self.industry_ai_adoption_stage]
            }
            
            # Create DataFrame
            df = pd.DataFrame(custom_data_input_dict)
            
            # Convert dtypes to match training data
            dtype_mapping = {
                'job_id': 'object',
                'posting_year': 'int64',
                'country': 'object',
                'region': 'object',
                'city': 'object',
                'company_name': 'object',
                'company_size': 'object',
                'industry': 'object',
                'job_title': 'object',
                'seniority_level': 'object',
                'ai_mentioned': 'bool',
                'ai_keywords': 'object',
                'ai_intensity_score': 'float64',
                'core_skills': 'object',
                'ai_skills': 'object',
                'salary_usd': 'int64',
                'salary_change_vs_prev_year_percent': 'float64',
                'automation_risk_score': 'float64',
                'reskilling_required': 'bool',
                'ai_job_displacement_risk': 'object',
                'job_description_embedding_cluster': 'int64',
                'industry_ai_adoption_stage': 'object'
            }
            
            for col, dtype in dtype_mapping.items():
                if col in df.columns:
                    df[col] = df[col].astype(dtype)
            
            return df
            
        except Exception as e:
            raise CustomException(e, sys)