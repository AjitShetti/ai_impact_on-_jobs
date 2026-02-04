import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging


def run_train_pipeline():
    """
    Execute the complete training pipeline:
    1. Data Ingestion
    2. Data Transformation
    3. Model Training
    """
    try:
        logging.info("Starting the training pipeline")
        
        # Data Ingestion
        logging.info("=" * 60)
        logging.info("STEP 1: DATA INGESTION")
        logging.info("=" * 60)
        
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        
        logging.info(f"Data ingestion completed. Train path: {train_data_path}, Test path: {test_data_path}")
        
        # Data Transformation
        logging.info("\n" + "=" * 60)
        logging.info("STEP 2: DATA TRANSFORMATION")
        logging.info("=" * 60)
        
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
            train_data_path, 
            test_data_path
        )
        
        logging.info(f"Data transformation completed. Preprocessor saved at: {preprocessor_path}")
        
        # Model Training
        logging.info("\n" + "=" * 60)
        logging.info("STEP 3: MODEL TRAINING")
        logging.info("=" * 60)
        
        model_trainer = ModelTrainer()
        best_r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
        
        logging.info(f"Model training completed. Best R2 Score: {best_r2_score:.4f}")
        
        logging.info("\n" + "=" * 60)
        logging.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logging.info("=" * 60)
        
        return {
            'status': 'success',
            'train_data_path': train_data_path,
            'test_data_path': test_data_path,
            'preprocessor_path': preprocessor_path,
            'best_r2_score': best_r2_score
        }
        
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    result = run_train_pipeline()
    print(f"\nPipeline Result: {result}")
