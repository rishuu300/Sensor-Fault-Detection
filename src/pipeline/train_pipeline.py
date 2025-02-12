import os
import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException

class TrainingPipeline:
    def start_data_ingestion():
        try:
            data_ingestion = DataIngestion()
            feature_store_file_path = data_ingestion.initiate_data_ingestion()
            return feature_store_file_path
        
        except Exception as e:
            raise CustomException(e,sys) from e
    
    
    def start_data_transformation(self, feature_store_file_path):
        try:
            data_transformation = DataTransformation(feature_store_file_path)
            train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation()
            return train_arr, test_arr, preprocessor_path
        
        except Exception as e:
            raise CustomException(e,sys) from e
        
    
    def start_model_training(self, train_arr, test_arr):
        try:
            model_trainer = ModelTrainer()
            train_model_path = model_trainer.initiate_model_trainer(train_arr, test_arr)
            return train_model_path
        
        except Exception as e:
            raise CustomException(e,sys) from e
    
        
    def run_pipeline(self):
        try:
            feature_store_file_path = self.start_data_ingestion()
            train_arr, test_arr, preprocessor_path = self.start_data_transformation(feature_store_file_path)
            train_model_path = self.start_model_training(train_arr, test_arr)
            
            print("Training Completed. Trained model path: ", train_model_path)
            
        except Exception as e:
            raise CustomException(e,sys) from e