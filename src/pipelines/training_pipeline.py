import sys, os

from src.components.data_ingestion import DataIngestion
from src.components.data_tranformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException

# class TrainingPipeline:

#     def start_data_ingetion(self):
#         try:
#             data_ingesion = DataIngestion()
#             feature_store_file_path = data_ingesion.initiate_data_ingestion()
#             return feature_store_file_path
#         except Exception as e :
#             CustomException(e, sys)




class TrainPipeline:

    def __init__(self):
        # creating class object
        self.data_ingestion = DataIngestion()
        self.data_trasformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def run_pipeline(self):
        try:
            train_path, test_path = self.data_ingestion.initiate_data_ingestion()

            train_arr, test_arr, preprocessor_file_path = self.data_trasformation.initiate_data_transfromation(train_path, test_path)

            r2_square = self.model_trainer.initiate_model_trainer(
                train_arr=train_arr,    
                test_arr=test_arr,
                preprocessor_path=preprocessor_file_path
                ) 
            
            print("training completed. Trained model score : ", r2_square)


        except Exception as e:
            raise CustomException(e,sys)        

