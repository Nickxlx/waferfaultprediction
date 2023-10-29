import os, sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging

from dataclasses import dataclass

from sklearn.preprocessing import FunctionTransformer, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.combine import SMOTETomek

from src.utils import save_object

@dataclass
class DataTrasnformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTrasnformationConfig()

    def get_data_transformer_object(self):
        try:
            # define the steps for the preprocessor pipeline
            imputer_step = ('imputer', SimpleImputer(strategy='constant', fill_value=0))
            scaler_step = ('scaler', RobustScaler())

            preprocessor = Pipeline(
                steps=[
                imputer_step,
                scaler_step
                ]
            )
            
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transfromation(self,train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data is completed")

            logging.info("Obtaining preprocessing object")

            preprocessor = self.get_data_transformer_object()

            target_column_name = "Good/Bad"
            target_column_mapping = {"+1": 0, "-1": 1}

            logging.info("Dividing the data set") 

            # training dataframe
            input_train_df = train_df.drop(target_column_name, axis=1)
            target_train_df = train_df[target_column_name].map(target_column_mapping)

            # testing dataframe
            input_test_df = test_df.drop(target_column_name, axis = 1)
            target_test_df = test_df[target_column_name]. map(target_column_mapping)

            logging.info("Applying preprocessing object on training and testing datasets.")

            transformed_input_train_df = preprocessor.fit_transform(input_train_df)
            transformed_input_test_df = preprocessor.transform(input_test_df) 

            # # applling imputation
            # smt = SMOTETomek(sampling_strategy = "minority")

            # final_input_train_df, final_target_train_df = smt.fit_resample(transformed_input_train_df, target_train_df)

            # final_input_test_df, final_target_test_df = smt.fit_resample(transformed_input_test_df, target_test_df)

            # # concating train and test arr with target arr
            # train_arr = np.c_[final_input_train_df, np.array(final_target_train_df)]

            # test_arr = np.c_[final_input_test_df, np.array(final_target_test_df)]

            # concating train and test arr with target arr
            train_arr = np.c_[transformed_input_train_df, np.array(target_train_df)]

            test_arr = np.c_[transformed_input_test_df, np.array(target_test_df)]

            save_object(
                self.data_transformation_config.preprocessor_obj_file_path, 
                obj = preprocessor
            )
            logging.info("Preprocessor pickle is created and saved")

            return(
                train_arr, test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)
            