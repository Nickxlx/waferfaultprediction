import os, sys
import pandas as pd 
from pymongo import MongoClient

from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.utils import export_collection_as_dataframe


# @dataclass
# class DataIngestionConfig:
#     artifact_folder: str = os.path.join(artifact_folder)

# class DataIngestion:
#     def __init__(self):
#         # Inisiating the confugration and utilites
#         self.data_ingestion_config = DataIngestionConfig()
#         self.utils = M


@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join("artifacts", "train.csv")

    test_data_path = os.path.join("artifacts", "test.csv")   

    raw_data_path = os.path.join("artifacts", "raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(Self):
        logging.info("Data Ingetion method starts")

        try:
            df: pd.DataFrame = export_collection_as_dataframe(
                db_name="Projects", 
                collection_name="waferfault"
            )
            logging.info("Exported collection as dataframe")

            # # reading the dataset
            # df = pd.read_csv("notebooks/data", "wafer.csv")
            # logging.info("Data has been readed ")

            os.makedirs(os.path.dirname(Self.ingestion_config.raw_data_path), exist_ok=True)

            # Lets store the raw data 
            df.to_csv(Self.ingestion_config.raw_data_path, index=False, header = True)

            # lets split the data set
            train_set, test_set = train_test_split(df, train_size=0.2, random_state=42)

            # lets store the train_set
            train_set.to_csv(Self.ingestion_config.train_data_path, index = False, header = True)
                        
            # lets store the test_set
            test_set.to_csv(Self.ingestion_config.test_data_path, index = False, header = True)

            logging.info("Data Ingestion Has competed")

            # paths of the traning and test_set file path  
            return(Self.ingestion_config.train_data_path,  
                    Self.ingestion_config.test_data_path)

        except Exception as e:
            raise CustomException(e,sys)        