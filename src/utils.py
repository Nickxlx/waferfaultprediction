# contain important func throw out the pipeline
import os, sys
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from pymongo.mongo_client import MongoClient
from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import r2_score


def export_collection_as_dataframe(collection_name, db_name):
    try:
        # uniform resource indentifire
        uri = "mongodb+srv://nikhilsinghxlx:Nikhilsinghxlx@cluster0.9kjhcgg.mongodb.net/?retryWrites=true&w=majority"

        # create a new clinet and connect to the server
        mongo_client = MongoClient(uri)

        collection = mongo_client[db_name][collection_name]

        df = pd.DataFrame(list(collection.find()))

        if "_id" in df.columns.to_list():
            df = df.drop(columns=["_id"], axis=1)

        df.replace({"na": np.nan}, inplace=True)

        return df

    except Exception as e:
        raise CustomException(e, sys)
    
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X,y, models):
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        report = {}

        for model_name, model_instance in models.items():
            # model Building
            model = model_instance()
            
            # model Training 
            model.fit(x_train, y_train)

            # model report
            y_pred  = model.predict(x_test)

            score = r2_score(x_test, y_pred)
            
            # Store the score in the report dictionary with the model name as the key and value as score
            report[model_name] = score
            
        return report  # Return the evaluation report

    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)

def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)