import os, sys

from src.exception import CustomException
from src.logger import logging

from dataclasses import dataclass

from sklearn.ensemble import  (
        RandomForestClassifier,
        AdaBoostClassifier, 
        GradientBoostingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.utils import save_object, load_object
from src.utils import evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    trained_model_path= os.path.join("artifacts","model.pkl" )


class CustomModel:
    def __init__(self, preprocessing_object, trained_model_object):
        self.preprocessing_object = preprocessing_object

        self.trained_model_object = trained_model_object

    def predict(self, X):
        transformed_feature = self.preprocessing_object.transform(X)

        return self.trained_model_object.predict(transformed_feature)

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"


class ModelTrainer:
    def __init__(Self):
        Self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr, preprocessor_path):
        try:
            logging.info("Splitting Training and testing data.")

            x_train,y_train,x_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
                "Randon forest":RandomForestClassifier(),
                "AdaBoost Classifire":AdaBoostClassifier(),
                "Gradient Boosting Classifire":GradientBoostingClassifier(),
                "K-Mean Classifire": KNeighborsClassifier(),
                "Decision Tree Classifire":DecisionTreeClassifier(),
                "XGBClassifire":XGBClassifier()
            }

            logging.info("Initiating Model traning and Evaluation")

            model_report = evaluate_models(X=x_train, y=y_train, models=models)

            # To get best score model
            best_model_score = max(sorted(model_report.values()))

            # To get best score model name 
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
                ]
            
            if best_model_score < 0.6:
                raise Exception("No best model found")
            
            # best model to create pickel
            best_model = models[best_model_name]

            # print(f"Best Model Found , Model Name :{best_model_name}, R2 Score: {best_model_score}")

            # logging.info(f"Best Model Found , Model Name :{best_model_name}, R2 Score: {best_model_score}")

            logging.info(f"Best found model on both training and testing dataset")

            # loading preprocessing model 
            preprocessing_obj = load_object(file_path=preprocessor_path)

            custom_model = CustomModel(
                preprocessing_object=preprocessing_obj,
                trained_model_object=best_model,
            )

            logging.info(
                f"Saving model at path: {self.model_trainer_config.trained_model_file_path}"
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=custom_model,
            )
            
            predicted = best_model.predict(x_test)

            r2_square = r2_square(y_test, predicted)

            return r2_square
        
        except Exception as e:
            raise CustomException(e, sys)