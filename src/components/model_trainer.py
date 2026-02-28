import sys
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from neuro_mf import ModelFactory
from sklearn.pipeline import Pipeline
import os
import pandas as pd
from pandas import DataFrame
import numpy as np
from typing import Tuple

from src.exception import CustomException
from src.logger.logger import setup_logger, log_file
from src.constants import MODEL_TRAINER_CONFIG_PATH, MODEL_TRAINED_EXPECTED_SCORE
from src.entity.config_entity import ModelTrainerConfig
from src.utils.main_utils import read_yaml_file, load_object, save_object, load_numpy_array_data
from src.entity.artifact_entity import ModelTrainerArtifact, ClassificationMetricArtifact, DataTransformationArtifact
from src.entity.estimator import VisaModel

# File specific Logger;
logger = setup_logger('model_trainer', log_file)

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifact = data_transformation_artifact
        
    def get_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[object, object]:
        """
        Method Name :   get_model_object_and_report
        Description :   This function uses neuro_mf to get the best model object and report of the best model
        
        Output      :   Returns metric artifact object and best model object
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            
            # Model Factory;
            logger.info("Using neuro_mf to get best model object and report")
            model_factory = ModelFactory(model_config_path=self.model_trainer_config.model_config_path)
            
            # Train test split;
            x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]
            
            # Get best model object and report;
            best_model_detail = model_factory.get_best_model(
                X=x_train,y=y_train,base_accuracy=self.model_trainer_config.expected_score
            )
            model_obj = best_model_detail.best_model
            logger.info("Retrieved best model object from model factory")

            # Predict on test data using best model;
            y_pred = model_obj.predict(x_test)
            logger.info("Used best model to predict on test data")
            
            # Calculate metrics;
            accuracy = accuracy_score(y_test, y_pred) 
            f1 = f1_score(y_test, y_pred)  
            precision = precision_score(y_test, y_pred)  
            recall = recall_score(y_test, y_pred)
            metric_artifact = ClassificationMetricArtifact(model_f1_score=f1, model_precision=precision, model_recall=recall)
            
            # Log the metrics;
            logger.info(f"Classification Metrics: Accuracy={accuracy}, F1-score={f1}, Precision={precision}, Recall={recall}")
            
            return best_model_detail, metric_artifact
        
        except Exception as e:
            raise CustomException(e, sys) from e
    
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates a model trainer steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        
        try:
            
            # Load the transformed training and testing numpy array data
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_path)
            
            # Get the best model object and report from the get_model_object_and_report method
            best_model_detail ,metric_artifact = self.get_model_object_and_report(train=train_arr, test=test_arr)
            
            # Load the preprocessor object
            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.preprocessor_object_path)

            # Check if the best model score is more than expected score or not. If not then raise an exception
            if best_model_detail.best_score < self.model_trainer_config.expected_score:
                logger.info("No best model found with score more than base score")
                raise CustomException("No best model found with score more than base score", sys)

            # Create a usvisa model object which contains both the preprocessor and model object. This will be used in the prediction pipeline to predict on the transformed features
            usvisa_model = VisaModel(preprocessing_object=preprocessing_obj,
                                       trained_model_object=best_model_detail.best_model)
            logger.info("Created Visa model object with preprocessor and model")
            logger.info("Created best model file path.")
            save_object(self.model_trainer_config.trained_model_path, usvisa_model)

            # Prepare the model trainer artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_path=self.model_trainer_config.trained_model_path,
                model_metric_artifact=metric_artifact,
            )
            logger.info(f"Model trainer artifact: {model_trainer_artifact}")
            
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys) from e
