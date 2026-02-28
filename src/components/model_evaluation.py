from dataclasses import dataclass
from src.logger.logger import setup_logger,log_file
from src.constants import *
from src.entity.artifact_entity import ModelEvaluationArtifact, ModelTrainerArtifact, DataIngestionArtifact
from src.exception import CustomException
from src.entity.estimator import TargetValueMapping, VisaModel
from src.entity.s3_estimator import S3ModelEstimator
from src.entity.config_entity import ModelEvaluationConfig

import sys
import os
from sklearn.metrics import f1_score
from typing import Optional
import pandas as pd

# Initialize logger
logger = setup_logger("model_evaluation", log_file)

@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float

# Model Evaluation class
class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        """
        Initializes the ModelEvaluation class with the given configuration and artifacts.
        :param model_evaluation_config: Configuration for model evaluation.
        :param data_ingestion_artifact: Artifact containing data ingestion details.
        :param model_trainer_artifact: Artifact containing model trainer details.
        """
        self.model_evaluation_config = model_evaluation_config
        self.data_ingestion_artifact = data_ingestion_artifact
        self.model_trainer_artifact = model_trainer_artifact
    
    def get_best_model(self) -> Optional[S3ModelEstimator]:
        """
        Retrieves the best model from the production environment.
        :return: The best S3ModelEstimator if available, else None.
        """
        try:
            logger.info("Checking for existing model in production environment.")
            bucket_name = self.model_evaluation_config.bucket_name
            model_path = self.model_evaluation_config.s3_model_key_path
            visa_estimator = S3ModelEstimator(bucket_name=bucket_name, model_path=model_path)
            if visa_estimator.is_model_present(model_path=model_path):
                logger.info("Existing model found in production environment. Loading the model.")
                return visa_estimator
            else:
                logger.info("No existing model found in production environment.")
                return None
        except Exception as e:
            logger.error(f"Error while retrieving the best model: {e}")
            raise CustomException(e, sys)
        
    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Method Name :   evaluate_model
        Description :   This function is used to evaluate trained model 
                        with production model and choose best model 
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            
            # Test dataset loading and preprocessing
            test_df = pd.read_csv(self.data_ingestion_artifact.testing_file_path)
            test_df['company_age'] = CURRENT_YEAR-test_df['yr_of_estab']

            x, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]
            y = y.replace(
                TargetValueMapping()._asdict()
            )

            # Local Model F1 Score Calculation
            trained_model_f1_score = self.model_trainer_artifact.model_metric_artifact.model_f1_score

            # S3 Model F1 Score Calculation
            best_model_f1_score=None
            best_model = self.get_best_model()
            if best_model is not None:
                y_hat_best_model = best_model.predict(x)
                best_model_f1_score = f1_score(y, y_hat_best_model)
            
            tmp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score
            result = EvaluateModelResponse(trained_model_f1_score=trained_model_f1_score,
                                best_model_f1_score=best_model_f1_score,
                                is_model_accepted=trained_model_f1_score > tmp_best_model_score,
                                difference=trained_model_f1_score - tmp_best_model_score
                            )
            logger.info(f"Result: {result}")
            return result

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model evaluation
        
        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """  
        try:
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_evaluation_config.s3_model_key_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_path,
                changed_accuracy=evaluate_model_response.difference)

            logger.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise CustomException(e, sys) from e