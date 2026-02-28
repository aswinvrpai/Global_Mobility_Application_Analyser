import os
import sys
from dataclasses import dataclass
from src.logger.logger import setup_logger,log_file
from src.constants import *
from src.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from src.entity.config_entity import ModelPusherConfig, ModelTrainerConfig
from src.exception import CustomException
from src.entity.s3_estimator import S3ModelEstimator
from src.cloud_storage.aws_storage import SimpleStorageService

# Logger initialization
logger = setup_logger("model_pusher", log_file)

# Model Pusher class
class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig, model_evaluation_artifact: ModelEvaluationArtifact):
        """
        Initializes the ModelPusher class with the given configuration and artifacts.
        :param model_pusher_config: Configuration for model pushing.
        :param model_evaluation_artifact: Artifact containing model evaluation details.
        """
        self.model_pusher_config = model_pusher_config
        self.model_evaluation_artifact = model_evaluation_artifact
        self.s3 = SimpleStorageService()
        self.s3ModelEstimator = S3ModelEstimator(bucket_name=self.model_pusher_config.bucket_name, model_path=self.model_pusher_config.s3_model_key_path)
        
    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Initiates the model pushing process by saving the trained model to the S3 bucket.
        :return: ModelPusherArtifact containing details of the pushed model.
        """
        try:
            logger.info("Initiating model pushing process.")
            self.s3ModelEstimator.save_model(from_file=self.model_evaluation_artifact.trained_model_path, remove=False)
            logger.info("Model successfully pushed to S3 bucket.")
            
            return ModelPusherArtifact(s3_model_path=self.model_pusher_config.s3_model_key_path, bucket_name=self.model_pusher_config.bucket_name)
        except Exception as e:
            logger.error(f"Error while pushing the model: {e}")
            raise CustomException(e, sys)
    