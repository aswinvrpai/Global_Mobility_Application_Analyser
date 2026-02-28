import os
import sys

from src.components import data_transformation
from src.exception import CustomException
from src.entity.config_entity import DataIngestionConfig, DataTransformationConfig, DataValidationConfig, ModelTrainerConfig, ModelEvaluationConfig, ModelPusherConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact, DataValidationArtifact, ModelTrainerArtifact, ModelEvaluationArtifact, ModelPusherArtifact
from src.logger.logger import setup_logger, log_file

# Initialize logger
logger = setup_logger("training_pipeline", log_file)

class TrainingPipeline:
    '''
    TrainingPipeline class to orchestrate the training pipeline components.
    '''
    def __init__(self):
        '''
        Initializes the TrainingPipeline object with the given configuration.
        :param self: Description
        :param data_ingestion_config: Description
        :type data_ingestion_config: DataIngestionConfig
        '''
        try:
            logger.info("Training Pipeline initialized.")
            self.data_ingestion_config = DataIngestionConfig()
            self.data_validation_config = DataValidationConfig()
            self.data_transformation_config = DataTransformationConfig()
        except Exception as e:
            raise CustomException(e, sys)
    
    def start_data_ingestion(self) -> DataIngestionArtifact:
        '''
        This function initiates the data ingestion process.
        
        :param self: Description
        :return: Description
        :rtype: DataIngestionArtifact
        '''
        try:
            from src.components.data_ingestion import DataIngestion
            
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            
            logger.info(f"Data Ingestion Artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def start_data_validation(self,data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        '''
        This function initiates the data validation process.
        
        :param self: Description
        :return: Description
        :rtype: DataValidationArtifact
        '''
        try:
            from src.components.data_validation import DataValidation
            
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                data_validation_config=self.data_validation_config)
            data_validation_artifact = data_validation.initiate_data_validation()
            
            logger.info(f"Data Validation Artifact: {data_validation_artifact}")
            return data_validation_artifact
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        """
        This method of TrainPipeline class is responsible for starting data transformation component
        """
        try:
            
            from src.components.data_transformation import DataTransformation
            data_transformation = DataTransformation(data_ingestion_artifact=data_ingestion_artifact,
                                    data_transformation_config=self.data_transformation_config,
                                    data_validation_artifact=data_validation_artifact)
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            return data_transformation_artifact
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        """
        This method of TrainPipeline class is responsible for starting model trainer component
        """
        try:
            from src.components.model_trainer import ModelTrainer
            
            model_trainer = ModelTrainer(model_trainer_config=ModelTrainerConfig(), data_transformation_artifact=data_transformation_artifact)
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            
            logger.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def start_model_evaluation(self, data_ingestion_artifact: DataIngestionArtifact, model_trainer_artifact: ModelTrainerArtifact) -> ModelEvaluationArtifact:
        """
        This method of TrainPipeline class is responsible for starting model evaluation component
        """
        try:
            from src.components.model_evaluation import ModelEvaluation
            
            model_evaluation = ModelEvaluation(model_evaluation_config=ModelEvaluationConfig(),
                                            data_ingestion_artifact=data_ingestion_artifact,
                                            model_trainer_artifact=model_trainer_artifact)
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
            
            logger.info(f"Model Evaluation Artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def start_model_pusher(self, model_evaluation_artifact: ModelEvaluationArtifact) -> ModelPusherArtifact:
        """
        This method of TrainPipeline class is responsible for starting model pusher component
        """
        try:
            from src.components.model_pusher import ModelPusher
            
            model_pusher = ModelPusher(model_pusher_config=ModelPusherConfig(), model_evaluation_artifact=model_evaluation_artifact)
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            
            logger.info(f"Model Pusher Artifact: {model_pusher_artifact}")
            return model_pusher_artifact
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def run_pipeline(self) -> None:
        '''
        This function runs the entire training pipeline.
        
        :param self: Description
        :return: Description
        :rtype: None
        '''
        try:
            
            # Data ingestion;
            data_ingestion_artifact = self.start_data_ingestion()
            
            # Data Validation;
            data_validation_artifact = self.start_data_validation            (data_ingestion_artifact=data_ingestion_artifact)
            
            # Data Transformation;
            data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact=data_ingestion_artifact,
            data_validation_artifact=data_validation_artifact)
            
            # Model Trainer;
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            
            # Model Evaluation;
            model_evaluation_artifact = self.start_model_evaluation(data_ingestion_artifact=data_ingestion_artifact, model_trainer_artifact=model_trainer_artifact)
            
            if not model_evaluation_artifact.is_model_accepted:
                logger.info("Trained model is not better than the best model. Model pusher will not be initiated.")
                return
            
            # Model Pusher;
            model_pusher_artifact = self.start_model_pusher(model_evaluation_artifact=model_evaluation_artifact)
            
            # Log message;
            logger.info("Training Pipeline executed successfully.")
        except Exception as e:
            raise CustomException(e, sys)