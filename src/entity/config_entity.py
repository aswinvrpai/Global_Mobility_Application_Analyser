import os
from dataclasses import dataclass
from src.constants import *
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y__%H_%M_%S")

@dataclass
class TrainingPipelineConfig:
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    PIPELINE_NAME: str = PIPELINE_NAME
    TIMESTAMP: str = TIMESTAMP

training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()

@dataclass
class DataIngestionConfig:
    database_name: str = DATABASE_NAME
    collection_name: str = DATA_INGESTION_COLLECTION_NAME
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    data_ingestion_artifact_dir: str = os.path.join(
        training_pipeline_config.artifact_dir,
        DATA_INGESTION_ARTIFACT_DIR
    )
    feature_store_path: str = os.path.join(
        data_ingestion_artifact_dir,
        DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME
    )
    ingested_dir: str = os.path.join(
        data_ingestion_artifact_dir,
        DATA_INGESTION_INGESTED_DIR
    )
    training_file_path: str = os.path.join(ingested_dir, TRAIN_FILE_NAME)
    testing_file_path: str = os.path.join(ingested_dir, TEST_FILE_NAME)
    
@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR)
    drift_report_file: str = os.path.join(data_validation_dir,DATA_VALIDATION_DRIFT_REPORT_DIR,DATA_VALIDATION_DRIFT_REPORT_FILE_NAME)
    
@dataclass
class DataTransformationConfig:
    data_transformation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR_NAME)
    transformed_data_dir: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR)
    transformed_train_path: str = os.path.join(transformed_data_dir, TRAIN_FILE_NAME.replace(".csv", ".npy"))
    transformed_test_path: str = os.path.join(transformed_data_dir, TEST_FILE_NAME.replace(".csv", ".npy"))
    preprocessor_object_dir: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR)
    preprocessor_object_path: str = os.path.join(preprocessor_object_dir, "preprocessor.pkl")
    
@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME)
    trained_model_dir: str = os.path.join(model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR)
    trained_model_path: str = os.path.join(trained_model_dir, MODEL_TRAINER_TRAINED_MODEL_FILE_NAME)
    expected_score: float = MODEL_TRAINED_EXPECTED_SCORE
    model_config_path: str = MODEL_TRAINER_CONFIG_PATH
    
@dataclass
class ModelEvaluationConfig:
    changed_threshold_score: float = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE
    bucket_name: str = MODEL_BUCKET_NAME
    s3_model_key_path: str = MODEL_FILE_NAME
    
@dataclass
class ModelPusherConfig:
    bucket_name: str = MODEL_BUCKET_NAME
    s3_model_key_path: str = MODEL_FILE_NAME
    
@dataclass
class ModelPredictorConfig:
    bucket_name: str = MODEL_BUCKET_NAME
    s3_model_key_path: str = MODEL_FILE_NAME