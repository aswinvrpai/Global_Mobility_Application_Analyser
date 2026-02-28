import os
from datetime import datetime

# # Load values from .env file;
# from dotenv import dotenv_values
# config = dotenv_values(".env")
# MONGO_DB_URL = config['MONGO_DB_URL']

from dotenv import load_dotenv
load_dotenv()  # reads .env and loads it into environment
MONGO_DB_URL = os.getenv("MONGO_DB_URL")

DATABASE_NAME = "VisaData"
COLLECTION_NAME = "VisaApplications"

PIPELINE_NAME: str = "usvisa"
ARTIFACT_DIR: str = "artifacts"

MODEL_FILE_NAME: str = "model.pkl"
PREPROCESSOR_FILE_NAME: str = "preprocessor.pkl"

TARGET_COLUMN: str = "case_status"
CURRENT_YEAR: int = datetime.now().year

FILE_NAME = "us_visa_data.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")

AWS_ACCESS_KEY: str = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY: str = os.getenv("AWS_SECRET_KEY")
AWS_REGION_NAME: str = os.getenv("AWS_REGION_NAME")

# YAML Config Folder
CONFIG_PATH = "config"

# Data Ingestion constants
DATA_INGESTION_COLLECTION_NAME: str = "VisaApplications"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2
DATA_INGESTION_ARTIFACT_DIR: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested_data"

# Data Validation constants
DATA_VALIDATION_DIR: str = "data_validation"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "data_drift"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "drift_report.yaml"

# Data Transformation constants
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

# Model Trainer constants
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_FILE_NAME: str = "model.pkl"
MODEL_TRAINED_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_CONFIG_PATH: str = os.path.join(CONFIG_PATH, "model.yaml")

# Model Evaluation constants
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_BUCKET_NAME = "visabucket2025"
MODEL_PUSHER_S3_KEY_PATH = "model-registry"

# AWS S3 constants
AWS_ACCESS_KEY_ID_ENV_KEY = os.getenv("AWS_ACCESS_KEY_ID_ENV_KEY")
AWS_SECRET_ACCESS_KEY_ENV_KEY = os.getenv("AWS_SECRET_ACCESS_KEY_ENV_KEY")
REGION_NAME = os.getenv("REGION_NAME")

APP_HOST = "127.0.0.1"
APP_PORT = "8000"