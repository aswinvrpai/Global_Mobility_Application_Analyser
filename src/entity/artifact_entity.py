from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    feature_store_path: str
    training_file_path: str
    testing_file_path: str 
    
@dataclass
class DataValidationArtifact:
    validation_status: str
    message: str
    drift_report_path: str

@dataclass
class DataTransformationArtifact:
    transformed_train_path: str
    transformed_test_path: str
    preprocessor_object_path: str
    
@dataclass
class ClassificationMetricArtifact:
    model_precision: float
    model_recall: float
    model_f1_score: float
    
@dataclass
class ModelTrainerArtifact:
    trained_model_path: str
    model_metric_artifact: ClassificationMetricArtifact
    
@dataclass
class ModelEvaluationArtifact:
    is_model_accepted:bool
    changed_accuracy:float
    s3_model_path:str 
    trained_model_path:str
    
@dataclass
class ModelPusherArtifact:
    bucket_name:str
    s3_model_path:str