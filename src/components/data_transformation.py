import sys

from flask import logging
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer 

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from src.entity.config_entity import DataTransformationConfig 
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import CustomException
from src.logger.logger import setup_logger, log_file
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file, drop_columns
from src.entity.estimator import TargetValueMapping 

# Logger;
logger = setup_logger("data_transformation", log_file)

class DataTransformation:
    
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_artifact: DataValidationArtifact, data_transformation_config: DataTransformationConfig):
        '''
        Docstring for __init__
        
        :param self: Description
        :param data_ingestion_artifact: Description
        :type data_ingestion_artifact: DataIngestionArtifact
        :param data_validation_artifact: Description
        :type data_validation_artifact: DataValidationArtifact
        :param data_transformation_config: Description
        :type data_transformation_config: DataTransformationConfig
        '''
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            self.schema_file_data = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException(e,sys)
    
    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        '''
        Docstring for read_data
        
        :param self: Description
        :param file_path: Description
        :type file_path: str
        :return: Description
        :rtype: pd.DataFrame
        '''
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e,sys)
    
    def get_data_transformer_object(self) -> Pipeline:
        """
        Method Name :   get_data_transformer_object
        Description :   This method creates and returns a data transformer object for the data
        
        Output      :   data transformer object is created and returned 
        On Failure  :   Write an exception log and then raise an exception
        """
        logger.info(
            "Entered get_data_transformer_object method of DataTransformation class"
        )

        try:
            
            # Logger info
            logger.info("Got numerical cols from schema config")

            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder()
            ordinal_encoder = OrdinalEncoder()

            # Logger info
            logger.info("Initialized StandardScaler, OneHotEncoder, OrdinalEncoder")

            # One Hot Encoding columns, Ordinal Encoding columns and Transformation columns
            oh_columns = self.schema_file_data['oh_columns']
            or_columns = self.schema_file_data['or_columns']
            transform_columns = self.schema_file_data['transform_columns']
            num_features = self.schema_file_data['num_features']

            logger.info("Initialize PowerTransformer")

            # Power Transformer
            transform_pipe = Pipeline(steps=[
                ('transformer', PowerTransformer(method='yeo-johnson'))
            ])
            
            # Preprocessor Column Transformer
            preprocessor = ColumnTransformer(
                [
                    ("OneHotEncoder", oh_transformer, oh_columns),
                    ("Ordinal_Encoder", ordinal_encoder, or_columns),
                    ("Transformer", transform_pipe, transform_columns),
                    ("StandardScaler", numeric_transformer, num_features)
                ]
            )
            
        
            logger.info("Created preprocessor object from ColumnTransformer")

            logger.info(
                "Exited get_data_transformer_object method of DataTransformation class"
            )
            
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Method Name :   initiate_data_transformation
        Description :   This method initiates the data transformation component for the pipeline 
        
        Output      :   data transformer steps are performed and preprocessor object is created  
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            if self.data_validation_artifact.validation_status:
                
                # Prepocessing object
                logger.info("Starting data transformation")
                preprocessor = self.get_data_transformer_object()
                logger.info("Got the preprocessor object")

                # Retrieveing train and test file data frames
                train_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.training_file_path)
                test_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.testing_file_path)

                # Getting input feature and target feature from training and testing dataframes
                input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_train_df = train_df[TARGET_COLUMN]
                logger.info("Got train features and test features of Training dataset")

                # Adding new feature - company_age
                input_feature_train_df['company_age'] = CURRENT_YEAR-input_feature_train_df['yr_of_estab']
                logger.info("Added company_age column to the Training dataset")

                # Drop columns which are not required for model training
                drop_cols = self.schema_file_data['drop_columns']
                logger.info("drop the columns in drop_cols of Training dataset")
                input_feature_train_df = drop_columns(df=input_feature_train_df, cols = drop_cols)
                
                # Replace target feature values as per the mapping in Training dataset
                target_feature_train_df = target_feature_train_df.replace(
                    TargetValueMapping()._asdict()
                )

                # Test Dataset
                input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_test_df = test_df[TARGET_COLUMN]

                # Adding new feature - company_age
                input_feature_test_df['company_age'] = CURRENT_YEAR-input_feature_test_df['yr_of_estab']
                logger.info("Added company_age column to the Test dataset")

                # Drop columns which are not required for model training
                input_feature_test_df = drop_columns(df=input_feature_test_df, cols = drop_cols)
                logger.info("drop the columns in drop_cols of Test dataset")

                # Replace target feature values as per the mapping in Test dataset
                target_feature_test_df = target_feature_test_df.replace(
                    TargetValueMapping()._asdict()
                )
                logger.info("Got train features and test features of Testing dataset")
                
                # Applying fit transform preprocessor object on training dataframe
                logger.info(
                    "Applying preprocessing object on training dataframe and testing dataframe"
                )
                input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)

                # Applying transform on test data
                logger.info(
                    "Used the preprocessor object to fit transform the train features"
                )
                input_feature_test_arr = preprocessor.transform(input_feature_test_df)
                logger.info("Used the preprocessor object to transform the test features")

                # SMOTEENN for handling imbalanced dataset on Training Dataset
                logger.info("Applying SMOTEENN on Training dataset")
                smt = SMOTEENN(sampling_strategy="minority")
                input_feature_train_final, target_feature_train_final = smt.fit_resample(
                    input_feature_train_arr, target_feature_train_df
                )
                logger.info("Applied SMOTEENN on training dataset")

                # SMOTEENN for handling imbalanced dataset on Testing Dataset
                logger.info("Applying SMOTEENN on testing dataset")
                input_feature_test_final, target_feature_test_final = smt.fit_resample(
                    input_feature_test_arr, target_feature_test_df
                )
                logger.info("Applied SMOTEENN on testing dataset")
                logger.info("Created train array and test array")

                # Combining input and target features for train datasets
                train_arr = np.c_[
                    input_feature_train_final, np.array(target_feature_train_final)
                ]

                # Combining input and target features for test dataset
                test_arr = np.c_[
                    input_feature_test_final, np.array(target_feature_test_final)
                ]

                # Saving the preprocessor object and transformed train and test arrays to respective file paths
                save_object(self.data_transformation_config.preprocessor_object_path, preprocessor)
                save_numpy_array_data(self.data_transformation_config.transformed_train_path, array=train_arr)
                save_numpy_array_data(self.data_transformation_config.transformed_test_path, array=test_arr)

                logger.info("Saved the preprocessor object")

                logger.info(
                    "Exited initiate_data_transformation method of Data_Transformation class"
                )

                data_transformation_artifact = DataTransformationArtifact(
                    preprocessor_object_path=self.data_transformation_config.preprocessor_object_path,
                    transformed_train_path=self.data_transformation_config.transformed_train_path,
                    transformed_test_path=self.data_transformation_config.transformed_test_path
                )
                return data_transformation_artifact
            else:
                raise Exception(self.data_validation_artifact.message)

        except Exception as e:
            raise CustomException(e, sys) from e
    
if __name__ == "__main__":
    try:
        data_transformation = DataTransformation("","","")
        preprocessor = data_transformation.get_data_transformer_object()
        
        # import joblib
        # preprocessor = joblib.load(r"C:\Work_Directory\Learn\DS_Projects\Global_Mobility_Application_Analyser\artifacts\02_25_2026__12_06_45\data_transformation\transformed_object\preprocessor.pkl")
        # from src.pipeline.prediction_pipeline import ModelData
        # input_data = ModelData(
        #     continent="Asia",
        #     education_of_employee="Master's",
        #     has_job_experience="N",
        #     requires_job_training="N",
        #     no_of_employees=100,
        #     region_of_employment="South",
        #     prevailing_wage=148138.59,
        #     unit_of_wage="Year",
        #     full_time_position="Y",
        #     company_age=10
        # )
        # input_df = input_data.get_input_data_frame()
        # pred_arr = preprocessor.transform(input_df)
        # logger.info(f"Preprocessor object created: {pred_arr}")
        
        logger.info(f"Preprocessor object created: {preprocessor}")
    except Exception as e:
        raise CustomException(e, sys)