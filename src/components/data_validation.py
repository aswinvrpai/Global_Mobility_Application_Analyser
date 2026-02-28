import os
import sys
import pandas as pd
from pandas import DataFrame

from evidently import Report
from evidently.metrics import *
from evidently.presets import *

from src.exception import CustomException
from src.logger.logger import setup_logger, log_file
from src.entity.config_entity import DataIngestionConfig, DataValidationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.constants import SCHEMA_FILE_PATH
from src.utils.main_utils import read_yaml_file, write_yaml_file

# Logger;
logger = setup_logger("data_validation", log_file)

class DataValidation:
    
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        '''
        Docstring for __init__
        
        :param self: Description
        :param data_ingestion_artifact: Description
        :type data_ingestion_artifact: DataIngestionArtifact
        :param data_validation_config: Description
        :type data_validation_config: DataValidationConfig
        '''
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema_file_data = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException(e,sys)
    
    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        '''
        Docstring for validate_number_of_columns
        
        :param self: class
        :param df: dataframe from the input data
        :type df: Pandas DataFrame
        '''
        try:
            status=len(dataframe.columns)==len(self.schema_file_data["columns"])
            logger.info(f"Number of Columns Validation:{status}")
            return status
        except Exception as e:
            raise CustomException(e,sys)
            
    def does_columns_exist(self,dataframe: DataFrame) -> bool:
        '''
        Docstring for does_columns_exist
        
        :param self: Description
        :param dataframe: Description
        :type dataframe: DataFrame
        '''
        
        try:
            
            missing_num_col_flag = False
            missing_cat_col_flag = False
            
            # Column list in Dataframe;
            dataframe_columns = dataframe.columns
            missing_num_columns = []
            for col_name in self.schema_file_data["numerical_columns"]:
                if col_name not in dataframe_columns:
                    missing_num_columns.append(col_name)
            
            if len(missing_num_columns):
                missing_num_col_flag = True
                logger.info(f"Missing Numerical Columns: {missing_num_columns}")
            else:
                logger.info(f"Missing Numerical Columns - NULL")
            
            missing_categorical_columns = []
            for col_name in self.schema_file_data["categorical_columns"]:
                if col_name not in dataframe_columns:
                    missing_categorical_columns.append(col_name)
            
            if len(missing_categorical_columns):
                missing_cat_col_flag = True
                logger.info(f"Missing Categorical Columns: {missing_categorical_columns}")
            else:
                logger.info(f"Missing Categorical Columns - NULL")
            
            return False if missing_num_col_flag or missing_cat_col_flag else True
            
        except Exception as e:
            raise CustomException(e,sys)
    
    @staticmethod
    def read_data(filepath):
        try:
            df = pd.read_csv(filepath)
            return df
        except Exception as e:
            raise CustomException(e,sys)
    
    def detect_data_drift(self, reference_df: DataFrame, current_df: DataFrame) -> bool:
        '''
        Docstring for detect_data_drift
        
        :param self: Description
        :param reference_df: Description
        :type reference_df: DataFrame
        :param current_df: Description
        :type current_df: DataFrame
        :return: Description
        :rtype: bool
        '''
        
        try:
            # Evidently report;
            report = Report([
                DataDriftPreset()
            ])

            # Evaluate for Data Drift Report;
            report_eval = report.run(reference_df, current_df)
            json_report = report_eval.json()
            print(f"json_report:{json_report}")
            
            # Write the Data dript report to a yaml File
            write_yaml_file(self.data_validation_config.drift_report_file,json_report)
            
            # Number of features
            metrics = json_report["metrics"]
            feature_metrics = [
                m for m in metrics
                if m["metric_name"].startswith("ValueDrift")
            ]
            total_features = len(feature_metrics)
            print("Total features:", total_features)
            
            # Drifted Features;
            drifted_features = []
            for m in feature_metrics:
                column = m["config"]["column"]
                p_value = m["value"]
                threshold = m["config"]["threshold"]

                if p_value < threshold:
                    drifted_features.append(column)
            
            # Number of Drifted Features;
            drift_count_metric = next(
                m for m in metrics
                if m["metric_name"].startswith("DriftedColumnsCount")
            )

            drifted_count = drift_count_metric["value"]["count"]
            drifted_share = drift_count_metric["value"]["share"]

            print("Drifted feature count:", drifted_count)
            print("Drifted feature share:", drifted_share)

            # Overal Drift;
            dataset_drift = drifted_share >= drift_count_metric["config"]["drift_share"]
            print("Dataset drift detected:", dataset_drift)

            summary = {
                "total_features": total_features,
                "drifted_feature_count": int(drifted_count),
                "drifted_features": drifted_features,
                "dataset_drift": dataset_drift
            }
            print(summary)
            
            return dataset_drift
        
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_validation(self):
        
        try:
            
            validation_err_msg = ""
            logger.info("Starting Data Validation")
            
            # Get Train and Test CSV;
            logger.info("Retrieve Training and Test sets")
            train_df,test_df = (DataValidation.read_data(self.data_ingestion_artifact.training_file_path),DataValidation.read_data(self.data_ingestion_artifact.testing_file_path))
            
            # Validate Number of Columns;
            status = self.validate_number_of_columns(train_df)
            logger.info(f"Training Set: Validate Number of Columns - Status:{status}")
            if not status:
                validation_err_msg += f"Columns missing in Training Set"
            
            status = self.validate_number_of_columns(test_df)
            logger.info(f"Test Set: Validate Number of Columns - Status:{status}")
            if not status:
                validation_err_msg += f"Columns missing in Test Set"
            
            # Does all columns Exist;
            status = self.does_columns_exist(train_df)
            logger.info(f"Training Set: Missing Columns - Status:{status}")
            if not status:
                validation_err_msg += f"Columns missing in Training Set"
            
            status = self.does_columns_exist(test_df)
            logger.info(f"Test Set: Missing Columns - Status:{status}")
            if not status:
                validation_err_msg += f"Columns missing in Test Set"
            
            # Data Drift To be added;
            
            # Flag for Validation status;
            validation_status = True if len(validation_err_msg) == 0 else False
            
            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_err_msg,
                drift_report_path=""
            )
            
            logger.info("Data Validation Finished")
            
            return data_validation_artifact
            
        except Exception as e:
            logger.info(f"Exception Error :{e}")
            raise CustomException(e,sys)
        pass
            
if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv(r"C:\Work_Directory\Learn\DS_Projects\Global_Mobility_Application_Analyser\artifacts\02_04_2026__22_21_04\data_ingestion\feature_store\us_visa_data.csv")
    print(f"Columns List in Input Dataframe : {df.columns.to_list()}")
    clss = DataValidation('a','b')
    # clss.validate_number_of_columns(df)
    # ret = clss.does_columns_exist(df)
    # print(ret)
    
    ref_df = df[:60]
    curr_df = df[61:120]
    clss.detect_data_drift(ref_df, curr_df)
            

