import os
import from_root
import pandas as pd
import sys
from sklearn.model_selection import train_test_split

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.logger.logger import setup_logger, log_file
from src.exception import CustomException
from src.data_access.data_access import DataAccess
from src.constants import FILE_NAME, TRAIN_FILE_NAME, TEST_FILE_NAME

# Initialize logger
logger = setup_logger("data_ingestion", log_file)

class DataIngestion:
    '''
    DataIngestion class for handling data ingestion from MongoDB to local storage.
    '''
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        '''
        Initializes the DataIngestion object with the given configuration
        :param self: Description
        :param data_ingestion_config: Description
        :type data_ingestion_config: DataIngestionConfig
        '''
        try:
            logger.info(f"Data Ingestion log started.")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CustomException(e, sys)
    
    def export_data_to_feature_store(self) -> None:
        '''
        This function initiates the data ingestion process.
        
        :param self: Description
        :return: Description
        :rtype: DataIngestionArtifact
        '''
        try:
            # Exporting data from MongoDB to DataFrame
            data_access = DataAccess()
            df: pd.DataFrame = data_access.export_data_from_db(
                collection_name=self.data_ingestion_config.collection_name,
                database_name=self.data_ingestion_config.database_name
            )
            
            # Loading data from Local CSV file to DataFrame as MongoDB export is not working due to some issues. Just for testing purpose, we are loading data from local CSV file.
            # df: pd.DataFrame = pd.read_csv(r"C:\Work_Directory\Learn\DS_Projects\Global_Mobility_Application_Analyser\artifacts\02_08_2026__14_16_26\data_ingestion\feature_store\us_visa_data.csv")
            
            logger.info("Exported data from MongoDB to DataFrame.")
            
            # Creating feature store directory if it doesn't exist
            feature_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_path)
            os.makedirs(feature_store_dir, exist_ok=True)
            
            # Saving the DataFrame to feature store path
            df.to_csv(self.data_ingestion_config.feature_store_path, index=False)
            logger.info(f"Saved data to feature store at {self.data_ingestion_config.feature_store_path}.")
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def split_data_as_train_test(self) -> None:
        '''
        This function splits the data into training and testing sets and saves them to respective file paths.
        
        :param self: Description
        :return: Description
        :rtype: DataIngestionArtifact
        '''
        try:
            # Reading data from feature store
            df = pd.read_csv(self.data_ingestion_config.feature_store_path)
            logger.info("Read data from feature store for train-test split.")
            
            # Splitting the data into training and testing sets
            logger.info(f"Transforming data into train and test sets. Test size: {self.data_ingestion_config.train_test_split_ratio}")
            train_df, test_df = train_test_split(
                df,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42
            )
            
            # Creating ingested directory if it doesn't exist
            os.makedirs(self.data_ingestion_config.ingested_dir, exist_ok=True)
            
            # Saving training and testing data to respective file paths
            train_df.to_csv(self.data_ingestion_config.training_file_path, index=False)
            test_df.to_csv(self.data_ingestion_config.testing_file_path, index=False)
            
            logger.info(f"Saved training data at {self.data_ingestion_config.training_file_path}.")
            logger.info(f"Saved testing data at {self.data_ingestion_config.testing_file_path}.")
        
        except Exception as e:
            raise CustomException(e, sys)
         
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        '''
        This function initiates the data ingestion process and returns the DataIngestionArtifact.
        
        :param self: Description
        :return: Description
        :rtype: DataIngestionArtifact
        '''
        try:
            self.export_data_to_feature_store()
            self.split_data_as_train_test()
            
            # Creating DataIngestionArtifact
            data_ingestion_artifact = DataIngestionArtifact(
                feature_store_path=self.data_ingestion_config.feature_store_path,
                training_file_path=self.data_ingestion_config.training_file_path,
                testing_file_path=self.data_ingestion_config.testing_file_path
            )
            
            logger.info(f"Data Ingestion Artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
            
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    data_ingestion_config = DataIngestionConfig()
    data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
    data_ingestion.export_data_to_feature_store()
    data_ingestion.split_data_as_train_test()
    data_ingestion_artifact = data_ingestion.initiate_data_ingestion()