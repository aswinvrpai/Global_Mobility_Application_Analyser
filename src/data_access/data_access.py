import os
import sys
from typing import Optional
import pandas as pd
import numpy as np

from src.configuration.mongo_db_connection import MongoDbClient
from src.exception import CustomException
from src.constants import DATABASE_NAME, COLLECTION_NAME
from src.exception import CustomException
from src.logger.logger import setup_logger, log_file


# Initialize logger
logger = setup_logger("data_access", log_file)

class DataAccess:
    '''
    DataAccess Class for handling data operations with MongoDB.
    '''
    def __init__(self):
        '''
        Initializes the DataAccess object by creating a MongoDB client connection.
        :param self: Description
        '''

        try:
            # Mongodb Client;
            self.db_client = MongoDbClient(DATABASE_NAME)
            
            logger.info("Data access via Mongodb Client Successful.")
        except Exception as e:
            raise CustomException(e,sys)
        
    def export_data_from_db(self,collection_name:str,database_name:Optional[str]=None) ->pd.DataFrame:
        '''
        This function exports data from MongoDB collection as a pandas DataFrame.
        
        :param self: Description
        :param collection_name: Description
        :type collection_name: str
        :param database_name: Description
        :type database_name: Optional[str]
        :return: Description
        :rtype: DataFrame
        '''
        
        try:
            if database_name is None:
                collection = self.db_client.collection
            else:
                database = self.db_client.database
                collection = database[collection_name]
            
            df = pd.DataFrame(collection.find())
            if "_id" in df.columns.to_list():
                df = df.drop(columns="_id",axis=1)
            df.replace({"na":np.nan}, inplace=True)
            return df
        
        except Exception as e:
            raise CustomException(e,sys)
    
if __name__ == "__main__":
    data_access = DataAccess()
    df=data_access.export_data_from_db(COLLECTION_NAME)
    print(df.head())
    