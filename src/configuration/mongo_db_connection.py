from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from src.logger.logger import setup_logger, log_file
from src.exception import CustomException
import sys

from src.constants import MONGO_DB_URL, DATABASE_NAME, COLLECTION_NAME

# Initialize Logger;
logging = setup_logger(__name__, log_file)

class MongoDbClient:
    def __init__(self,database_name=DATABASE_NAME):
        
        try:
            if MONGO_DB_URL is None:
                logging.error("MONGO_DB_URL not set in env")
                raise Exception("MONGO_DB_URL not set in env")
            
            # Create a new client and connect to the server
            self.client = MongoClient(MONGO_DB_URL, server_api=ServerApi('1'))
            self.database = self.client[database_name]
            self.collection = self.database[COLLECTION_NAME]
            
            logging.info("MongoDb Client connection is successful.")
        
        except Exception as e:
            raise CustomException(e,sys)
        
