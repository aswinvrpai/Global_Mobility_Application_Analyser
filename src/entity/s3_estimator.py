from src.cloud_storage.aws_storage import SimpleStorageService
from src.exception import CustomException
from src.entity.estimator import VisaModel
import sys
from pandas import DataFrame
from src.logger.logger import setup_logger,log_file

# Logger initialization
logger = setup_logger("s3_estimator", log_file)

class S3ModelEstimator:
    """
    This class is used to save and retrieve us_visas model in s3 bucket and to do prediction
    """

    def __init__(self,bucket_name,model_path):
        """
        :param bucket_name: Name of your model bucket
        :param model_path: Location of your model in bucket
        """
        self.bucket_name: str = bucket_name
        self.s3: SimpleStorageService = SimpleStorageService()
        self.model_path: str = model_path
        self.loaded_model:VisaModel=None
        
        # Create the bucket if it does not exist
        try:
            if not self.s3.is_bucket_present(bucket_name=self.bucket_name):
                self.s3.create_bucket(bucket_name=self.bucket_name)
        except Exception as e:
            logger.error(f"Error while creating bucket: {e}")
            raise CustomException(e, sys)     

    def is_model_present(self,model_path:str)->bool:
        try:
            return self.s3.s3_key_path_available(bucket_name=self.bucket_name, s3_key=model_path)
        except CustomException as e:
            logger.error(f"Error while checking model presence: {e}")
            return False

    def load_model(self)->VisaModel:
        """
        Load the model from the model_path
        :return:
        """
        try:
            self.loaded_model = self.s3.load_model(self.model_path,bucket_name=self.bucket_name)
            return self.loaded_model
        except Exception as e:
            logger.error(f"Error while loading model: {e}")
            raise CustomException(e, sys)

    def save_model(self,from_file,remove:bool=False)->None:
        """
        Save the model to the model_path
        :param from_file: Your local system model path
        :param remove: By default it is false that mean you will have your model locally available in your system folder
        :return:
        """
        try:
            self.s3.upload_file(from_file,
                                to_filename=self.model_path,
                                bucket_name=self.bucket_name,
                                remove=remove
                            )
        except Exception as e:
            logger.error(f"Error while saving model: {e}")
            raise CustomException(e, sys)

    def predict(self,dataframe:DataFrame):
        """
        :param dataframe:
        :return:
        """
        try:
            if self.loaded_model is None:
                self.loaded_model = self.load_model()
            return self.loaded_model.predict(dataframe=dataframe)
        except Exception as e:
            raise CustomException(e, sys)