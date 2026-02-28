import boto3
from io import StringIO
from typing import Union,List
import os,sys
from src.logger.logger import setup_logger, log_file
from src.exception import CustomException
from mypy_boto3_s3.service_resource import Bucket
from botocore.exceptions import ClientError
from pandas import DataFrame,read_csv
import pickle
from src.configuration.aws_connection import S3Client
from src.constants import REGION_NAME

# Set up logging;
logger = setup_logger("aws_storage", log_file)

class SimpleStorageService:

    def __init__(self):
        s3_client = S3Client()
        self.s3_resource = s3_client.s3_resource
        self.s3_client = s3_client.s3_client
        
    def create_bucket(self, bucket_name: str, region_name: str = REGION_NAME) -> Bucket:
        """
        Method Name :   create_bucket
        Description :   This method creates a new bucket with the given bucket_name

        Output      :   Bucket object is returned based on the bucket name
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        logger.info(f"Entered the create_bucket method of S3Operations class for bucket: {bucket_name}, region: {region_name}")

        try:
            if region_name == 'us-east-1':
                self.s3_client.create_bucket(Bucket=bucket_name)
            else:
                self.s3_client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': region_name}
                )
            logger.info("Exited the create_bucket method of S3Operations class")
            return self.get_bucket(bucket_name)
        except Exception as e:
            raise CustomException(e, sys) from e
    
    def is_bucket_present(self, bucket_name: str) -> bool:
        """
        Method Name :   is_bucket_present
        Description :   This method checks if the bucket_name bucket is present in s3 or not

        Output      :   True if bucket is present, else False
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        logger.info(f"Entered the is_bucket_present method of S3Operations class for bucket: {bucket_name}")

        try:
            buckets = self.s3_client.list_buckets()
            bucket_names = [bucket['Name'] for bucket in buckets['Buckets']]
            is_present = bucket_name in bucket_names
            logger.info(f"Exited the is_bucket_present method of S3Operations class with result: {is_present}")
            return is_present
        except Exception as e:
            raise CustomException(e, sys) from e

    def get_bucket(self, bucket_name: str) -> Bucket:
        """
        Method Name :   get_bucket
        Description :   This method gets the bucket object based on the bucket_name

        Output      :   Bucket object is returned based on the bucket name
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        logger.info("Entered the get_bucket method of S3Operations class")

        try:
            bucket = self.s3_resource.Bucket(bucket_name)
            logger.info("Exited the get_bucket method of S3Operations class")
            return bucket
        except Exception as e:
            raise CustomException(e, sys) from e
    
    def s3_key_path_available(self,bucket_name,s3_key)->bool:
        try:
            bucket = self.get_bucket(bucket_name)
            file_objects = [file_object for file_object in bucket.objects.filter(Prefix=s3_key)]
            if len(file_objects) > 0:
                return True
            else:
                return False
        except Exception as e:
            raise CustomException(e,sys)
        
    @staticmethod
    def read_object(object_name: str, decode: bool = True, make_readable: bool = False) -> Union[StringIO, str]:
        """
        Method Name :   read_object
        Description :   This method reads the object_name object with kwargs

        Output      :   The column name is renamed
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        logger.info("Entered the read_object method of S3Operations class")

        try:
            func = (
                lambda: object_name.get()["Body"].read().decode()
                if decode is True
                else object_name.get()["Body"].read()
            )
            conv_func = lambda: StringIO(func()) if make_readable is True else func()
            logger.info("Exited the read_object method of S3Operations class")
            return conv_func()

        except Exception as e:
            raise CustomException(e, sys) from e

    def get_file_object( self, filename: str, bucket_name: str) -> Union[List[object], object]:
        """
        Method Name :   get_file_object
        Description :   This method gets the file object from bucket_name bucket based on filename

        Output      :   list of objects or object is returned based on filename
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        logger.info("Entered the get_file_object method of S3Operations class")

        try:
            bucket = self.get_bucket(bucket_name)

            file_objects = [file_object for file_object in bucket.objects.filter(Prefix=filename)]

            func = lambda x: x[0] if len(x) == 1 else x

            file_objs = func(file_objects)
            logger.info("Exited the get_file_object method of S3Operations class")

            return file_objs

        except Exception as e:
            raise CustomException(e, sys) from e

    def load_model(self, model_name: str, bucket_name: str, model_dir: str = None) -> object:
        """
        Method Name :   load_model
        Description :   This method loads the model_name model from bucket_name bucket with kwargs

        Output      :   list of objects or object is returned based on filename
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        logger.info("Entered the load_model method of S3Operations class")

        try:
            func = (
                lambda: model_name
                if model_dir is None
                else model_dir + "/" + model_name
            )
            model_file = func()
            file_object = self.get_file_object(model_file, bucket_name)
            model_obj = self.read_object(file_object, decode=False)
            model = pickle.loads(model_obj)
            logger.info("Exited the load_model method of S3Operations class")
            return model

        except Exception as e:
            raise CustomException(e, sys) from e

    def create_folder(self, folder_name: str, bucket_name: str) -> None:
        """
        Method Name :   create_folder
        Description :   This method creates a folder_name folder in bucket_name bucket

        Output      :   Folder is created in s3 bucket
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        logger.info("Entered the create_folder method of S3Operations class")

        try:
            self.s3_resource.Object(bucket_name, folder_name).load()

        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                folder_obj = folder_name + "/"
                self.s3_client.put_object(Bucket=bucket_name, Key=folder_obj)
            else:
                pass
            logger.info("Exited the create_folder method of S3Operations class")

    def upload_file(self, from_filename: str, to_filename: str,  bucket_name: str,  remove: bool = True):
        """
        Method Name :   upload_file
        Description :   This method uploads the from_filename file to bucket_name bucket with to_filename as bucket filename

        Output      :   Folder is created in s3 bucket
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        logger.info("Entered the upload_file method of S3Operations class")

        try:
            logger.info(
                f"Uploading {from_filename} file to {to_filename} file in {bucket_name} bucket"
            )

            self.s3_resource.meta.client.upload_file(
                from_filename, bucket_name, to_filename
            )

            logger.info(
                f"Uploaded {from_filename} file to {to_filename} file in {bucket_name} bucket"
            )

            if remove is True:
                os.remove(from_filename)

                logger.info(f"Remove is set to {remove}, deleted the file")

            else:
                logger.info(f"Remove is set to {remove}, not deleted the file")

            logger.info("Exited the upload_file method of S3Operations class")

        except Exception as e:
            raise CustomException(e, sys) from e

    def upload_df_as_csv(self,data_frame: DataFrame,local_filename: str, bucket_filename: str,bucket_name: str,) -> None:
        """
        Method Name :   upload_df_as_csv
        Description :   This method uploads the dataframe to bucket_filename csv file in bucket_name bucket

        Output      :   Folder is created in s3 bucket
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        logger.info("Entered the upload_df_as_csv method of S3Operations class")

        try:
            data_frame.to_csv(local_filename, index=None, header=True)

            self.upload_file(local_filename, bucket_filename, bucket_name)

            logger.info("Exited the upload_df_as_csv method of S3Operations class")

        except Exception as e:
            raise CustomException(e, sys) from e

    def get_df_from_object(self, object_: object) -> DataFrame:
        """
        Method Name :   get_df_from_object
        Description :   This method gets the dataframe from the object_name object

        Output      :   Folder is created in s3 bucket
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        logger.info("Entered the get_df_from_object method of S3Operations class")

        try:
            content = self.read_object(object_, make_readable=True)
            df = read_csv(content, na_values="na")
            logger.info("Exited the get_df_from_object method of S3Operations class")
            return df
        except Exception as e:
            raise CustomException(e, sys) from e

    def read_csv(self, filename: str, bucket_name: str) -> DataFrame:
        """
        Method Name :   get_df_from_object
        Description :   This method gets the dataframe from the object_name object

        Output      :   Folder is created in s3 bucket
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        logger.info("Entered the read_csv method of S3Operations class")

        try:
            csv_obj = self.get_file_object(filename, bucket_name)
            df = self.get_df_from_object(csv_obj)
            logger.info("Exited the read_csv method of S3Operations class")
            return df
        except Exception as e:
            raise CustomException(e, sys) from e

# Test the connection       
if __name__ == "__main__":
    s3 = SimpleStorageService()
    
    # Create a new bucket;
    bucket_name = s3.create_bucket("csv-us-visa-data", region_name=REGION_NAME)
    
    # Upload a CSV file to the bucket;
    local_file_path = r"C:\Work_Directory\Learn\DS_Projects\Global_Mobility_Application_Analyser\artifacts\02_14_2026__18_27_57\data_ingestion\feature_store\us_visa_data.csv"  # Path to your local CSV file
    bucket_file_name = "csv-us-visa-data"  # Desired name in the bucket
    s3.upload_file(
        from_filename=local_file_path,
        to_filename=bucket_file_name,
        bucket_name=bucket_name.name
    )
    
    s3_client = s3.s3_client
    # List all buckets
    response = s3_client.list_buckets()

    # Extract and print bucket names
    buckets = response['Buckets']
    print("S3 Buckets in your account:")
    for i, bucket in enumerate(buckets):
        print(f" {i+1}: {bucket['Name']}")