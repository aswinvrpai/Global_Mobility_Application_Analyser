import os
import sys

import numpy as np
import dill
import yaml
from pandas import DataFrame

from src.exception import CustomException
from src.logger.logger import setup_logger, log_file

# File specific Logger;
logger = setup_logger('main_utils', log_file)

# Read Yaml File;
def read_yaml_file(file_path: str) -> dict:
    """
    Read Yaml file data
    file_path: str location of file to read
    dict: dictionary output
    """
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        logger.info(f"Error in read yaml file")
        raise CustomException(e, sys) from e
        
    

def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    """
    Write Yaml file data
    file_path: str location of file to read
    content: dictionary output
    replace: file to replaced;
    """
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        logger.info(f"Error in write yaml file")
        raise CustomException(e, sys) from e
        


def load_object(file_path: str) -> object:
    """
    Load a binary object
    file_path: str location of file to read
    content: dictionary output
    replace: file to replaced;
    """
    logger.info("Entered the load_object method of utils")

    try:

        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)

        logger.info("Exited the load_object method of utils")

        return obj

    except Exception as e:
        logger.info("Error in load_object method of utils") 
        raise CustomException(e, sys) from e
          

def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        logger.info("Error in save_numpy_array_data method of utils")
        raise CustomException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        logger.info("Error in load_numpy_array_data method of utils")
        raise CustomException(e, sys) from e


def save_object(file_path: str, obj: object) -> None:
    logger.info("Entered the save_object method of utils")

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logger.info("Exited the save_object method of utils")

    except Exception as e:
        logger.info("Error in save_object method of utils")
        raise CustomException(e, sys) from e


def drop_columns(df: DataFrame, cols: list)-> DataFrame:

    """
    drop the columns form a pandas DataFrame
    df: pandas DataFrame
    cols: list of columns to be dropped
    """
    logger.info("Entered drop_columns methon of utils")

    try:
        df = df.drop(columns=cols, axis=1)

        logger.info("Exited the drop_columns method of utils")
        
        return df
    except Exception as e:
        logger.info("Error in drop_columns method of utils")
        raise CustomException(e, sys) from e
    
if __name__ == "__main__":
    # Example usage of the utility functions
    try:
        # Example: Save and load a numpy array
        dict_data = read_yaml_file("config/model.yaml")
        # logger.info(f"Successfully read model config: {dict_data}")
        for key, value in dict_data.items():
            logger.info(f"{key}: {value}")
    except Exception as e:
        logger.error(f"Error in main block of utils: {e}")