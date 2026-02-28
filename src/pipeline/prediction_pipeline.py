import os
import sys
import pandas as pd
from pandas import DataFrame

from src.logger.logger import setup_logger, log_file
from src.exception import CustomException
from src.entity.config_entity import ModelPredictorConfig
from src.entity.s3_estimator import S3ModelEstimator

# Initialize logger
logger = setup_logger("prediction_pipeline", log_file)

class ModelDataForPrediction:
     
    def __init__(self,
        continent,
        education_of_employee,
        has_job_experience,
        requires_job_training,
        no_of_employees,
        region_of_employment,
        prevailing_wage,
        unit_of_wage,
        full_time_position,
        company_age
    ) -> None:
        """
        Usvisa Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            self.continent = continent
            self.education_of_employee = education_of_employee
            self.has_job_experience = has_job_experience
            self.requires_job_training = requires_job_training
            self.no_of_employees = no_of_employees
            self.region_of_employment = region_of_employment
            self.prevailing_wage = prevailing_wage
            self.unit_of_wage = unit_of_wage
            self.full_time_position = full_time_position
            self.company_age = company_age
        except Exception as e:
            raise CustomException(e, sys) from e
    
    def get_input_data_frame(self)-> DataFrame:
        """
        This function returns a DataFrame from USvisaData class input
        """
        try:
            
            input_dict = self.get_data_as_dict()
            return DataFrame(input_dict)
        
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def get_data_as_dict(self):
        """
        This function returns a dictionary from visaData class input 
        """
        logger.info("Entered get_data_as_dict method as visaData class")

        try:
            input_data = {
                "continent": [self.continent],
                "education_of_employee": [self.education_of_employee],
                "has_job_experience": [self.has_job_experience],
                "requires_job_training": [self.requires_job_training],
                "no_of_employees": [self.no_of_employees],
                "region_of_employment": [self.region_of_employment],
                "prevailing_wage": [self.prevailing_wage],
                "unit_of_wage": [self.unit_of_wage],
                "full_time_position": [self.full_time_position],
                "company_age": [self.company_age],
            }

            logger.info("Created visa data dict")
            return input_data

        except Exception as e:
            raise CustomException(e, sys) from e
    
class ModelPredictor:
    
    def __init__(self, model_predictor_config: ModelPredictorConfig):
        self.model_predictor_config = model_predictor_config
    
    def predict(self, dataframe) -> str:
        """
        This is the method of USvisaClassifier
        Returns: Prediction in string format
        """
        try:
            logger.info("Entered predict method of USvisaClassifier class")
            model = S3ModelEstimator(
                bucket_name=self.model_predictor_config.bucket_name,
                model_path=self.model_predictor_config.s3_model_key_path,
            )
            result =  model.predict(dataframe)
            logger.info("Exited predict method of USvisaClassifier class")
            return result
        
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        model_predictor_config = ModelPredictorConfig()
        model_predictor = ModelPredictor(model_predictor_config=model_predictor_config)
        input_data = ModelDataForPrediction(
            continent="Asia",
            education_of_employee="Master's",
            has_job_experience="N",
            requires_job_training="N",
            no_of_employees=2430,
            region_of_employment="South",
            prevailing_wage=148138.59,
            unit_of_wage="Year",
            full_time_position="Y",
            company_age=40
        )
        input_df = input_data.get_input_data_frame()
        
        prediction = model_predictor.predict(input_df)
        print(f"Prediction: {prediction}")
    
    except Exception as e:
        raise CustomException(e, sys)