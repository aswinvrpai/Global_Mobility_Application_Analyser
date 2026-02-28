from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from uvicorn import run as app_run
import uvicorn

from src.constants import APP_HOST, APP_PORT
from src.pipeline.prediction_pipeline import ModelPredictor, ModelDataForPrediction
from src.entity.config_entity import ModelPredictorConfig
from src.exception import CustomException
from src.logger.logger import setup_logger, log_file

# Logger initialization
logger = setup_logger("main", log_file)

from pydantic import BaseModel

# FastAPI application setup
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Pydantic schema just for parsing the incoming JSON
class PredictRequest(BaseModel):
    continent: str
    education_of_employee: str
    has_job_experience: str
    requires_job_training: str
    no_of_employees: int
    region_of_employment: str
    prevailing_wage: float
    unit_of_wage: str
    full_time_position: str
    company_age: int


# Routes and logic for the FastAPI application
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        # 1. Build your custom class from the parsed request
        input_data = ModelDataForPrediction(
            continent=request.continent,
            education_of_employee=request.education_of_employee,
            has_job_experience=request.has_job_experience,
            requires_job_training=request.requires_job_training,
            no_of_employees=request.no_of_employees,
            region_of_employment=request.region_of_employment,
            prevailing_wage=request.prevailing_wage,
            unit_of_wage=request.unit_of_wage,
            full_time_position=request.full_time_position,
            company_age=request.company_age
        )

        # 2. Convert to DataFrame
        input_df = input_data.get_input_data_frame()

        # 3. Predict
        model_predictor = ModelPredictor(ModelPredictorConfig())
        prediction = model_predictor.predict(input_df)

        result = prediction[0]
        logger.info(f"Prediction result: {result}")
        if hasattr(result, "item"):
            result = result.item()
            result = "Approved" if result == 1 else "Denied"

        return {"prediction": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# Run the application
if __name__ == "__main__":
    uvicorn.run("main:app", host=APP_HOST, port=int(APP_PORT), reload=True)