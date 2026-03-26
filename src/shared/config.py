from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    #this automatically finds the root of the project
    #__file__ is the path to this script
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    
    PROJECT_NAME: str="Anchor"
    ENV: str="development"

    #these are the data layer paths
    RAW_DATA_PATH: Path = BASE_DIR / "data" / "raw"
    PROCESSED_DATA_PATH: Path = BASE_DIR / "data" / "processed"

    #ml and tracking settings
    MLFLOW_TRACKING_URI: str="http://127.0.0.1:5050"
    MODEL_NAME: str="churn-prediction-model"
    CHURD_THRESHOLD: float=0.5

    #cloud deployment details we'll specify later

    #tells pydantic to look for a .env file
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

# basically instantiating this so other modules can import this as a singleton
settings = Settings()