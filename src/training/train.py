import pandas as pd
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from sklearn.model_selection import train_test_split
from src.shared.data_loader import DataLoader
from src.shared.config import settings
import structlog

logger = structlog.get_logger()

def train_model():
    #setup MLFlow tracking
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)   
    mlflow.set_experiment("anchor-churn-v1")

    with mlflow.start_run(run_name="initial_lightgbm_run"):
        #loading the engineered dataset
        loader=DataLoader()
        df = loader.get_full_dataset()

        #Feature selection
        features_to_drop = [
            'account_id', 'account_name', 'signup_date', 'subscription_id', 'start_date', 'end_date', 'churn_date', 'feedback_text', 'churn_flag'
        ]

        X = df.drop(columns=[c for c in features_to_drop if c in df.columns])
        y = df['churn_flag'] #target values

        #handling categorical data
        #we're convertung object type to category for lightgbm's native support (this is one of the reasons why we chose lightgbm over xgboost btw)
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].astype('category')

        #train test split (standard 80-20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        #defining and training model
        params ={
            "n_estimators":100,
            "learning_rate": 0.05,
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1
        }

        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)

        #logging everything onto mlflow (so tuff)
        accuracy = model.score(X_test, y_test)
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.lightgbm.log_model(model, "churn_model")

        logger.info("training_complete", accuracy=f"{accuracy:.4f}")

if __name__ == "__main__":
    train_model()