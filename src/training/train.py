import pandas as pd
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from sklearn.model_selection import train_test_split
from src.shared.data_loader import DataLoader
from src.shared.config import settings
import structlog
import shap
import matplotlib.pyplot as plt
import joblib

logger = structlog.get_logger()

def train_model():
    #setup MLFlow tracking
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)   
    mlflow.set_experiment("anchor-churn-v1")

    with mlflow.start_run(run_name="clean_baseline_with_shap"):
        #loading the engineered dataset
        loader=DataLoader()
        df = loader.get_full_dataset()

        #feature selection
        # We drop the 'Answer Key' (leaky columns)
        leaky_cols = [
            'churn_date', 'reason_code', 'refund_amount_usd', 
            'feedback_text', 'is_reactivation', 'churn_event_id'
        ]
        metadata_cols = ['account_id', 'account_name', 'signup_date', 'subscription_id', 'start_date', 'end_date']
        
        drop_list = leaky_cols + metadata_cols + ['churn_flag']

        X = df.drop(columns=[c for c in drop_list if c in df.columns])
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
        #--------------------------------------this is the shap bit-------------------------------------
        # SHAP explainability (fuh that black box effect)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        # save the explainer to a file
        explainer_path = "shap_explainer.pkl"
        joblib.dump(explainer, explainer_path)

        # log the explainer as an artifact so the API can find it later
        mlflow.log_artifact(explainer_path)

        # Generate a Force Plot for the first person in the test set
        # We save it as HTML so it stays interactive!
        expected_value = explainer.expected_value
        if isinstance(expected_value, list): # Handle multiclass output if necessary
            expected_value = expected_value[1]

        shap_plot = shap.force_plot(
            expected_value, 
            shap_values[0, :], 
            X_test.iloc[0, :], 
            matplotlib=False
        )

        shap.save_html("individual_explanation.html", shap_plot)
        mlflow.log_artifact("individual_explanation.html")

        # Save summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, show=False)
        plt.savefig("shap_summary.png", bbox_inches='tight')
        mlflow.log_artifact("shap_summary.png")

        #logging everything onto mlflow (so tuff)
        accuracy = model.score(X_test, y_test)
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.lightgbm.log_model(model, "churn_model")

        logger.info("training_complete", accuracy=f"{accuracy:.4f}")

if __name__ == "__main__":
    train_model()