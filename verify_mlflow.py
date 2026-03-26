import mlflow
from src.shared.config import settings
import requests

def verify_mlflow():
    uri = settings.MLFLOW_TRACKING_URI
    print(f"Testing connection to MLflow at: {uri}")
    
    mlflow.set_tracking_uri(uri)
    
    try:
        # Attempt to list experiments to check connectivity
        experiments = mlflow.search_experiments()
        print(f"Successfully connected! Found {len(experiments)} experiments.")
        for exp in experiments:
            print(f" - {exp.name} (ID: {exp.experiment_id})")
            
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Could not connect to {uri}. Is the MLflow server running?")
        print("Try starting it with: mlflow server --port 5050")
        
    except Exception as e:
        if "403" in str(e):
            print("\n" + "!"*60)
            print("ERROR 403 Forbidden detected!")
            print("This is likely caused by macOS AirPlay Receiver occupying port 5000/5050.")
            print("ACTION REQUIRED: Go to System Settings -> General -> AirDrop & Handoff")
            print("and turn OFF 'AirPlay Receiver'.")
            print("!"*60 + "\n")
        else:
            print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    verify_mlflow()
