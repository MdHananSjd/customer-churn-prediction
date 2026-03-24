import pandas as pd
import structlog 
from src.shared.config import settings

logger = structlog.get_logger()

class DataLoader:
    def __init__(self):
        #setting the raw data path from the imported config settings
        self.raw_data_path = settings.RAW_DATA_PATH

    def load_raw_csv(self, file_name:str) -> pd.DataFrame:
        #loads a ravenstack csv right from the raw directory
        file_path = self.raw_data_path / f"{file_name}.csv"

        if not file_path.exists():
            logger.error("File not found", path=str(file_path))
            raise FileNotFoundError(f"Required file {file_name} not found in {self.raw_data_path}")
        
        try:
            df = pd.read_csv(file_path)
            logger.info("File loaded successfully", file=file_name, rows=len(df), columns=list(df.columns))
            return df
        except Exception as e:
            logger.error("Failed to load file", file=file_name, error=str(e))
            raise

    def get_full_dataset(self) -> pd.DataFrame:
        """
        Joins accounts, subscriptions, and churn events into a single 
        refined dataframe with exactly one row per customer.
        """
        # load the tables
        accounts = self.load_raw_csv('ravenstack_accounts')
        subs = self.load_raw_csv('ravenstack_subscriptions')
        events = self.load_raw_csv('ravenstack_churn_events')

        # Deduplicate subscriptions
        # We sort by 'start_date' so the most recent subscription record 
        # for each account is at the bottom, then we keep only that one.
        subs_latest = subs.sort_values("start_date").drop_duplicates("account_id", keep="last")

        # Deduplicate events
        # Similarly, we keep only the most recent churn event per account.
        events_latest = events.sort_values("churn_date").drop_duplicates("account_id", keep="last")

        # Perform the Joins (1-to-1 Mapping)
        # Join subscriptions to accounts
        df = pd.merge(
            accounts, 
            subs_latest, 
            on="account_id", 
            how="left", 
            suffixes=('', '_sub')  # Keep account columns clean
        )

        # Join churn events
        df = pd.merge(df, events_latest, on="account_id", how="left")

        # 5. Final Target Cleanup
        # Ensure the churn_flag is a clean integer (0 or 1)
        df["churn_flag"] = df["churn_flag"].fillna(0).astype(int)

        # 6. Drop redundant columns to keep the feature set clean
        cols_to_drop = ["churn_flag_sub"]
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

        logger.info("master_dataset_refined", 
                    final_rows=len(df), 
                    total_churners=df["churn_flag"].sum())
        
        return df
    
if __name__ == "__main__":
    #to test the scripts directly
    loader = DataLoader()
    test_df = loader.get_full_dataset()

    print("\n--- Master Dataset Summary ---")
    print(f"Total rows: {len(test_df)}")
    print(f"Churn Rate: {test_df['churn_flag'].mean():.2%}")
    print(f"Columns: {test_df.columns.tolist()}")
    print(test_df.head())
