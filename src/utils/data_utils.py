import pandas as pd
import os

def load_and_clean_data(matches_filepath, deliveries_filepath):
    # Check if files exist
    if not os.path.exists(matches_filepath) or not os.path.exists(deliveries_filepath):
        raise FileNotFoundError("One or both of the files do not exist")

    try:
        # Load the data separately
        matches_df = pd.read_csv(matches_filepath)
        deliveries_df = pd.read_csv(deliveries_filepath)
    except Exception as e:
        raise ValueError(f"Error reading CSV: {e}")

    # Strip column names to avoid issues with extra spaces
    matches_df.columns = matches_df.columns.str.strip()
    deliveries_df.columns = deliveries_df.columns.str.strip()

    return matches_df, deliveries_df
