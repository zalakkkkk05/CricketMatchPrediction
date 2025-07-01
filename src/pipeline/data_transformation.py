import sys
import os

# Add root path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.data_utils import load_and_clean_data

# Example usage
if __name__ == "__main__":
    # File paths for the raw data
    matches_filepath = os.path.join("Dataset", "matches.csv")
    deliveries_filepath = os.path.join("Dataset", "deliveries.csv")
    
    # Load the data
    matches_df, deliveries_df = load_and_clean_data(matches_filepath, deliveries_filepath)
    
    # Print the first few rows of each dataset to verify loading
    print("Matches DataFrame:")
    print(matches_df.head())
    print("\nDeliveries DataFrame:")
    print(deliveries_df.head())
