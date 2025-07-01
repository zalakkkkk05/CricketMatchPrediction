import pandas as pd
import numpy as np

# Assuming matches_df and deliveries_df are already loaded, you can add them here if not loaded.
# matches_df = pd.read_csv('path_to_matches.csv')
# deliveries_df = pd.read_csv('path_to_deliveries.csv')

def clean_matches_data(matches_df):
    print("\nCleaning Matches Data...")

    # Fill missing 'city' based on 'venue' mode (smart fill)
    venue_city_map = matches_df.groupby('venue')['city'].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
    )
    matches_df['city'] = matches_df.apply(
        lambda row: venue_city_map[row['venue']] if pd.isnull(row['city']) else row['city'], axis=1
    )

    # Fill missing values for categorical data in matches
    matches_df['city'] = matches_df['city'].fillna(matches_df['city'].mode()[0])
    matches_df['winner'] = matches_df['winner'].fillna('No Result')
    matches_df['player_of_match'] = matches_df['player_of_match'].fillna('No Award')
    matches_df['umpire1'] = matches_df['umpire1'].fillna('Unavailable')
    matches_df['umpire2'] = matches_df['umpire2'].fillna('Unavailable')
    matches_df['umpire3'] = matches_df['umpire3'].fillna('Unavailable')

    print("Matches Data Cleaning Completed!")
    return matches_df


def clean_deliveries_data(deliveries_df):
    print("\nCleaning Deliveries Data...")

    # Fill missing values for 'player_dismissed', 'dismissal_kind', and 'fielder'
    deliveries_df.fillna({
        'player_dismissed': 'None',
        'dismissal_kind': 'None',
        'fielder': 'None'
    }, inplace=True)

    # Check for missing values after filling
    print("\nMissing Values in Deliveries Dataset (After Filling):")
    print(deliveries_df.isnull().sum())

    print("Deliveries Data Cleaning Completed!")
    return deliveries_df


# Main Function to run the cleaning process
def run_data_cleaning():
    print("Starting Data Cleaning Process...\n")

    # If the data is not already loaded, load the datasets here.
    # For testing purposes, make sure to replace these paths with the correct paths.
    matches_df = pd.read_csv(r'D:\CricketMatchPrediction\Dataset\matches.csv')
    deliveries_df = pd.read_csv(r'D:\CricketMatchPrediction\Dataset\deliveries.csv')

    print("Matches Data Sample Before Cleaning:")
    print(matches_df.head())
    
    print("\nDeliveries Data Sample Before Cleaning:")
    print(deliveries_df.head())

    # Check missing values before cleaning
    print("\nMissing Values in Matches Dataset (Before Cleaning):")
    print(matches_df.isnull().sum())

    print("\nMissing Values in Deliveries Dataset (Before Cleaning):")
    print(deliveries_df.isnull().sum())

    # Clean the data
    matches_df = clean_matches_data(matches_df)
    deliveries_df = clean_deliveries_data(deliveries_df)

    # Check for missing values after cleaning
    print("\nMissing Values in Matches Dataset (After Cleaning):")
    print(matches_df.isnull().sum())

    print("\nMissing Values in Deliveries Dataset (After Cleaning):")
    print(deliveries_df.isnull().sum())

    # Optionally, save the cleaned datasets to new files
    matches_df.to_csv(r'D:\CricketMatchPrediction\Dataset\cleaned_matches.csv', index=False)
    deliveries_df.to_csv(r'D:\CricketMatchPrediction\Dataset\cleaned_deliveries.csv', index=False)

    print("\nData Cleaning Process Completed Successfully!")

# Run the data cleaning function
if __name__ == "__main__":
    run_data_cleaning()
