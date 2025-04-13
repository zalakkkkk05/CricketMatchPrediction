import pandas as pd
import os

# ✅ Existing Function - Keep as is
def add_player_impact_feature(df, runs_weight=0.5, wickets_weight=10):
    if "runs" not in df.columns or "wickets" not in df.columns:
        raise ValueError("DataFrame must contain 'runs' and 'wickets' columns")
    
    df["player_impact"] = df["runs"] * runs_weight + df["wickets"] * wickets_weight
    return df


# ✅ Function to Add 'win_type'
def add_win_type_feature(df):
    def get_win_type(row):
        if row['win_by_runs'] > 0:
            return 'bat_first_win'
        elif row['win_by_wickets'] > 0:
            return 'chase_win'
        else:
            return 'tie_or_no_result'  # Includes no result or tie
    df['win_type'] = df.apply(get_win_type, axis=1)
    return df


# 🆕 NEW: Team Form Feature
def add_team_form_feature(df, n=5):
    print("Calculating team form feature...")

    # Create a dictionary to hold recent match results for each team
    team_history = {}
    team_form_scores = []

    for idx, row in df.iterrows():
        team1 = row['team1']
        team2 = row['team2']
        winner = row['winner'] if pd.notna(row['winner']) else None
        
        # Initialize win history if not exists
        for team in [team1, team2]:
            if team not in team_history:
                team_history[team] = []

        # Compute win ratio for last n matches
        team1_recent = team_history[team1][-n:]
        team2_recent = team_history[team2][-n:]
        team1_form = team1_recent.count(team1) / n if len(team1_recent) >= n else 0.0
        team2_form = team2_recent.count(team2) / n if len(team2_recent) >= n else 0.0

        team_form_scores.append((team1_form, team2_form))

        # Update win history after calculating form
        if winner:
            team_history[team1].append(winner)
            team_history[team2].append(winner)

    df[['team1_form_last5', 'team2_form_last5']] = pd.DataFrame(team_form_scores, index=df.index)
    return df


# ✅ Function to Encode Categorical Columns + win_type
def encode_categorical_features():
    print("Starting Encoding Process...")

    # Load cleaned matches data
    cleaned_matches_path = os.path.join("D:/CricketMatchPrediction/Dataset", "cleaned_matches.csv")
    df = pd.read_csv(cleaned_matches_path)
    
    print("\nOriginal Data Sample:")
    print(df.head())

    # ➕ Add win_type column
    df = add_win_type_feature(df)

    # ➕ Add team form features
    df = add_team_form_feature(df, n=5)

    # ➕ Encode win_type numerically
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['win_type_encoded'] = le.fit_transform(df['win_type'])

    # Columns to encode
    categorical_cols = ['team1', 'team2', 'city', 'venue', 'toss_winner', 'toss_decision']

    print("\nEncoding Categorical Columns:", categorical_cols)

    # One-Hot Encode
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

    print("\nEncoded Data Sample:")
    print(df_encoded.head())

    print("\nEncoded Data Shape:", df_encoded.shape)

    # Save encoded file
    save_path = os.path.join("D:/CricketMatchPrediction/Dataset", "matches_encoded.csv")
    df_encoded.to_csv(save_path, index=False)
    print(f"\nEncoded Data Saved at: {save_path}")
    print("Encoding Process Completed Successfully!")


# ✅ Deliveries Encoding
def encode_deliveries_categorical_features():
    print("Starting Encoding Process for Deliveries...")

    # Load cleaned deliveries data
    cleaned_deliveries_path = os.path.join("D:/CricketMatchPrediction/Dataset", "cleaned_deliveries.csv")
    df_deliveries = pd.read_csv(cleaned_deliveries_path)
    
    print("\nOriginal Deliveries Data Sample:")
    print(df_deliveries.head())

    # Columns to encode
    categorical_cols_deliveries = ['batting_team', 'bowling_team', 'batsman', 'non_striker', 'bowler', 'player_dismissed', 'dismissal_kind', 'fielder']

    print("\nEncoding Categorical Columns in Deliveries:", categorical_cols_deliveries)

    # One-Hot Encode
    df_deliveries_encoded = pd.get_dummies(df_deliveries, columns=categorical_cols_deliveries, drop_first=False)

    print("\nEncoded Deliveries Data Sample:")
    print(df_deliveries_encoded.head())

    print("\nEncoded Deliveries Data Shape:", df_deliveries_encoded.shape)

    # Save encoded file
    save_path_deliveries = os.path.join("D:/CricketMatchPrediction/Dataset", "deliveries_encoded.csv")
    df_deliveries_encoded.to_csv(save_path_deliveries, index=False)
    print(f"\nEncoded Deliveries Data Saved at: {save_path_deliveries}")
    print("Encoding Process for Deliveries Completed Successfully!")


# 🔽 Run the encoding when script is run directly
if __name__ == "__main__":
    encode_categorical_features()
    encode_deliveries_categorical_features()
