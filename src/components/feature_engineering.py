import pandas as pd
import os

# ✅ Win type: bat-first or chase
def add_win_type_feature(df):
    def get_win_type(row):
        if row['win_by_runs'] > 0:
            return 'bat_first_win'
        elif row['win_by_wickets'] > 0:
            return 'chase_win'
        else:
            return 'tie_or_no_result'
    df['win_type'] = df.apply(get_win_type, axis=1)
    return df

# ✅ Team form feature: win % in last n matches
def add_team_form_feature(df, n=5):
    print("Calculating team form feature...")
    team_history = {}
    team_form_scores = []
    for idx, row in df.iterrows():
        team1 = row['team1']
        team2 = row['team2']
        winner = row['winner'] if pd.notna(row['winner']) else None
        for team in [team1, team2]:
            if team not in team_history:
                team_history[team] = []
        team1_recent = team_history[team1][-n:]
        team2_recent = team_history[team2][-n:]
        team1_form = team1_recent.count(team1) / n if len(team1_recent) >= n else 0.0
        team2_form = team2_recent.count(team2) / n if len(team2_recent) >= n else 0.0
        team_form_scores.append((team1_form, team2_form))
        if winner:
            team_history[team1].append(winner)
            team_history[team2].append(winner)
    df[['team1_form_last5', 'team2_form_last5']] = pd.DataFrame(team_form_scores, index=df.index)
    return df

# ✅ Head-to-head win ratio
def add_head_to_head_feature(df):
    print("Calculating head-to-head win ratio...")
    head_to_head = {}
    head_to_head_ratios = []
    for idx, row in df.iterrows():
        team1 = row['team1']
        team2 = row['team2']
        winner = row['winner'] if pd.notna(row['winner']) else None
        matchup_key = tuple(sorted([team1, team2]))
        if matchup_key not in head_to_head:
            head_to_head[matchup_key] = []
        matchup_history = head_to_head[matchup_key]
        total_matches = len(matchup_history)
        team1_wins = matchup_history.count(team1)
        win_ratio = team1_wins / total_matches if total_matches > 0 else 0.0
        head_to_head_ratios.append(win_ratio)
        if winner:
            head_to_head[matchup_key].append(winner)
    df['team1_head_to_head_win_ratio'] = head_to_head_ratios
    return df

# ✅ Venue win ratio
def add_venue_win_ratio_feature(df):
    print("Calculating venue win ratio for team1...")
    venue_history = {}
    venue_ratios = []
    for idx, row in df.iterrows():
        venue = row['venue']
        team1 = row['team1']
        winner = row['winner'] if pd.notna(row['winner']) else None
        if venue not in venue_history:
            venue_history[venue] = []
        history = venue_history[venue]
        total_played = len(history)
        team1_wins = history.count(team1)
        win_ratio = team1_wins / total_played if total_played > 0 else 0.0
        venue_ratios.append(win_ratio)
        if winner:
            venue_history[venue].append(winner)
    df['team1_venue_win_ratio'] = venue_ratios
    return df

# ✅ Home ground win ratio
def add_home_win_ratio_feature(df):
    print("Calculating home win ratio for team1...")
    home_grounds = {
        'Mumbai Indians': ['Wankhede Stadium'],
        'Chennai Super Kings': ['MA Chidambaram Stadium'],
        'Royal Challengers Bangalore': ['M Chinnaswamy Stadium'],
        'Kolkata Knight Riders': ['Eden Gardens'],
        'Sunrisers Hyderabad': ['Rajiv Gandhi International Stadium, Uppal'],
        'Delhi Daredevils': ['Feroz Shah Kotla'],
        'Delhi Capitals': ['Arun Jaitley Stadium'],
        'Kings XI Punjab': ['Punjab Cricket Association Stadium, Mohali'],
        'Rajasthan Royals': ['Sawai Mansingh Stadium'],
        'Gujarat Lions': ['Saurashtra Cricket Association Stadium'],
        'Rising Pune Supergiant': ['Maharashtra Cricket Association Stadium'],
        'Lucknow Super Giants': ['Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium'],
        'Gujarat Titans': ['Narendra Modi Stadium']
    }
    home_history = {}
    home_ratios = []
    for idx, row in df.iterrows():
        team1 = row['team1']
        venue = row['venue']
        winner = row['winner'] if pd.notna(row['winner']) else None
        is_home = venue in home_grounds.get(team1, [])
        if team1 not in home_history:
            home_history[team1] = []
        history = home_history[team1]
        total_home_games = len(history)
        home_wins = history.count(team1)
        win_ratio = home_wins / total_home_games if total_home_games > 0 else 0.0
        home_ratios.append(win_ratio)
        if is_home and winner:
            home_history[team1].append(winner)
    df['team1_home_win_ratio'] = home_ratios
    return df

# ✅ Toss impact feature
def add_toss_win_match_win_ratio(df):
    print("Calculating toss impact (toss win to match win ratio for team1)...")
    toss_win_history = {}
    toss_win_ratios = []
    for idx, row in df.iterrows():
        team1 = row['team1']
        toss_winner = row['toss_winner']
        match_winner = row['winner'] if pd.notna(row['winner']) else None
        if team1 not in toss_win_history:
            toss_win_history[team1] = []
        toss_won = (team1 == toss_winner)
        toss_win_record = toss_win_history[team1]
        total_toss_won = len(toss_win_record)
        matches_won_after_toss = toss_win_record.count(True)
        win_ratio = matches_won_after_toss / total_toss_won if total_toss_won > 0 else 0.0
        toss_win_ratios.append(win_ratio)
        if toss_won and match_winner:
            toss_win_history[team1].append(team1 == match_winner)
    df['team1_toss_win_match_win_ratio'] = toss_win_ratios
    return df

# ✅ Player impact from deliveries data
def add_team_player_impact_scores(matches_df, deliveries_path, runs_weight=0.5, wickets_weight=10):
    print("Calculating team player impact scores from deliveries data...")
    deliveries_df = pd.read_csv(deliveries_path)
    deliveries_df['is_wicket'] = deliveries_df['dismissal_kind'].notna().astype(int)
    deliveries_df['impact_score'] = (
        deliveries_df['batsman_runs'] * runs_weight +
        deliveries_df['is_wicket'] * wickets_weight
    )
    team_impact = deliveries_df.groupby(['match_id', 'batting_team'])['impact_score'].sum().reset_index()
    team_impact.rename(columns={'batting_team': 'team', 'impact_score': 'team_total_impact_score'}, inplace=True)
    matches_df = matches_df.merge(team_impact, how='left', left_on=['id', 'team1'], right_on=['match_id', 'team'])
    matches_df.rename(columns={'team_total_impact_score': 'team1_total_impact_score'}, inplace=True)
    matches_df.drop(columns=['match_id', 'team'], inplace=True)
    matches_df = matches_df.merge(team_impact, how='left', left_on=['id', 'team2'], right_on=['match_id', 'team'])
    matches_df.rename(columns={'team_total_impact_score': 'team2_total_impact_score'}, inplace=True)
    matches_df.drop(columns=['match_id', 'team'], inplace=True)
    return matches_df

# ✅ Main pipeline
def encode_categorical_features():
    print("Starting Encoding Process...")
    cleaned_matches_path = os.path.join("D:/CricketMatchPrediction/Dataset", "cleaned_matches.csv")
    deliveries_path = os.path.join("D:/CricketMatchPrediction/Dataset", "cleaned_deliveries.csv")
    df = pd.read_csv(cleaned_matches_path)
    df = add_win_type_feature(df)
    df = add_team_form_feature(df)
    df = add_head_to_head_feature(df)
    df = add_venue_win_ratio_feature(df)
    df = add_home_win_ratio_feature(df)
    df = add_toss_win_match_win_ratio(df)
    df = add_team_player_impact_scores(df, deliveries_path)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['win_type_encoded'] = le.fit_transform(df['win_type'])
    categorical_cols = ['team1', 'team2', 'city', 'venue', 'toss_winner', 'toss_decision']
    print("\nEncoding Categorical Columns:", categorical_cols)
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    save_path = os.path.join("D:/CricketMatchPrediction/Dataset", "matches_encoded.csv")
    df_encoded.to_csv(save_path, index=False)
    print(f"\nEncoded Data Saved at: {save_path}")
    print("Encoding Process Completed Successfully!")

if __name__ == "__main__":
    encode_categorical_features()