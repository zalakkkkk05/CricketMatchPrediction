import pickle
import pandas as pd
import numpy as np

# Load model + feature columns + label encoder
model = pickle.load(open("src/models/pickles/stacked_model.pkl", "rb"))
feature_cols = pickle.load(open("src/models/pickles/feature_columns.pkl", "rb"))
le = pickle.load(open("src/models/pickles/label_encoder.pkl", "rb"))

DUMMY_NUMERIC = 0.5

# Define batch of scenarios you want to predict
scenarios = [
    {
        "team1": "Mumbai Indians",
        "team2": "Royal Challengers Bangalore",
        "city": "Mumbai",
        "venue": "Wankhede Stadium",
        "toss_winner": "Mumbai Indians",
        "toss_decision": "field",
        "win_type": "chase_win",
    },
    {
        "team1": "Chennai Super Kings",
        "team2": "Delhi Capitals",
        "city": "Chennai",
        "venue": "MA Chidambaram Stadium, Chepauk",
        "toss_winner": "Chennai Super Kings",
        "toss_decision": "bat",
        "win_type": "bat_first_win",
    },
    {
        "team1": "Kolkata Knight Riders",
        "team2": "Sunrisers Hyderabad",
        "city": "Kolkata",
        "venue": "Eden Gardens",
        "toss_winner": "Sunrisers Hyderabad",
        "toss_decision": "field",
        "win_type": "chase_win",
    },
]

for idx, s in enumerate(scenarios, 1):
    print(f"\n🔎 Scenario {idx}: {s['team1']} vs {s['team2']} at {s['venue']}")
    
    input_df = pd.DataFrame(0, index=[0], columns=feature_cols)
    
    # Set dummy numeric features
    numeric_features = [
        "team1_form_last5", "team2_form_last5", "team1_head_to_head_win_ratio",
        "team1_venue_win_ratio", "team1_home_win_ratio", "team1_toss_win_match_win_ratio",
        "team1_total_impact_score", "team2_total_impact_score",
    ]
    input_df.loc[0, numeric_features] = np.float32(DUMMY_NUMERIC)

    # Features to set (one-hot)
    one_hot_features = [
        f"team1_{s['team1']}",
        f"team2_{s['team2']}",
        f"city_{s['city']}",
        f"venue_{s['venue']}",
        f"toss_winner_{s['toss_winner']}",
        f"toss_decision_{s['toss_decision']}",
        f"win_type_{s['win_type']}",
    ]

    for f in one_hot_features:
        if f in input_df.columns:
            input_df.loc[0, f] = 1
            print(f" ✅ Set {f}")
        else:
            print(f" ⚠️ WARNING: Feature {f} not found in model feature columns.")

    # Predict
    pred_idx = model.predict(input_df)[0]
    pred_winner = le.inverse_transform([pred_idx])[0]
    print(f"\n✅ Predicted Winner: {pred_winner}")
