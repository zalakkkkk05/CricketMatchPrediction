# src/features/compute_features.py

# src/features/compute_match_features.py

import os
import numpy as np
import pandas as pd
from functools import lru_cache
from typing import Dict

# -----------------------------
# Helpers: safety + clipping
# -----------------------------
def _safe_div(a, b, default=0.0):
    """Division that never throws and never returns NaN/Inf."""
    try:
        a = float(a); b = float(b)
        if b == 0.0:
            return default
        v = a / b
        if not np.isfinite(v):
            return default
        return v
    except Exception:
        return default

def _clip(v, lo=0.0, hi=1.0):
    """Clip a value into [lo, hi], returning float; safe on bad inputs."""
    try:
        return float(np.clip(float(v), lo, hi))
    except Exception:
        return lo

def _ratio(wins, total, lo=0.0, hi=1.0):
    """Wins/total ratio, safely clipped."""
    return _clip(_safe_div(wins, total, default=0.0), lo, hi)

# -----------------------------
# Data loading (cached)
# -----------------------------
@lru_cache(maxsize=1)
def _load_matches() -> pd.DataFrame:
    """
    Loads your encoded matches dataset.
    Expected at: Dataset/matches_encoded.csv (project root relative).
    Returns empty DataFrame if not found, so features degrade gracefully.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, "..", ".."))
    csv_path = os.path.join(project_root, "Dataset", "matches_encoded.csv")

    if not os.path.exists(csv_path):
        # Return empty df with minimal columns referenced below
        return pd.DataFrame(columns=[
            "team1", "team2", "winner", "venue", "city", "toss_winner", "season", "date"
        ])

    try:
        df = pd.read_csv(csv_path)
        # Normalize a few expected columns if they exist
        # (We won't enforce types strongly—just keep things flexible)
        for col in ["team1", "team2", "winner", "venue", "city", "toss_winner", "season", "date"]:
            if col in df.columns:
                df[col] = df[col].astype(str)
        return df
    except Exception:
        # Fail safe
        return pd.DataFrame(columns=[
            "team1", "team2", "winner", "venue", "city", "toss_winner", "season", "date"
        ])

# -----------------------------
# Feature calculators
# -----------------------------
def _venue_win_ratio(df: pd.DataFrame, team: str, venue: str) -> float:
    """Fraction of matches at this venue won by `team`."""
    if df.empty or "venue" not in df.columns or "winner" not in df.columns:
        return 0.0
    subset = df[df["venue"] == venue]
    total = len(subset)
    if total == 0:
        return 0.0
    wins = (subset["winner"] == team).sum()
    return _ratio(wins, total)

def _city_win_ratio(df: pd.DataFrame, team: str, city: str) -> float:
    """Fraction of matches in this city won by `team` (proxy for 'home')."""
    if df.empty or "city" not in df.columns or "winner" not in df.columns:
        return 0.0
    subset = df[df["city"] == city]
    total = len(subset)
    if total == 0:
        return 0.0
    wins = (subset["winner"] == team).sum()
    return _ratio(wins, total)

def _recent_form(df: pd.DataFrame, team: str, n: int = 5) -> float:
    """
    Win ratio over the last n matches played by `team` (in chronological order if 'date' or 'season' exists).
    """
    if df.empty or "winner" not in df.columns:
        return 0.0

    # Filter matches where the team participated
    played_mask = ((df.get("team1", "") == team) | (df.get("team2", "") == team))
    df_team = df[played_mask].copy()

    # Try to sort chronologically if possible
    if "date" in df_team.columns:
        # robust parse
        try:
            df_team["__parsed_date"] = pd.to_datetime(df_team["date"], errors="coerce")
            df_team = df_team.sort_values(["__parsed_date"]).drop(columns=["__parsed_date"])
        except Exception:
            pass
    elif "season" in df_team.columns:
        # Season-only sort as fallback
        try:
            df_team["__season_num"] = pd.to_numeric(df_team["season"], errors="coerce")
            df_team = df_team.sort_values(["__season_num"]).drop(columns=["__season_num"])
        except Exception:
            pass

    if len(df_team) == 0:
        return 0.0

    # Take last n matches
    df_recent = df_team.tail(n)
    wins = (df_recent["winner"] == team).sum()
    total = len(df_recent)
    return _ratio(wins, total)

def _h2h_win_ratio(df: pd.DataFrame, team_a: str, team_b: str) -> float:
    """Head-to-head win ratio for team_a vs team_b across all seasons."""
    if df.empty or "winner" not in df.columns or "team1" not in df.columns or "team2" not in df.columns:
        return 0.0

    h2h = df[((df["team1"] == team_a) & (df["team2"] == team_b)) |
             ((df["team1"] == team_b) & (df["team2"] == team_a))]

    total = len(h2h)
    if total == 0:
        return 0.0
    wins = (h2h["winner"] == team_a).sum()
    return _ratio(wins, total)

def _toss_to_win_ratio(df: pd.DataFrame, team: str) -> float:
    """
    Historical fraction of matches where team won the toss AND eventually won the match.
    (A crude measure of 'toss helps this team'.)
    """
    if df.empty or "toss_winner" not in df.columns or "winner" not in df.columns:
        return 0.0
    tw = df[df["toss_winner"] == team]
    total = len(tw)
    if total == 0:
        return 0.0
    wins = (tw["winner"] == team).sum()
    return _ratio(wins, total)

# -----------------------------
# Public API
# -----------------------------
def compute_match_features(team1: str, team2: str, venue: str, toss_winner: str, city: str = None) -> Dict[str, float]:
    """
    Compute numeric features for a (team1, team2, venue, toss_winner [, city]) configuration.
    Returns a dict of floats. Any keys not present in your model's feature_columns will be ignored upstream.
    """
    df = _load_matches()

    # If city not provided, try to infer the most common city for this venue
    if city is None:
        if not df.empty and "venue" in df.columns and "city" in df.columns:
            venue_rows = df[df["venue"] == venue]
            if len(venue_rows) > 0:
                # mode() may return multiple—take the first
                try:
                    city = venue_rows["city"].mode().iloc[0]
                except Exception:
                    city = None

    # Core ratios (all clipped to [0,1])
    venue_win_ratio_team1 = _venue_win_ratio(df, team1, venue)
    venue_win_ratio_team2 = _venue_win_ratio(df, team2, venue)

    city_win_ratio_team1 = _city_win_ratio(df, team1, city) if city else 0.0
    city_win_ratio_team2 = _city_win_ratio(df, team2, city) if city else 0.0

    recent_form_team1 = _recent_form(df, team1, n=5)
    recent_form_team2 = _recent_form(df, team2, n=5)

    h2h_team1 = _h2h_win_ratio(df, team1, team2)
    h2h_team2 = _h2h_win_ratio(df, team2, team1)

    toss_win_help_team1 = _toss_to_win_ratio(df, team1)
    toss_win_help_team2 = _toss_to_win_ratio(df, team2)

    # Binary flags (0/1)
    toss_winner_is_team1 = 1.0 if toss_winner == team1 else 0.0
    toss_winner_is_team2 = 1.0 if toss_winner == team2 else 0.0

    # Optional: simple home/neutral proxy
    # If both city ratios are ~0 (no data), call it neutral.
    home_adv_team1 = 1.0 if city_win_ratio_team1 > city_win_ratio_team2 else 0.0
    home_adv_team2 = 1.0 if city_win_ratio_team2 > city_win_ratio_team1 else 0.0
    neutral_venue = 1.0 if (city is None or (city_win_ratio_team1 == 0.0 and city_win_ratio_team2 == 0.0)) else 0.0

    # Build output dict (floats only)
    features = {
        # Venue performance
        "venue_win_ratio_team1": _clip(venue_win_ratio_team1, 0.0, 1.0),
        "venue_win_ratio_team2": _clip(venue_win_ratio_team2, 0.0, 1.0),

        # City/home proxy
        "city_win_ratio_team1": _clip(city_win_ratio_team1, 0.0, 1.0),
        "city_win_ratio_team2": _clip(city_win_ratio_team2, 0.0, 1.0),
        "home_adv_team1": _clip(home_adv_team1, 0.0, 1.0),
        "home_adv_team2": _clip(home_adv_team2, 0.0, 1.0),
        "neutral_venue": _clip(neutral_venue, 0.0, 1.0),

        # Recent form
        "recent_form_team1": _clip(recent_form_team1, 0.0, 1.0),
        "recent_form_team2": _clip(recent_form_team2, 0.0, 1.0),

        # Head-to-head ratios
        "h2h_win_ratio_team1": _clip(h2h_team1, 0.0, 1.0),
        "h2h_win_ratio_team2": _clip(h2h_team2, 0.0, 1.0),

        # Toss related
        "toss_winner_is_team1": _clip(toss_winner_is_team1, 0.0, 1.0),
        "toss_winner_is_team2": _clip(toss_winner_is_team2, 0.0, 1.0),
        "toss_win_help_team1": _clip(toss_win_help_team1, 0.0, 1.0),
        "toss_win_help_team2": _clip(toss_win_help_team2, 0.0, 1.0),
    }

    # Ensure no NaN/Inf gets returned (extra guard)
    for k, v in list(features.items()):
        if not np.isfinite(v):
            features[k] = 0.0

    return features

