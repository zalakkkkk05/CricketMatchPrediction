import os
import sys
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from src.features.compute_features import compute_match_features
except ModuleNotFoundError:
    from features.compute_features import compute_match_features

from src.data.cricbuzz_client import (
    list_matches,
    get_match_scorecard,
    _rapid_get,
    diagnose_matches,
    diagnose_scorecard,
)

PICKLE_DIR = os.path.join(SRC_DIR, "models", "pickles")
MODEL_PATH = os.path.join(PICKLE_DIR, "stacked_ensemble.pkl")
FEATURE_COLUMNS_PATH = os.path.join(PICKLE_DIR, "feature_columns.pkl")
LABEL_ENCODER_PATH = os.path.join(PICKLE_DIR, "label_encoder.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(FEATURE_COLUMNS_PATH, "rb") as f:
    feature_columns = pickle.load(f)
with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

app = Flask(__name__)
CORS(app)
REQUESTS_SERVED = 0

def _extract_values(prefix: str):
    pref = prefix + "_"
    return sorted({c[len(pref):] for c in feature_columns if c.startswith(pref)})

def _sanitize_input_df(df: pd.DataFrame) -> pd.DataFrame:
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.fillna(0.0).clip(lower=-1e6, upper=1e6)
    if not np.issubdtype(df.values.dtype, np.floating):
        df = df.astype(float)
    return df

@app.route("/get_metadata", methods=["GET"])
def get_metadata():
    teams = sorted(set(map(str, getattr(label_encoder, "classes_", []))))
    venues = _extract_values("venue")
    cities = _extract_values("city")
    toss_decisions = ["bat", "field"]
    return jsonify({"teams": teams, "venues": venues, "cities": cities, "toss_decisions": toss_decisions})

@app.route("/live_matches", methods=["GET"])
def live_matches():
    try:
        data = list_matches("live")
        return jsonify({"matches": data, "no_matches_now": len(data) == 0})
    except Exception as e:
        return jsonify({"matches": [], "error": str(e)}), 200

@app.route("/recent_matches", methods=["GET"])
def recent_matches():
    try:
        data = list_matches("recent")
        return jsonify({"matches": data, "empty_recent": len(data) == 0})
    except Exception as e:
        return jsonify({"matches": [], "error": str(e)}), 200

@app.route("/upcoming_matches", methods=["GET"])
def upcoming_matches():
    try:
        data = list_matches("upcoming")
        return jsonify({"matches": data, "empty_upcoming": len(data) == 0})
    except Exception as e:
        return jsonify({"matches": [], "error": str(e)}), 200

@app.route("/debug_provider", methods=["GET"])
def debug_provider():
    env = {
        "provider": os.getenv("CRICBUZZ_PROVIDER", "rapidapi"),
        "RAPIDAPI_HOST": os.getenv("RAPIDAPI_HOST"),
        "RAPI_PATH_MATCHES_LIVE": os.getenv("RAPI_PATH_MATCHES_LIVE"),
        "RAPI_PATH_MATCHES_UPCOMING": os.getenv("RAPI_PATH_MATCHES_UPCOMING"),
        "RAPI_PATH_MATCHES_RECENT": os.getenv("RAPI_PATH_MATCHES_RECENT"),
        "RAPI_PATH_MATCH_INFO": os.getenv("RAPI_PATH_MATCH_INFO"),
        "RAPI_PATH_MATCH_INFO_ALT": os.getenv("RAPI_PATH_MATCH_INFO_ALT"),
        "has_key": bool(os.getenv("RAPIDAPI_KEY")),
        "key_prefix": (os.getenv("RAPIDAPI_KEY") or "")[:6],
        "ENABLE_FALLBACKS": os.getenv("ENABLE_FALLBACKS"),
        "RAPI_DEBUG": os.getenv("RAPI_DEBUG"),
    }
    return jsonify(env)

@app.route("/match_info", methods=["GET"])
def match_info():
    match_id = request.args.get("match_id")
    if not match_id:
        return jsonify({"error": "missing match_id"}), 400
    try:
        info = get_match_scorecard(match_id)
        return jsonify(info)
    except Exception as e:
        return jsonify({"error": f"Failed to fetch match data: {e}"}), 502

@app.route("/predict_live", methods=["GET"])
def predict_live():
    global REQUESTS_SERVED
    REQUESTS_SERVED += 1
    match_id = request.args.get("match_id")
    if not match_id:
        return jsonify({"error": "missing match_id"}), 400
    try:
        mi = get_match_scorecard(match_id)
    except Exception as e:
        return jsonify({"error": f"Failed to fetch match data: {e}"}), 502
    required = ["team1", "team2", "venue", "city", "toss_winner"]
    missing = [k for k in required if not mi.get(k)]
    if missing:
        return jsonify({"error": f"Missing fields from data source: {missing}", "raw_available": bool(mi.get("raw"))}), 502
    team1 = mi["team1"]; team2 = mi["team2"]
    venue = mi["venue"]; city = mi["city"]
    toss_winner = mi["toss_winner"]
    toss_decision = "bat"
    try:
        real_features = compute_match_features(team1, team2, venue, toss_winner, city=city)
    except Exception:
        real_features = {}
    input_df = pd.DataFrame(0.0, index=[0], columns=feature_columns)
    for col, val in real_features.items():
        if col in input_df.columns:
            try:
                input_df.at[0, col] = float(val)
            except Exception:
                input_df.at[0, col] = 0.0
    for prefix, value in {
        "team1": team1, "team2": team2, "city": city, "venue": venue,
        "toss_winner": toss_winner, "toss_decision": toss_decision,
    }.items():
        colname = f"{prefix}_{value}"
        if colname in input_df.columns:
            input_df.at[0, colname] = 1.0
    input_df = _sanitize_input_df(input_df)
    try:
        raw_pred = model.predict(input_df)[0]
        predicted_team = label_encoder.inverse_transform([raw_pred])[0]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    if predicted_team not in [team1, team2]:
        try:
            proba = model.predict_proba(input_df)[0]
            class_labels = label_encoder.inverse_transform(np.arange(len(proba)))
            team_probas = dict(zip(class_labels, proba))
            predicted_team = max({team1: team_probas.get(team1, 0.0), team2: team_probas.get(team2, 0.0)},
                                 key=lambda k: team_probas.get(k, 0.0))
        except Exception:
            pass
    return jsonify({
        "predicted_winner": predicted_team,
        "inputs": {"team1": team1, "team2": team2, "venue": venue, "city": city, "toss_winner": toss_winner},
        "requests_served": REQUESTS_SERVED
    })

@app.route("/debug_fetch", methods=["GET"])
def debug_fetch():
    path = request.args.get("path")
    if not path or not path.startswith("/"):
        return jsonify({"error": "pass ?path=/matches/recent"}), 400
    try:
        j = _rapid_get(path)
        return jsonify({"ok": True, "path": path, "sample": j})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "path": path}), 200

@app.route("/debug_scorecard", methods=["GET"])
def debug_scorecard():
    mid = request.args.get("id")
    if not mid:
        return jsonify({"error": "pass ?id=<matchId>"}), 400
    try:
        j = get_match_scorecard(mid)
        return jsonify(j)
    except Exception as e:
        return jsonify({"error": str(e)}), 200

@app.route("/diag/matches", methods=["GET"])
def diag_matches_route():
    kind = request.args.get("kind", "recent")
    try:
        report = diagnose_matches(kind)
        return jsonify(report)
    except Exception as e:
        return jsonify({"error": str(e)}), 200

@app.route("/diag/scorecard", methods=["GET"])
def diag_scorecard_route():
    mid = request.args.get("id")
    if not mid:
        return jsonify({"error": "pass ?id=<matchId>"}), 400
    try:
        report = diagnose_scorecard(mid)
        return jsonify(report)
    except Exception as e:
        return jsonify({"error": str(e)}), 200

@app.route("/debug_fetch_url2", methods=["GET"])
def debug_fetch_url2():
    import requests
    url = request.args.get("url")
    if not url or not (url.startswith("http://") or url.startswith("https://")):
        return jsonify({"error": "pass ?url=https://..."}), 400
    try:
        r = requests.get(url, timeout=15)
        status = r.status_code
        try:
            j = r.json()
            summary = {}
            if isinstance(j, dict):
                for k, v in j.items():
                    if isinstance(v, list):
                        summary[k] = f"<list len {len(v)}>"
                    elif isinstance(v, dict):
                        summary[k] = f"<dict keys {len(v)}>"
                    else:
                        summary[k] = type(v).__name__
            return jsonify({
                "status": status,
                "url": url,
                "top_level_type": type(j).__name__,
                "top_level_keys": list(j.keys()) if isinstance(j, dict) else None,
                "summary": summary,
                "raw": j
            })
        except Exception:
            return jsonify({"status": status, "url": url, "text": r.text[:2000]})
    except Exception as e:
        return jsonify({"error": str(e), "url": url}), 200

if __name__ == "__main__":
    app.run(debug=True)
