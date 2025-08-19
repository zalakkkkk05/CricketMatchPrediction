import argparse, itertools, pickle, random, time, requests
from tqdm import tqdm

FEATURE_COLUMNS_PATH = "src/models/pickles/feature_columns.pkl"
LABEL_ENCODER_PATH   = "src/models/pickles/label_encoder.pkl"

def load_feature_columns(path=FEATURE_COLUMNS_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_teams_from_label_encoder(path=LABEL_ENCODER_PATH):
    with open(path, "rb") as f:
        le = pickle.load(f)
    return sorted(set(map(str, le.classes_)))

def suffix_values(feature_columns, prefix):
    pref = prefix + "_"
    return {c[len(pref):] for c in feature_columns if c.startswith(pref)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:5000/predict")
    ap.add_argument("--max-tests", type=int, default=3000)
    ap.add_argument("--max-failures", type=int, default=50)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=5.0)
    args = ap.parse_args()

    feature_columns = load_feature_columns()
    teams  = load_teams_from_label_encoder()               # ✅ only real teams
    venues = sorted(suffix_values(feature_columns, "venue"))
    cities = sorted(suffix_values(feature_columns, "city"))
    toss_decisions = ["bat", "field"]

    print(f"✅ Known teams: {len(teams)} | venues: {len(venues)} | cities: {len(cities)}")

    combos = []
    for t1, t2 in itertools.permutations(teams, 2):
        for v in venues:
            for c in cities:
                for tw in (t1, t2):
                    for td in toss_decisions:
                        cols = [f"team1_{t1}", f"team2_{t2}", f"venue_{v}", f"city_{c}",
                                f"toss_winner_{tw}", f"toss_decision_{td}"]
                        if all(col in feature_columns for col in cols):
                            combos.append((t1, t2, v, c, tw, td))

    random.shuffle(combos)
    combos = combos[:args.max_tests]
    print(f"🧪 Will test {len(combos)} combinations (sampled)")

    ok = fails = 0
    failed_cases = []

    for (t1, t2, v, c, tw, td) in tqdm(combos, desc="Testing"):
        payload = {"team1": t1, "team2": t2, "venue": v, "city": c,
                   "toss_winner": tw, "toss_decision": td}
        try:
            r = requests.post(args.url, json=payload, timeout=args.timeout)
            jd = r.json()
            if r.status_code == 200 and jd.get("predicted_winner") in [t1, t2]:
                ok += 1
            else:
                fails += 1
                failed_cases.append((payload, jd))
        except Exception as e:
            fails += 1
            failed_cases.append((payload, str(e)))

        if args.sleep:
            time.sleep(args.sleep)
        if fails >= args.max_failures:
            print("⛔ Too many failures, stopping early.")
            break

    print(f"\n✅ Done. OK: {ok} | ❌ Failed: {fails}")
    if failed_cases:
        with open("failed_test_cases.txt", "w") as f:
            for p, err in failed_cases:
                f.write(f"{p} -> {err}\n")
        print("📝 Saved failed cases to failed_test_cases.txt")

if __name__ == "__main__":
    main()
