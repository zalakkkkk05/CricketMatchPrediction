# src/data/cricbuzz_client.py
import os
import json
import time
from typing import Any, Dict, List, Optional, Tuple

import requests


# -------------------------
# basic helpers / settings
# -------------------------
RAPI_DEBUG = os.getenv("RAPI_DEBUG", "0") == "1"
TIMEOUT = int(os.getenv("RAPI_TIMEOUT", "15"))

def _log(msg: str) -> None:
    if RAPI_DEBUG:
        print(msg, flush=True)

def _get_json(url: str, headers: Optional[Dict[str, str]] = None) -> Any:
    t0 = time.time()
    r = requests.get(url, headers=headers or {}, timeout=TIMEOUT)
    r.raise_for_status()
    try:
        j = r.json()
    except Exception:
        # some mirrors return text; try to salvage
        txt = r.text.strip()
        try:
            j = json.loads(txt)
        except Exception:
            raise
    dt = (time.time() - t0) * 1000.0
    _log(f"[RAPI] GET {url} -> {r.status_code} in {dt:.1f}ms")
    return j


# -------------------------
# providers catalog
# -------------------------
# you can choose primary by CRICBUZZ_PROVIDER, and allow fallbacks with ENABLE_FALLBACKS=1
PRIMARY = os.getenv("CRICBUZZ_PROVIDER", "cricketdata")

PROVIDERS: Dict[str, Dict[str, Any]] = {
    # 1) CricketData.org (recommended / compliant)
    "cricketdata": {
        "base": os.getenv("CRICKETDATA_BASE", "https://api.cricapi.com/v1"),
        "headers": {"Accept": "application/json"},
        "paths": {
            # lists
            "matches_live":     ["/currentMatches?apikey={apikey}&offset=0"],
            "matches_recent":   ["/matches?apikey={apikey}&offset=0"],
            "matches_upcoming": ["/matches?apikey={apikey}&offset=0"],
            # scorecard-like
            "match_info":       ["/match_info?apikey={apikey}&id={match_id}"],
        },
    },

    # 2) RapidAPI (unofficial) — keep as optional fallback if you have a key/plan
    "rapidapi": {
        "base": "https://" + os.getenv("RAPIDAPI_HOST", "unofficial-cricbuzz.p.rapidapi.com"),
        "headers": {
            "x-rapidapi-host": os.getenv("RAPIDAPI_HOST", "unofficial-cricbuzz.p.rapidapi.com"),
            "x-rapidapi-key": os.getenv("RAPIDAPI_KEY", ""),
        },
        "paths": {
            "matches_live": [
                os.getenv("RAPI_PATH_MATCHES_LIVE", "/matches/live"),
                "/matches",                    # older aggregator
                "/matches/v1/live",            # other variant we've seen in the wild
            ],
            "matches_recent": [
                os.getenv("RAPI_PATH_MATCHES_RECENT", "/matches/recent"),
                "/matches",
                "/matches/v1/recent",
            ],
            "matches_upcoming": [
                os.getenv("RAPI_PATH_MATCHES_UPCOMING", "/matches/upcoming"),
                "/matches",
                "/matches/v1/upcoming",
            ],
            "match_info": [
                os.getenv("RAPI_PATH_MATCH_INFO", "/mcenter/v1/{match_id}/scard"),
                os.getenv("RAPI_PATH_MATCH_INFO_ALT", "/mcenter/{match_id}/scard"),
            ],
        },
    },

    # 3) public mirror (thin data; sometimes empty)
    "vercel2": {
        "base": "https://cricket-api.vercel.app",
        "headers": {"Accept": "application/json"},
        "paths": {
            "matches_live":     ["/matches/live"],
            "matches_recent":   ["/matches/recent"],
            "matches_upcoming": ["/matches/upcoming"],
            "match_info":       ["/scorecard/{match_id}"],
        },
    },

    # 4) another mirror (paywalled frequently)
    "vercel1": {
        "base": "https://cricbuzz-live.vercel.app",
        "headers": {"Accept": "application/json"},
        "paths": {
            "matches_live":     ["/matches/live"],
            "matches_recent":   ["/matches/recent"],
            "matches_upcoming": ["/matches/upcoming"],
            "match_info":       ["/scorecard/{match_id}"],
        },
    },
}


# -------------------------
# provider chain / formatting
# -------------------------
def _provider_chain() -> List[str]:
    chain = [PRIMARY]
    if os.getenv("ENABLE_FALLBACKS", "1") == "1":
        for k in ["rapidapi", "vercel1", "vercel2"]:
            if k not in chain:
                chain.append(k)
    return chain

def _fmt_for_provider(provider_name: str, fmt: Dict[str, Any]) -> Dict[str, Any]:
    """inject api key etc when needed"""
    fmt = dict(fmt or {})
    if provider_name == "cricketdata":
        fmt.setdefault("apikey", os.getenv("CRICKETDATA_KEY", ""))
    return fmt


# -------------------------
# low-level attempt runner
# -------------------------
def _try_key_attempts(provider_name: str, key: str, **fmt) -> Dict[str, Any]:
    p = PROVIDERS[provider_name]
    base, headers = p["base"], p["headers"]
    paths = p["paths"].get(key, [])
    if not paths:
        return {"provider": provider_name, "winner_url": None, "data": None, "attempts": []}

    fmt = _fmt_for_provider(provider_name, fmt)
    attempts = []
    for path in paths:
        url = base + path.format(**fmt)
        try:
            data = _get_json(url, headers)
            attempts.append({"url": url, "ok": True, "error": None})
            return {"provider": provider_name, "winner_url": url, "data": data, "attempts": attempts}
        except Exception as e:
            attempts.append({"url": url, "ok": False, "error": str(e)})
            _log(f"[{provider_name}] fail {url}: {e}")
            continue

    return {"provider": provider_name, "winner_url": None, "data": None, "attempts": attempts}

def _try_key(key: str, **fmt) -> Optional[Dict[str, Any]]:
    for name in _provider_chain():
        pack = _try_key_attempts(name, key, **fmt)
        if pack.get("winner_url") and pack.get("data") is not None:
            _log(f"[{name}] winner {key} top-level: {type(pack['data']).__name__}")
            return pack
    return None


# -------------------------
# shapes → normalized rows
# -------------------------
def _extract_match_row_czb(series_row: Dict[str, Any]) -> Dict[str, Any]:
    """Normalizes a Cricbuzz-like 'seriesMatches.matches[i]' item."""
    info = series_row.get("matchInfo") or {}
    score = series_row.get("matchScore") or {}
    t1 = (info.get("team1") or {}).get("teamName") or (info.get("team1") or {}).get("name")
    t2 = (info.get("team2") or {}).get("teamName") or (info.get("team2") or {}).get("name")
    vinfo = info.get("venueInfo") or {}
    venue = vinfo.get("ground", "")
    city = vinfo.get("city", "")

    t1s = score.get("team1Score") or {}
    t2s = score.get("team2Score") or {}
    t1i = (t1s.get("inngs1") or {}) or (t1s.get("inngs2") or {})
    t2i = (t2s.get("inngs1") or {}) or (t2s.get("inngs2") or {})

    return {
        "matchId": info.get("matchId"),
        "matchFormat": info.get("matchFormat"),
        "matchDesc": info.get("matchDesc"),
        "seriesId": info.get("seriesId"),
        "seriesName": info.get("seriesName"),
        "state": info.get("state"),
        "status": info.get("status"),
        "team1": t1, "team2": t2,
        "venue": venue, "city": city,
        "startDate": info.get("startDate"),
        "endDate": info.get("endDate"),
        "t1_runs": t1i.get("runs"), "t1_wkts": t1i.get("wickets"), "t1_overs": t1i.get("overs"),
        "t2_runs": t2i.get("runs"), "t2_wkts": t2i.get("wickets"), "t2_overs": t2i.get("overs"),
    }

def _extract_match_row_cricketdata(m: Dict[str, Any]) -> Dict[str, Any]:
    """Normalizes a CricketData match object."""
    # common fields seen on currentMatches/matches
    teams = m.get("teams") or []
    team1 = teams[0] if len(teams) > 0 else m.get("teamInfo", [{}])[0].get("name") if isinstance(m.get("teamInfo"), list) and m["teamInfo"] else None
    team2 = teams[1] if len(teams) > 1 else m.get("teamInfo", [{}, {}])[1].get("name") if isinstance(m.get("teamInfo"), list) and len(m.get("teamInfo", [])) > 1 else None

    # scores are typically: [{"r": runs, "w": wickets, "o": overs, "inning": "Team 1 Inning 1"}, ...]
    t1_runs = t1_wkts = t1_overs = None
    t2_runs = t2_wkts = t2_overs = None
    if isinstance(m.get("score"), list):
        scr = m["score"]
        if len(scr) > 0 and isinstance(scr[0], dict):
            t1_runs = scr[0].get("r"); t1_wkts = scr[0].get("w"); t1_overs = scr[0].get("o")
        if len(scr) > 1 and isinstance(scr[1], dict):
            t2_runs = scr[1].get("r"); t2_wkts = scr[1].get("w"); t2_overs = scr[1].get("o")

    return {
        "matchId": m.get("id") or m.get("matchId"),
        "matchFormat": m.get("matchType") or m.get("type"),
        "matchDesc": m.get("name"),
        "seriesId": m.get("series_id") or m.get("seriesId"),
        "seriesName": m.get("series"),
        "state": m.get("status"),           # often text like "Match over" / "Live"
        "status": m.get("status"),
        "team1": team1, "team2": team2,
        "venue": m.get("venue") or "",
        "city": "",                         # not always provided
        "startDate": m.get("date"),
        "endDate": m.get("dateTimeGMT"),
        "t1_runs": t1_runs, "t1_wkts": t1_wkts, "t1_overs": t1_overs,
        "t2_runs": t2_runs, "t2_wkts": t2_wkts, "t2_overs": t2_overs,
    }

def _flatten_matches(bundle: Any) -> List[Dict[str, Any]]:
    """
    Accepts different provider shapes and returns a flat list of normalized rows.
    """
    out: List[Dict[str, Any]] = []
    if not bundle:
        return out

    # CricketData shapes: {"status":"success","data":[...]} or just {"data":[...]}
    if isinstance(bundle, dict) and "data" in bundle and isinstance(bundle["data"], list):
        for m in bundle["data"]:
            if isinstance(m, dict):
                out.append(_extract_match_row_cricketdata(m))
        return out

    # Some mirrors just give a list of matches directly
    if isinstance(bundle, list):
        for m in bundle:
            if isinstance(m, dict):
                # try generic fields
                if "teams" in m or "teamInfo" in m:
                    out.append(_extract_match_row_cricketdata(m))
                else:
                    # last resort: return item back
                    out.append(m)
        return out

    # Cricbuzz-like "typeMatches" tree
    if isinstance(bundle, dict) and "typeMatches" in bundle:
        tms = bundle.get("typeMatches") or []
        for tm in tms:
            for wrapper in (tm.get("seriesMatches") or []):
                series = wrapper.get("seriesMatches") or {}
                for m in (series.get("matches") or []):
                    out.append(_extract_match_row_czb(m))
        return out

    # Cricbuzz-like "seriesMatches" at top
    if isinstance(bundle, dict) and "seriesMatches" in bundle:
        for wrapper in (bundle.get("seriesMatches") or []):
            series = wrapper.get("seriesMatches") or {}
            for m in (series.get("matches") or []):
                out.append(_extract_match_row_czb(m))
        return out

    # unknown: return empty but log keys
    if isinstance(bundle, dict):
        _log(f"[flatten] unknown keys: {list(bundle.keys())[:10]}")
    return out


# -------------------------
# public API (used by app.py)
# -------------------------
def list_matches(kind: str) -> List[Dict[str, Any]]:
    """
    kind in {"live","recent","upcoming"}
    """
    key_map = {
        "live": "matches_live",
        "recent": "matches_recent",
        "upcoming": "matches_upcoming",
    }
    key = key_map.get(kind, "matches_recent")

    # try providers in chain
    pack = _try_key(key)
    rows = _flatten_matches(pack["data"]) if pack else []

    # optional: fallback to sample file if absolutely nothing
    if not rows and os.getenv("USE_SAMPLE_WHEN_EMPTY", "0") == "1":
        try:
            sample_path = os.path.join(os.path.dirname(__file__), "_sample_recent.json")
            with open(sample_path, "r") as f:
                sample = json.load(f)
            rows = _flatten_matches(sample)
            for r in rows:
                r["_note"] = "sample_fallback"
        except Exception as e:
            _log(f"[sample] failed reading _sample_recent.json: {e}")

    # optional: live falls back to recent so UI shows something
    if kind == "live" and not rows and os.getenv("FALLBACK_RECENT_WHEN_LIVE_EMPTY", "0") == "1":
        pack2 = _try_key("matches_recent")
        rows = _flatten_matches(pack2["data"]) if pack2 else []
        for r in rows:
            r["_note"] = "fallback_recent"

    return rows


def get_match_scorecard(match_id: str | int) -> Dict[str, Any]:
    """
    Returns a simple dict the model/UI needs: team1, team2, venue, city, toss_winner, raw
    """
    match_id = str(match_id)

    def _extract_from_cricketdata(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # payload can be:
        #   {"status":"success","data":[{...}], "info": {...}}
        #   {"status":"success","data":{...},  "info": {...}}
        # or on error: {"status":"failure","reason":"..."}
        if not isinstance(payload, dict):
            return None
        if payload.get("status") == "failure":
            return None

        data = payload.get("data")
        md = None
        if isinstance(data, list) and data:
            md = data[0]
        elif isinstance(data, dict):
            md = data
        else:
            # some responses embed under 'info' or put minimal fields in 'data' and details in 'info'
            md = payload.get("info")

        if not isinstance(md, dict):
            return None

        # Try multiple shapes seen in CricketData
        # Teams may appear as:
        #  - "teams": ["Team A", "Team B"]
        #  - "teamInfo": [{"name": "Team A"}, {"name": "Team B"}]
        teams = md.get("teams")
        if not (isinstance(teams, list) and teams):
            ti = md.get("teamInfo")
            if isinstance(ti, list) and ti:
                teams = [x.get("name") for x in ti if isinstance(x, dict)]
            else:
                teams = []

        team1 = teams[0] if len(teams) > 0 else ""
        team2 = teams[1] if len(teams) > 1 else ""

        # Venue/city may be split or combined
        venue = md.get("venue") or md.get("venueInfo") or ""
        if isinstance(venue, dict):
            city = venue.get("city", "") or ""
            venue = venue.get("name", "") or venue.get("ground", "") or ""
        else:
            city = md.get("city") or ""

        # Toss fields vary a bit
        toss_text = (
            md.get("tossWinner")
            or md.get("tossChoice")
            or md.get("toss")
            or ""
        )

        return {
            "team1": team1 or "",
            "team2": team2 or "",
            "venue": venue or "",
            "city": city or "",
            "toss_winner": toss_text or "",
            "raw": payload,
        }

    for name in _provider_chain():
        pack = _try_key_attempts(name, "match_info", match_id=match_id)
        if not pack.get("winner_url"):
            continue

        raw = pack["data"]

        if name == "cricketdata":
            parsed = _extract_from_cricketdata(raw if isinstance(raw, dict) else {})
            if parsed and any([parsed["team1"], parsed["team2"], parsed["venue"]]):
                parsed["_provider"] = name
                parsed["_src_url"] = pack["winner_url"]
                return parsed
            _log(f"[cricketdata] unexpected match_info shape for {match_id}: keys={list(raw.keys()) if isinstance(raw, dict) else type(raw)}")

        else:
            # cricbuzz-like mirrors
            bundle = raw.get("raw", raw) if isinstance(raw, dict) else {}
            team1 = ((bundle.get("team1") or {}).get("teamName") or (bundle.get("team1") or {}).get("name") or "")
            team2 = ((bundle.get("team2") or {}).get("teamName") or (bundle.get("team2") or {}).get("name") or "")
            vinfo = (bundle.get("venueInfo") or {})
            venue = vinfo.get("ground", "")
            city = vinfo.get("city", "")
            toss_text = (bundle.get("toss") or "")
            if any([team1, team2, venue, city, toss_text]):
                return {
                    "team1": team1,
                    "team2": team2,
                    "venue": venue,
                    "city": city,
                    "toss_winner": toss_text,
                    "raw": bundle,
                    "_provider": name,
                    "_src_url": pack["winner_url"],
                }

        _log(f"[{name}] could not extract normalized scorecard for {match_id}")

    raise RuntimeError(f"Could not fetch scorecard for match_id={match_id} from any provider")



# -------------------------
# diag / debug endpoints support
# -------------------------
def diagnose_matches(kind: str) -> Dict[str, Any]:
    key_map = {"live": "matches_live", "recent": "matches_recent", "upcoming": "matches_upcoming"}
    key = key_map.get(kind, "matches_recent")
    attempts_all = []
    winner: Optional[Tuple[str, str]] = None
    data = None

    for name in _provider_chain():
        pack = _try_key_attempts(name, key)
        attempts_all.append({"provider": name, "attempts": pack["attempts"]})
        if pack.get("winner_url"):
            if winner is None:
                winner = (name, pack["winner_url"])
                data = pack["data"]
                break

    rows = _flatten_matches(data) if data is not None else []
    return {
        "input_kind": kind,
        "winner": {"provider": winner[0], "url": winner[1]} if winner else None,
        "attempts": attempts_all,
        "match_count": len(rows),
        "sample": rows[:2],
        "assumption_no_matches": len(rows) == 0,
    }

def diagnose_scorecard(match_id: str | int) -> Dict[str, Any]:
    out = {"match_id": str(match_id), "attempts": [], "sample_keys": None}
    for name in _provider_chain():
        pack = _try_key_attempts(name, "match_info", match_id=str(match_id))
        out["attempts"].append({"provider": name, "attempts": pack["attempts"]})
        if pack.get("winner_url") and pack.get("data") is not None:
            raw = pack["data"]
            if isinstance(raw, dict):
                out["sample_keys"] = list(raw.keys())[:20]
            out["winner"] = {"provider": name, "url": pack["winner_url"]}
            break
    return out


# -------------------------
# legacy passthrough for /debug_fetch
# -------------------------
def _rapid_get(path: str) -> Any:
    """
    Hit the 'current provider base' + given path.
    This is used by /debug_fetch in app.py
    """
    base = PROVIDERS[PRIMARY]["base"]
    headers = PROVIDERS[PRIMARY]["headers"]
    # interpolate possible {match_id} or {apikey}? not needed for this debug path
    url = base + path
    return _get_json(url, headers)
