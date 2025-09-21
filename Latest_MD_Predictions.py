# Latest_MD_Predictions.py
# ---------------------------------------------------------------------
# Premier League — Latest Matchday Predictions & Season Projection
# Streamlit dashboard
#
# - Loads historical & auxiliary CSVs from repo and data_cache/
# - Builds a large feature set (Elo, forms, H2H, rest, tm/transfers/managers, priors)
# - Trains RandomForest with time-ordered CV + isotonic calibration
# - Predicts next matchday fixture outcomes + season projection to MD38
# - No files are written; everything displays live
# ---------------------------------------------------------------------

from __future__ import annotations

# ========== Standard lib
import os
import io
import re
import math
import time
import textwrap
import warnings
import calendar
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict

# ========== Third-party
import numpy as np
import pandas as pd

# ML
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, log_loss, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

# Viz / App
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)

# =========================
# --------- THEME ---------
# =========================
PRIMARY = "#2563eb"        # Tailwind indigo-600-ish
ACCENT  = "#06b6d4"        # cyan-500
GOOD    = "#16a34a"
BAD     = "#dc2626"
MUTED   = "#64748b"
DRAW    = "#9333ea"

# =========================
# ---- REPO DISCOVERY -----
# =========================

@st.cache_data(show_spinner=False)
def repo_root() -> Path:
    # Assume this file lives at repo root; else fallback to CWD
    here = Path(__file__).resolve().parent
    return here

@st.cache_data(show_spinner=False)
def list_csvs() -> Dict[str, Path]:
    """
    Discover CSVs used by the app; return a mapping of short keys to Paths.
    The app never shows local paths in the UI; they are only used to read files.
    """
    root = repo_root()
    dc   = root / "data_cache"
    poss = [
        ("fixtures", root / "fixtures_2025_26.csv"),
        ("current_table", root / "current_table.csv"),
        ("odds_out", root / "odds_out.csv"),
        ("pl_players", root / "pl_player_stats_players.csv"),
        ("pl_teams", root / "pl_player_stats_team.csv"),
        ("pl_clubs", root / "pl_club_stats.csv"),
        ("pl_managers", root / "pl_managers_2025_26.csv"),
        ("tm_mv_clubs", root / "tm_market_values_clubs.csv"),
        ("tm_mv_players", root / "tm_market_values_players.csv"),
        ("tm_jahrestabelle", root / "tm_jahrestabelle.csv"),
        ("transfers", root / "transfers.csv"),
    ]
    # Add data_cache seasons
    if dc.exists():
        for p in sorted(dc.glob("E0_*.csv")):
            poss.append((p.stem, p))
    out = {}
    for k, p in poss:
        if isinstance(p, str):
            p = Path(p)
        if p.exists():
            out[k] = p
    return out

CSV_MAP = list_csvs()

# =========================
# ----- TEAM NORMALIZE ----
# =========================

TEAM_ALIASES: dict[str, str] = {
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Man Utd": "Manchester United",
    "Nott'm Forest": "Nottingham Forest",
    "Wolves": "Wolverhampton Wanderers",
    "West Brom": "West Bromwich Albion",
    "Tottenham": "Tottenham Hotspur",
    "Spurs": "Tottenham Hotspur",
    "Leeds": "Leeds United",
    "Newcastle": "Newcastle United",
    "Brighton": "Brighton and Hove Albion",
    "Bournemouth": "AFC Bournemouth",
    "West Ham": "West Ham United",
    "Sheffield Utd": "Sheffield United",
}

def canon_team(name: str) -> str:
    if pd.isna(name): return name
    s = str(name).strip()
    return TEAM_ALIASES.get(s, s)

# =========================
# --------- HELPERS -------
# =========================

def safe_read_csv(path: Path, **kw) -> pd.DataFrame:
    try:
        return pd.read_csv(path, **kw)
    except Exception:
        # sometimes encoded; try fallback
        try:
            return pd.read_csv(path, encoding="latin-1", **kw)
        except Exception:
            return pd.DataFrame()

def to_date(s) -> pd.Timestamp:
    return pd.to_datetime(s, errors="coerce")

def season_code_to_int(sc: str) -> int:
    m = re.search(r"E0_(\d{4})", sc)
    if m: return int(m.group(1))
    try:
        return int(sc)
    except Exception:
        return 0

def add_badge_by_position(pos: int) -> str:
    if pos <= 5:   return "UCL"
    if pos <= 7:   return "UEL"
    if pos == 8:   return "UECL"
    if pos >= 18:  return "Relegation"
    return ""

def odds_to_prob(x):
    """
    Accepts decimal odds or implied-prob strings; returns float prob in [0,1] or NaN.
    """
    if pd.isna(x): return np.nan
    s = str(x).strip()
    try:
        v = float(s)
        if v > 1.0:  # decimal odds -> implied prob
            return 1.0 / v
        # already in [0,1]
        if 0 <= v <= 1:
            return v
        return np.nan
    except Exception:
        # try % or like '0.43'
        s = s.replace("%","")
        try:
            v = float(s)
            if v > 1: v = v/100.0
            return v
        except Exception:
            return np.nan

# =========================
# ----- DATA LOADING ------
# =========================

@st.cache_data(show_spinner=False)
def load_history() -> pd.DataFrame:
    """
    Load *all* historical E0_*.csv as one frame.
    Required columns: Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR
    """
    rows = []
    for key, p in CSV_MAP.items():
        if key.startswith("E0_"):
            df = safe_read_csv(p)
            # normalize
            needed = {"Date","HomeTeam","AwayTeam","FTHG","FTAG"}
            if not needed <= set(df.columns):
                continue
            df = df.copy()
            df["Date"] = to_date(df["Date"])
            df["Season"] = season_code_to_int(key)
            df["HomeTeam"] = df["HomeTeam"].map(canon_team)
            df["AwayTeam"] = df["AwayTeam"].map(canon_team)
            if "FTR" not in df.columns:
                # infer
                df["FTR"] = np.where(df["FTHG"] > df["FTAG"], "H",
                              np.where(df["FTHG"] < df["FTAG"], "A", "D"))
            rows.append(df[["Season","Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR"]])
    if not rows: return pd.DataFrame(columns=["Season","Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR"])
    hist = pd.concat(rows, ignore_index=True).sort_values("Date").reset_index(drop=True)
    return hist

@st.cache_data(show_spinner=False)
def load_fixtures() -> pd.DataFrame:
    """
    Load fixtures_2025_26.csv if present.
    Expected columns: date, home, away, md, FTR?, FTHG?, FTAG?, played?
    """
    p = CSV_MAP.get("fixtures")
    if not p: return pd.DataFrame(columns=["date","home","away","md","played","FTR","FTHG","FTAG"])
    fx = safe_read_csv(p)
    if fx.empty:
        return pd.DataFrame(columns=["date","home","away","md","played","FTR","FTHG","FTAG"])
    # normalize
    fx = fx.rename(columns={"Date":"date","HomeTeam":"home","AwayTeam":"away"})
    for c in ["home","away"]:
        fx[c] = fx[c].map(canon_team)
    if "date" in fx.columns:
        fx["date"] = to_date(fx["date"])
    if "played" not in fx.columns:
        fx["played"] = fx.get("FTR").notna()
    if "md" not in fx.columns:
        # derive by ranking date blocks; safe fallback
        fx = fx.sort_values("date").reset_index(drop=True)
        fx["md"] = fx.groupby(["date"]).ngroup() + 1
    return fx[["date","home","away","md","played","FTR","FTHG","FTAG"]]

@st.cache_data(show_spinner=False)
def load_current_table() -> pd.DataFrame:
    p = CSV_MAP.get("current_table")
    if not p: 
        return pd.DataFrame(columns=["Pos","Team","Pl","W","D","L","GF","GA","GD","Pts"])
    tab = safe_read_csv(p)
    # let "Pos" be present; else try to derive
    if tab.empty:
        return pd.DataFrame(columns=["Pos","Team","Pl","W","D","L","GF","GA","GD","Pts"])
    # normalize
    ren = {"Pld":"Pl","Pld.":"Pl","Played":"Pl","GoalDiff":"GD","TeamName":"Team"}
    tab = tab.rename(columns={k:v for k,v in ren.items() if k in tab.columns})
    for c in ["Team"]:
        if c in tab.columns:
            tab[c] = tab[c].map(canon_team)
    if "GD" not in tab.columns and {"GF","GA"} <= set(tab.columns):
        tab["GD"] = tab["GF"] - tab["GA"]
    if "Pts" not in tab.columns and {"W","D","L"} <= set(tab.columns):
        tab["Pts"] = 3*tab["W"] + tab["D"]
    if "Pos" not in tab.columns and {"Pts","GD","GF"} <= set(tab.columns):
        tab = tab.sort_values(["Pts","GD","GF"], ascending=[False,False,False]).reset_index(drop=True)
        tab["Pos"] = np.arange(1, len(tab)+1)
    tab["Badge"] = tab["Pos"].apply(add_badge_by_position)
    # ensure order
    cols = ["Pos","Team","Pl","W","D","L","GF","GA","GD","Pts","Badge"]
    return tab[[c for c in cols if c in tab.columns]]

@st.cache_data(show_spinner=False)
def load_aux() -> dict[str, pd.DataFrame]:
    aux = {}
    for k in ["odds_out","pl_players","pl_teams","pl_clubs","pl_managers",
              "tm_mv_clubs","tm_mv_players","tm_jahrestabelle","transfers"]:
        p = CSV_MAP.get(k)
        aux[k] = safe_read_csv(p) if p else pd.DataFrame()
        # light normalization
        if k in ("tm_mv_clubs","tm_mv_players","pl_clubs","pl_players","pl_teams","pl_managers","tm_jahrestabelle"):
            for c in aux[k].columns:
                if c.lower() in ("club","team","team_name","Club"):
                    aux[k][c] = aux[k][c].map(canon_team)
    return aux

# =========================
# ------- FEATURES --------
# =========================

def elo_expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

def elo_update(ra, rb, score_a, k=20.0, home_adv=60.0, is_home_a=True, goal_diff=None) -> Tuple[float,float]:
    a_eff = ra + (home_adv if is_home_a else 0.0)
    b_eff = rb + (home_adv if not is_home_a else 0.0)
    ea = elo_expected(a_eff, b_eff)
    g = 1.0 if (goal_diff is None or goal_diff <= 1) else (1.5 if goal_diff == 2 else 1.75)
    delta = k * g * (score_a - ea)
    return ra + delta, rb - delta

def _points_for_result(ftr: str, side: str) -> int:
    if ftr == "D": return 1
    if (ftr == "H" and side == "H") or (ftr == "A" and side == "A"): return 3
    return 0

def _infer_ftr(row):
    if pd.notna(row.get("FTR")): return row["FTR"]
    if pd.notna(row.get("FTHG")) and pd.notna(row.get("FTAG")):
        if row["FTHG"] > row["FTAG"]: return "H"
        if row["FTHG"] < row["FTAG"]: return "A"
        return "D"
    return np.nan

def build_elo(df: pd.DataFrame, base_rating=1500.0) -> pd.DataFrame:
    need = {"Date","HomeTeam","AwayTeam","FTR","FTHG","FTAG"}
    if not need <= set(df.columns):
        raise ValueError("build_elo: missing columns")
    df = df.sort_values(["Date","HomeTeam","AwayTeam"]).reset_index(drop=True).copy()
    teams = pd.unique(pd.concat([df["HomeTeam"], df["AwayTeam"]]))
    rating = {t: float(base_rating) for t in teams}
    elo_h, elo_a = [], []
    for _, r in df.iterrows():
        h, a = r["HomeTeam"], r["AwayTeam"]
        eh, ea = rating.get(h, base_rating), rating.get(a, base_rating)
        elo_h.append(eh); elo_a.append(ea)
        ftr = _infer_ftr(r)
        score_h = 1.0 if ftr == "H" else (0.5 if ftr == "D" else 0.0)
        try:
            gd = int(abs(float(r["FTHG"]) - float(r["FTAG"])))
        except Exception:
            gd = None
        nh, na = elo_update(eh, ea, score_h, goal_diff=gd)
        rating[h], rating[a] = nh, na
    df["EloHome_pre"] = elo_h
    df["EloAway_pre"] = elo_a
    return df

def add_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    need = {"Date","HomeTeam","AwayTeam"}
    if not need <= set(df.columns):
        return df
    out = df.sort_values(["Date","HomeTeam","AwayTeam"]).reset_index(drop=True).copy()
    out["RestH"] = np.nan; out["RestA"] = np.nan
    last_seen: Dict[str, pd.Timestamp] = {}
    for i, r in out.iterrows():
        d = r["Date"]; h = r["HomeTeam"]; a = r["AwayTeam"]
        out.at[i, "RestH"] = (d - last_seen[h]).days if h in last_seen else np.nan
        out.at[i, "RestA"] = (d - last_seen[a]).days if a in last_seen else np.nan
        last_seen[h] = d; last_seen[a] = d
    out["RestH"] = out["RestH"].fillna(7.0)
    out["RestA"] = out["RestA"].fillna(7.0)
    return out

def add_lastN(df: pd.DataFrame, w_list=(3,5,10)) -> pd.DataFrame:
    """
    Adds rolling, strictly prior features for each N in w_list:
    - H_/A_ rGF, rGA, rGD, rW, rD, rL, rPPG
    """
    need = {"Date","HomeTeam","AwayTeam","FTR","FTHG","FTAG"}
    if not need <= set(df.columns):
        raise ValueError("add_lastN: missing columns")
    df = df.sort_values(["Date","HomeTeam","AwayTeam"]).reset_index(drop=True).copy()
    df["MatchID"] = np.arange(len(df))

    long_rows = []
    for i, r in df.iterrows():
        long_rows += [
            {"MatchID": i,"Date": r["Date"],"Team": r["HomeTeam"],"Opp": r["AwayTeam"],"Side":"H",
             "GF": float(r["FTHG"]), "GA": float(r["FTAG"]),
             "W": 1 if r["FTR"]=="H" else 0, "D": 1 if r["FTR"]=="D" else 0, "L": 1 if r["FTR"]=="A" else 0,
             "Pts": _points_for_result(r["FTR"], "H")},
            {"MatchID": i,"Date": r["Date"],"Team": r["AwayTeam"],"Opp": r["HomeTeam"],"Side":"A",
             "GF": float(r["FTAG"]), "GA": float(r["FTHG"]),
             "W": 1 if r["FTR"]=="A" else 0, "D": 1 if r["FTR"]=="D" else 0, "L": 1 if r["FTR"]=="H" else 0,
             "Pts": _points_for_result(r["FTR"], "A")},
        ]
    long = pd.DataFrame(long_rows).sort_values(["Team","Date","MatchID"]).reset_index(drop=True)

    def _roll(gr: pd.DataFrame) -> pd.DataFrame:
        gr = gr.sort_values(["Date","MatchID"]).copy()
        for w in w_list:
            for col in ["GF","GA","Pts","W","D","L"]:
                gr[f"r{col}{w}"] = gr[col].shift(1).rolling(w, min_periods=1).mean()
            gr[f"rGD{w}"] = gr[f"rGF{w}"] - gr[f"rGA{w}"]
        return gr

    long = long.groupby("Team", group_keys=False).apply(_roll)

    out = df.copy()
    for w in w_list:
        H = (long[long["Side"]=="H"][["MatchID", f"rGF{w}",f"rGA{w}",f"rGD{w}",f"rPts{w}",f"rW{w}",f"rD{w}",f"rL{w}"]]
             .rename(columns={
                 f"rGF{w}":f"H_rGF{w}", f"rGA{w}":f"H_rGA{w}", f"rGD{w}":f"H_rGD{w}",
                 f"rPts{w}":f"H_rPPG{w}", f"rW{w}":f"H_rW{w}", f"rD{w}":f"H_rD{w}", f"rL{w}":f"H_rL{w}"}))
        A = (long[long["Side"]=="A"][["MatchID", f"rGF{w}",f"rGA{w}",f"rGD{w}",f"rPts{w}",f"rW{w}",f"rD{w}",f"rL{w}"]]
             .rename(columns={
                 f"rGF{w}":f"A_rGF{w}", f"rGA{w}":f"A_rGA{w}", f"rGD{w}":f"A_rGD{w}",
                 f"rPts{w}":f"A_rPPG{w}", f"rW{w}":f"A_rW{w}", f"rD{w}":f"A_rD{w}", f"rL{w}":f"A_rL{w}"}))
        out = out.merge(H, on="MatchID", how="left").merge(A, on="MatchID", how="left")
    out = out.drop(columns=["MatchID"])
    return out

def add_h2h_rivalry(df: pd.DataFrame, n=5) -> pd.DataFrame:
    need = {"Date","HomeTeam","AwayTeam","FTR"}
    if not need <= set(df.columns):
        return df
    df = df.sort_values(["Date","HomeTeam","AwayTeam"]).reset_index(drop=True).copy()
    from collections import defaultdict, deque
    hist = defaultdict(lambda: deque(maxlen=n))
    h2h_h, h2h_a, h2h_d = [], [], []
    for _, r in df.iterrows():
        h, a = r["HomeTeam"], r["AwayTeam"]
        key = tuple(sorted((h, a)))
        q = hist[key]
        if len(q) == 0:
            h2h_h.append(np.nan); h2h_a.append(np.nan); h2h_d.append(np.nan)
        else:
            wh = sum(1 for w in q if w == h)
            wa = sum(1 for w in q if w == a)
            wd = sum(1 for w in q if w is None)
            t  = len(q)
            h2h_h.append(wh / t); h2h_a.append(wa / t); h2h_d.append(wd / t)
        # update
        if r["FTR"] == "H": q.append(h)
        elif r["FTR"] == "A": q.append(a)
        else: q.append(None)
    df["H2H_H_win_rate"] = h2h_h
    df["H2H_A_win_rate"] = h2h_a
    df["H2H_draw_rate"]  = h2h_d
    # Rivalry (tight set; add obvious ones)
    RIVAL = {
        ("Arsenal","Tottenham Hotspur"), ("Chelsea","Tottenham Hotspur"),
        ("Liverpool","Everton"), ("Liverpool","Manchester United"),
        ("Manchester City","Manchester United"),
        ("Crystal Palace","Brighton and Hove Albion"),
        ("Brentford","Fulham"),
        ("Leeds United","Manchester United"),
        ("Newcastle United","Sunderland"),
        ("West Ham United","Tottenham Hotspur"),
    }
    riv = []
    for _, r in df.iterrows():
        riv.append(1 if (r["HomeTeam"], r["AwayTeam"]) in RIVAL else 0)
    df["Rivalry"] = riv
    return df

def _eur_to_float(x) -> float:
    if pd.isna(x): return np.nan
    s = str(x).replace("€","").replace(",","").replace(" ","").lower()
    m = re.match(r"([0-9.]+)\s*([mbk])", s)
    try:
        if m:
            n = float(m.group(1))
            u = m.group(2)
            if u == "b": return n * 1_000_000_000
            if u == "m": return n * 1_000_000
            if u == "k": return n * 1_000
        return float(s)
    except Exception:
        return np.nan

@st.cache_data(show_spinner=False)
def build_team_features(aux: dict) -> pd.DataFrame:
    """
    Merge external CSVs to create per-team static features for the current season.
    Returns one row per Team with columns prefixed by T_* .
    """
    frames = []

    # Transfermarkt club market values
    mv_clubs = aux.get("tm_mv_clubs", pd.DataFrame())
    if not mv_clubs.empty:
        # choose likely columns
        cols = mv_clubs.columns
        c_team = next((c for c in cols if re.search(r"(club|team)", c, re.I)), None)
        c_val  = next((c for c in cols if re.search(r"(current|squad).*value", c, re.I)), None)
        if c_team and c_val:
            t = mv_clubs[[c_team, c_val]].copy()
            t.columns = ["Team","T_squad_value_eur"]
            t["T_squad_value_eur"] = t["T_squad_value_eur"].apply(_eur_to_float)
            frames.append(t)

    # Sum of players market values (players file)
    mv_players = aux.get("tm_mv_players", pd.DataFrame())
    if not mv_players.empty:
        cols = mv_players.columns
        c_team = next((c for c in cols if re.search(r"(club|team)", c, re.I)), None)
        c_val  = next((c for c in cols if re.search(r"value", c, re.I)), None)
        if c_team and c_val:
            t = mv_players[[c_team, c_val]].copy()
            t.columns = ["Team","_val"]
            t["_val"] = t["_val"].apply(_eur_to_float)
            t = t.groupby("Team", as_index=False)["_val"].sum().rename(columns={"_val":"T_players_mv_sum"})
            frames.append(t)

    # Transfers net spend (profit=positive, spend=negative)
    transfers = aux.get("transfers", pd.DataFrame())
    if not transfers.empty:
        cols = transfers.columns
        c_team = next((c for c in cols if re.search(r"(club|team)", c, re.I)), None)
        c_spent = next((c for c in cols if re.search(r"(spent|purchases|expenditure|out)", c, re.I)), None)
        c_in    = next((c for c in cols if re.search(r"(income|sales|in)", c, re.I)), None)
        if c_team and (c_spent or c_in):
            t = transfers.copy()
            t = t.rename(columns={c_team:"Team"})
            if c_spent: t["spent"] = t[c_spent].apply(_eur_to_float)
            else: t["spent"] = 0.0
            if c_in: t["income"] = t[c_in].apply(_eur_to_float)
            else: t["income"] = 0.0
            t["T_net_spend"] = (t["income"].fillna(0) - t["spent"].fillna(0))
            t = t[["Team","T_net_spend"]]
            frames.append(t)

    # Managers
    managers = aux.get("pl_managers", pd.DataFrame())
    if not managers.empty:
        cols = managers.columns
        c_team = next((c for c in cols if re.search(r"(team|club)", c, re.I)), None)
        c_ten  = next((c for c in cols if re.search(r"(tenure|days|since)", c, re.I)), None)
        if c_team and c_ten:
            t = managers[[c_team, c_ten]].copy()
            t.columns = ["Team","T_manager_tenure_days"]
            frames.append(t)

    # Jahrestabelle (per-year points/rank/prior strength proxy)
    jt = aux.get("tm_jahrestabelle", pd.DataFrame())
    if not jt.empty:
        cols = jt.columns
        c_team = next((c for c in cols if re.search(r"(team|club)", c, re.I)), None)
        c_rank = next((c for c in cols if re.search(r"rank", c, re.I)), None)
        c_pts  = next((c for c in cols if re.search(r"(pts|points)", c, re.I)), None)
        c_gp   = next((c for c in cols if re.search(r"(match|games|played)", c, re.I)), None)
        if c_team and (c_rank or c_pts):
            t = jt.copy()
            t = t.rename(columns={c_team:"Team"})
            if c_pts and c_gp:
                with np.errstate(divide="ignore", invalid="ignore"):
                    t["T_jt_ppm"] = pd.to_numeric(t[c_pts], errors="coerce") / pd.to_numeric(t[c_gp], errors="coerce").replace(0, np.nan)
            if c_rank:
                t["T_jt_rank"] = pd.to_numeric(t[c_rank], errors="coerce")
            t = t.groupby("Team", as_index=False).agg({"T_jt_ppm":"mean","T_jt_rank":"mean"})
            frames.append(t)

    # Club stats (pl_clubs) — prefix numeric columns with T_c_
    pl_clubs = aux.get("pl_clubs", pd.DataFrame())
    if not pl_clubs.empty:
        t = pl_clubs.copy()
        c_team = next((c for c in t.columns if re.search(r"(team_name|team|club)", c, re.I)), None)
        if c_team:
            t = t.rename(columns={c_team:"Team"})
            for c in list(t.columns):
                if c == "Team": continue
                if pd.api.types.is_numeric_dtype(t[c]):
                    t = t.rename(columns={c: f"T_c_{c}"})
            frames.append(t[["Team"] + [c for c in t.columns if c != "Team"]])

    # players/teams tables can be used similarly — we aggregate by team if sensible
    pl_players = aux.get("pl_players", pd.DataFrame())
    if not pl_players.empty:
        t = pl_players.copy()
        c_team = next((c for c in t.columns if re.search(r"(team|club)", c, re.I)), None)
        c_metric = "metric" if "metric" in t.columns else None
        c_value  = "value" if "value" in t.columns else None
        if c_team and c_metric and c_value:
            g = t.groupby([c_team, c_metric])[c_value].sum().unstack(c_metric).fillna(0.0)
            g.columns = [f"T_p_{c}" for c in g.columns]
            g = g.reset_index().rename(columns={c_team:"Team"})
            frames.append(g)

    # Combine all
    if not frames:
        return pd.DataFrame(columns=["Team"])
    out = frames[0]
    for nxt in frames[1:]:
        out = out.merge(nxt, on="Team", how="outer")
    # Fill basic medians for stability
    for c in out.columns:
        if c != "Team" and pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].astype(float)
            med = out[c].median()
            out[c] = out[c].fillna(med)
    return out

def last3_season_priors(hist: pd.DataFrame) -> pd.DataFrame:
    """
    Compute last-3-season priors per team: PPG, GDPM, homePPG (home), awayPPG (away).
    """
    if hist.empty:
        return pd.DataFrame(columns=["Team","T3_ppg","T3_gdpm","T3_home_ppg","T3_away_ppg"])
    by = hist.copy()
    by["TeamH"] = by["HomeTeam"]; by["TeamA"] = by["AwayTeam"]
    # global per-team tallies
    rows=[]
    for _, r in by.iterrows():
        # home row
        rows.append({"Team": r["TeamH"], "Pl":1, "GF":int(r["FTHG"]), "GA":int(r["FTAG"]),
                     "Pts": (3 if r["FTR"]=="H" else (1 if r["FTR"]=="D" else 0))})
        # away row
        rows.append({"Team": r["TeamA"], "Pl":1, "GF":int(r["FTAG"]), "GA":int(r["FTHG"]),
                     "Pts": (3 if r["FTR"]=="A" else (1 if r["FTR"]=="D" else 0))})
    g = pd.DataFrame(rows).groupby("Team", as_index=False).sum(numeric_only=True)
    g["T3_ppg"]  = g["Pts"] / g["Pl"].replace(0, np.nan)
    g["T3_gdpm"] = (g["GF"] - g["GA"]) / g["Pl"].replace(0, np.nan)
    g = g[["Team","T3_ppg","T3_gdpm"]]

    # home/away splits
    H = by.groupby("HomeTeam")["FTR"].apply(lambda s: (s=="H").sum()*3 + (s=="D").sum()).to_frame("homePts")
    H["homePl"] = by.groupby("HomeTeam")["FTR"].count()
    H["T3_home_ppg"] = H["homePts"] / H["homePl"].replace(0, np.nan)
    H = H.reset_index().rename(columns={"HomeTeam":"Team"})
    A = by.groupby("AwayTeam")["FTR"].apply(lambda s: (s=="A").sum()*3 + (s=="D").sum()).to_frame("awayPts")
    A["awayPl"] = by.groupby("AwayTeam")["FTR"].count()
    A["T3_away_ppg"] = A["awayPts"] / A["awayPl"].replace(0, np.nan)
    A = A.reset_index().rename(columns={"AwayTeam":"Team"})

    out = g.merge(H[["Team","T3_home_ppg"]], on="Team", how="outer") \
           .merge(A[["Team","T3_away_ppg"]], on="Team", how="outer")
    for c in out.columns:
        if c != "Team":
            out[c] = out[c].astype(float).fillna(out[c].median())
    return out

def add_static_team_feats(df: pd.DataFrame, team_feats: pd.DataFrame, priors: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join per-team features (home + away prefixes) to fixture rows.
    """
    out = df.copy()
    if not team_feats.empty:
        H = team_feats.add_prefix("H_").rename(columns={"H_Team":"HomeTeam"})
        A = team_feats.add_prefix("A_").rename(columns={"A_Team":"AwayTeam"})
        out = out.merge(H, on="HomeTeam", how="left").merge(A, on="AwayTeam", how="left")
    if not priors.empty:
        H = priors.add_prefix("H_").rename(columns={"H_Team":"HomeTeam"})
        A = priors.add_prefix("A_").rename(columns={"A_Team":"AwayTeam"})
        out = out.merge(H, on="HomeTeam", how="left").merge(A, on="AwayTeam", how="left")
    # Fill remaining numeric with medians
    num_cols = [c for c in out.columns if c not in ("Date","HomeTeam","AwayTeam","FTR") and pd.api.types.is_numeric_dtype(out[c])]
    for c in num_cols:
        out[c] = out[c].fillna(out[c].median())
    return out

def build_training_frame(hist: pd.DataFrame, aux: dict) -> Tuple[pd.DataFrame, List[str]]:
    """
    Main feature pipeline.
    """
    # Clean
    df = hist.dropna(subset=["FTHG","FTAG"]).copy()
    df["FTR"] = df.apply(_infer_ftr, axis=1)
    # Elo + rolling + H2H + rest
    df = build_elo(df)
    df = add_lastN(df, w_list=(3,5,10))
    df = add_h2h_rivalry(df, n=5)
    df = add_rest_days(df)
    # Team static features + last-3-season priors
    team_feats = build_team_features(aux)
    priors = last3_season_priors(df)
    df = add_static_team_feats(df, team_feats, priors)

    # Book odds if provided (odds_out.csv): merge on (Date, HomeTeam, AwayTeam) forgivingly
    odds = aux.get("odds_out", pd.DataFrame())
    if not odds.empty:
        o = odds.copy()
        # try to detect columns
        cand_home = [c for c in o.columns if re.search("home.*(prob|odd)", c, re.I)]
        cand_draw = [c for c in o.columns if re.search("draw.*(prob|odd)", c, re.I)]
        cand_away = [c for c in o.columns if re.search("away.*(prob|odd)", c, re.I)]
        c_date = next((c for c in o.columns if re.search(r"date", c, re.I)), None)
        c_h = next((c for c in o.columns if re.search(r"(home|hometeam)", c, re.I)), None)
        c_a = next((c for c in o.columns if re.search(r"(away|awayteam)", c, re.I)), None)
        if c_date and c_h and c_a and cand_home and cand_draw and cand_away:
            o = o.rename(columns={c_h:"HomeTeam", c_a:"AwayTeam", c_date:"Date"})
            o["Date"] = to_date(o["Date"])
            oh = o[cand_home[0]].apply(odds_to_prob).rename("Odds_H")
            od = o[cand_draw[0]].apply(odds_to_prob).rename("Odds_D")
            oa = o[cand_away[0]].apply(odds_to_prob).rename("Odds_A")
            o = pd.concat([o[["Date","HomeTeam","AwayTeam"]], oh, od, oa], axis=1)
            df = df.merge(o, on=["Date","HomeTeam","AwayTeam"], how="left")
    # Label
    LABEL = {"A":0,"D":1,"H":2}
    df["y"] = df["FTR"].map(LABEL).astype(int)

    # Feature columns
    base = [
        "EloHome_pre","EloAway_pre","RestH","RestA",
        "H2H_H_win_rate","H2H_A_win_rate","H2H_draw_rate","Rivalry",
        # lastN
        "H_rPPG3","A_rPPG3","H_rGF3","H_rGA3","A_rGF3","A_rGA3","H_rGD3","A_rGD3",
        "H_rPPG5","A_rPPG5","H_rGF5","H_rGA5","A_rGF5","A_rGA5","H_rGD5","A_rGD5",
        "H_rPPG10","A_rPPG10","H_rGF10","H_rGA10","A_rGF10","A_rGA10","H_rGD10","A_rGD10",
        # odds if any
        "Odds_H","Odds_D","Odds_A",
    ]
    # include static team features + priors
    extra = [c for c in df.columns if c.startswith(("H_T_","A_T_","H_T3_","A_T3_"))]
    # interactions
    inter = []
    for f in ["squad_value_eur","players_mv_sum","net_spend","manager_tenure_days","jt_ppm","jt_rank"]:
        h, a = f"H_T_{f}", f"A_T_{f}"
        if h in df.columns and a in df.columns:
            dname = f"Diff_{f}"
            df[dname] = df[h] - df[a]
            inter.append(dname)
    for f in ["ppg","gdpm","home_ppg","away_ppg"]:
        h, a = f"H_T3_{f}", f"A_T3_{f}"
        if h in df.columns and a in df.columns:
            dname = f"Diff_T3_{f}"
            df[dname] = df[h] - df[a]
            inter.append(dname)

    feat_cols = [c for c in base + extra + inter if c in df.columns]
    # fill NA
    df[feat_cols] = df[feat_cols].fillna(0.0)
    return df, feat_cols

# =========================
# ------- MODELING --------
# =========================

@dataclass
class ModelBundle:
    model: CalibratedClassifierCV
    rf: RandomForestClassifier
    train_df: pd.DataFrame
    feat_cols: List[str]
    oof_proba: np.ndarray
    cv: pd.DataFrame

@st.cache_resource(show_spinner=True)
def train_model(train_df: pd.DataFrame, feat_cols: List[str]) -> ModelBundle:
    X = train_df[feat_cols].astype(float).reset_index(drop=True)
    y = train_df["y"].astype(int).reset_index(drop=True)

    tscv = TimeSeriesSplit(n_splits=5)
    oof = np.zeros((len(X), 3), dtype=float)
    rows = []
    for k, (tr, va) in enumerate(tscv.split(X), start=1):
        rf = RandomForestClassifier(
            n_estimators=800, max_depth=None,
            min_samples_split=6, min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42 + k, n_jobs=-1
        )
        rf.fit(X.iloc[tr], y.iloc[tr])

        cal = CalibratedClassifierCV(rf, method="isotonic", cv=3)
        cal.fit(X.iloc[tr], y.iloc[tr])

        proba = cal.predict_proba(X.iloc[va])
        oof[va] = proba

        yhat = np.argmax(proba, axis=1)
        rows.append({
            "fold": k,
            "acc": accuracy_score(y.iloc[va], yhat),
            "logloss": log_loss(y.iloc[va], proba, labels=[0,1,2]),
            "macro_f1": f1_score(y.iloc[va], yhat, average="macro"),
        })

    cv = pd.DataFrame(rows)
    # final fit
    rf_final = RandomForestClassifier(
        n_estimators=1000, max_depth=None,
        min_samples_split=6, min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42, n_jobs=-1
    )
    rf_final.fit(X, y)
    cal_final = CalibratedClassifierCV(rf_final, method="isotonic", cv=3)
    cal_final.fit(X, y)
    return ModelBundle(cal_final, rf_final, train_df, feat_cols, oof, cv)

# =========================
# ---- PREDICTION UTILS ---
# =========================

def elo_dict_after(history: pd.DataFrame) -> Dict[str, float]:
    teams = pd.unique(pd.concat([history["HomeTeam"], history["AwayTeam"]]))
    R = {t: 1500.0 for t in teams}
    for _, r in history.sort_values("Date").iterrows():
        ra, rb = R[r["HomeTeam"]], R[r["AwayTeam"]]
        ftr = _infer_ftr(r)
        score_h = 1.0 if ftr=="H" else (0.5 if ftr=="D" else 0.0)
        try: gd = int(abs(float(r["FTHG"]) - float(r["FTAG"])))
        except Exception: gd = None
        nh, na = elo_update(ra, rb, score_h, goal_diff=gd)
        R[r["HomeTeam"]], R[r["AwayTeam"]] = nh, na
    return R

def features_for_fixtures(fixt: pd.DataFrame, bundle: ModelBundle, aux: dict, history: pd.DataFrame) -> pd.DataFrame:
    """
    Build (strictly pre-match) feature rows for fixt using the same columns as training.
    """
    df = fixt.copy()
    df["HomeTeam"] = df["home"].map(canon_team)
    df["AwayTeam"] = df["away"].map(canon_team)
    df["Date"] = df["date"]

    # Build an "as-of" history (only past)
    hist = history.copy()

    # Elo now
    elo_now = elo_dict_after(hist)

    # minimal scaffolding
    rows = []
    for _, m in df.iterrows():
        d, h, a = m["Date"], m["HomeTeam"], m["AwayTeam"]
        # lastN using history as-of d
        past = hist[hist["Date"] < d].copy()
        if past.empty:
            past = hist.copy()
        # compute small lastN for both sides
        def _lastn_side(past, team, side, N):
            if side == "H":
                t = past[past["HomeTeam"]==team].sort_values("Date").tail(N)
                if t.empty: return {"GF":0,"GA":0,"W":0,"D":0,"L":0,"PPG":0}
                GF = t["FTHG"].astype(float).values
                GA = t["FTAG"].astype(float).values
                W  = (t["FTR"]=="H").astype(int).sum()
                D  = (t["FTR"]=="D").astype(int).sum()
                L  = (t["FTR"]=="A").astype(int).sum()
            else:
                t = past[past["AwayTeam"]==team].sort_values("Date").tail(N)
                if t.empty: return {"GF":0,"GA":0,"W":0,"D":0,"L":0,"PPG":0}
                GF = t["FTAG"].astype(float).values
                GA = t["FTHG"].astype(float).values
                W  = (t["FTR"]=="A").astype(int).sum()
                D  = (t["FTR"]=="D").astype(int).sum()
                L  = (t["FTR"]=="H").astype(int).sum()
            PPG = (3*W + 1*D) / max(1, len(GF))
            return {"GF":GF.mean(), "GA":GA.mean(), "W":W, "D":D, "L":L, "PPG":PPG}

        feats = {
            "Date": d, "HomeTeam":h, "AwayTeam":a,
            "EloHome_pre": elo_now.get(h,1500.0),
            "EloAway_pre": elo_now.get(a,1500.0),
        }
        for N in (3,5,10):
            H = _lastn_side(past, h, "H", N); A = _lastn_side(past, a, "A", N)
            feats[f"H_rGF{N}"] = H["GF"]; feats[f"H_rGA{N}"] = H["GA"]; feats[f"H_rGD{N}"] = H["GF"]-H["GA"]; feats[f"H_rPPG{N}"] = H["PPG"]
            feats[f"A_rGF{N}"] = A["GF"]; feats[f"A_rGA{N}"] = A["GA"]; feats[f"A_rGD{N}"] = A["GF"]-A["GA"]; feats[f"A_rPPG{N}"] = A["PPG"]

        # h2h (last 5)
        h2h = past[((past["HomeTeam"]==h)&(past["AwayTeam"]==a)) | ((past["HomeTeam"]==a)&(past["AwayTeam"]==h))].sort_values("Date").tail(5)
        if h2h.empty:
            feats["H2H_H_win_rate"]=np.nan; feats["H2H_A_win_rate"]=np.nan; feats["H2H_draw_rate"]=np.nan
        else:
            WH = ((h2h["HomeTeam"]==h)&(h2h["FTR"]=="H")).sum() + ((h2h["AwayTeam"]==h)&(h2h["FTR"]=="A")).sum()
            WA = ((h2h["HomeTeam"]==a)&(h2h["FTR"]=="H")).sum() + ((h2h["AwayTeam"]==a)&(h2h["FTR"]=="A")).sum()
            DD = (h2h["FTR"]=="D").sum()
            T = len(h2h)
            feats["H2H_H_win_rate"]=WH/T; feats["H2H_A_win_rate"]=WA/T; feats["H2H_draw_rate"]=DD/T

        # rest days
        def _rest(team):
            t = past[(past["HomeTeam"]==team)|(past["AwayTeam"]==team)]
            if t.empty: return 7.0
            last = t["Date"].max()
            return float((d - last).days)
        feats["RestH"] = _rest(h); feats["RestA"] = _rest(a)
        rows.append(feats)

    X = pd.DataFrame(rows)
    # add static team feats + priors consistent with training
    team_feats = build_team_features(aux)
    priors = last3_season_priors(history)
    X = add_static_team_feats(X, team_feats, priors)

    # ensure all training features exist
    for c in bundle.feat_cols:
        if c not in X.columns:
            X[c] = 0.0
    return X[bundle.feat_cols], df[["date","home","away"]].reset_index(drop=True)

def pick_str(p_home, p_draw, p_away) -> str:
    if max(p_home, p_draw, p_away) == p_home: return "Home"
    if max(p_home, p_draw, p_away) == p_away: return "Away"
    return "Draw"

def format_pick_cell(pick: str, home: str, away: str) -> str:
    if pick == "Home":
        return f"<b style='color:{GOOD}'>{home}</b>"
    if pick == "Away":
        return f"<b style='color:{BAD}'>{away}</b>"
    return f"<i style='color:{DRAW}'>Draw</i>"

def badge_cell(pos: int) -> str:
    b = add_badge_by_position(int(pos))
    if b == "UCL":  color = PRIMARY
    elif b == "UEL": color = ACCENT
    elif b == "UECL": color = "#f59e0b"
    elif b == "Relegation": color = BAD
    else: color = MUTED
    return f"<span style='padding:2px 8px;border-radius:999px;background:{color}20;color:{color};font-weight:600'>{b or ''}</span>"

# =========================
# ---------- APP ----------
# =========================

st.set_page_config(
    page_title="Premier League — Latest MD Predictions & Projection",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
st.markdown(
    f"""
    <div style="display:flex;align-items:center;gap:12px">
      <span style="font-size:28px">⚽</span>
      <h1 style="margin:0">Premier League — Latest Matchday Predictions & Season Projection</h1>
    </div>
    <p style="color:{MUTED}">
      Random Forest model over historical E0 seasons with Elo + multi-horizon form + H2H + rest + squad value / transfers / manager tenure / Jahrestabelle / priors (+ optional odds).
      <br>No files are written; everything renders live in this dashboard.
    </p>
    """,
    unsafe_allow_html=True
)

# Load data
with st.spinner("Loading data and building features…"):
    hist = load_history()
    fx   = load_fixtures()
    tab  = load_current_table()
    aux  = load_aux()
    train_df, FEATS = build_training_frame(hist, aux)
    bundle = train_model(train_df, FEATS)

# Sidebar Legend
with st.sidebar:
    st.subheader("Legend")
    st.markdown(f"- **Green bold**: predicted winner")
    st.markdown(f"- **Red bold**: predicted loser")
    st.markdown(f"- *Purple italic*: predicted draw")
    st.markdown("- Top 5: Champions League")
    st.markdown("- 6–7: Europa League; 8: Conference League")
    st.markdown("- Bottom 3: Relegation")
    st.markdown("---")
    if not bundle.cv.empty:
        st.caption("RF CV (time-ordered, 5-fold):")
        st.metric("Accuracy", f"{bundle.cv['acc'].mean():.3f}")
        st.metric("Macro F1", f"{bundle.cv['macro_f1'].mean():.3f}")
        st.metric("Log loss", f"{bundle.cv['logloss'].mean():.3f}")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Current Table", "🧮 Next Matchday", "🏁 MD38 Projection"])

# --------- TAB 1: Current Table
with tab1:
    st.subheader(f"Current Table (after MD {int(fx[fx['played']]['md'].max()) if not fx.empty and fx['played'].any() else '—'})")
    if tab.empty:
        st.info("`current_table.csv` not found or empty.")
    else:
        show = tab.copy()
        show["Badge"] = show["Pos"].apply(badge_cell)
        st.dataframe(
            show.style.format(precision=0).hide(axis="index"),
            use_container_width=True
        )
        # Visuals Row (GD, GF, GA)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption("Goal Difference")
            fig = px.bar(tab, x="Team", y="GD", color="GD", color_continuous_scale="RdBu", height=320)
            fig.update_layout(margin=dict(l=10,r=10,t=10,b=10), coloraxis_showscale=False)
            fig.update_xaxes(tickangle=90)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.caption("Goals Scored (GF)")
            fig = px.bar(tab, x="Team", y="GF", color="GF", color_continuous_scale="Blues", height=320)
            fig.update_layout(margin=dict(l=10,r=10,t=10,b=10), coloraxis_showscale=False)
            fig.update_xaxes(tickangle=90)
            st.plotly_chart(fig, use_container_width=True)
        with col3:
            st.caption("Goals Conceded (GA)")
            fig = px.bar(tab, x="Team", y="GA", color="GA", color_continuous_scale="Reds", height=320)
            fig.update_layout(margin=dict(l=10,r=10,t=10,b=10), coloraxis_showscale=False)
            fig.update_xaxes(tickangle=90)
            st.plotly_chart(fig, use_container_width=True)

# --------- TAB 2: Next Matchday
with tab2:
    if fx.empty:
        st.info("`fixtures_2025_26.csv` not found or empty.")
    else:
        next_md = int(fx.loc[~fx["played"], "md"].min()) if (~fx["played"]).any() else int(fx["md"].max())
        st.subheader(f"Matchday {next_md} Predictions")
        target = fx[fx["md"] == next_md].copy()
        if target.empty:
            st.info("No fixtures found for the next matchday.")
        else:
            X_pred, meta = features_for_fixtures(target, bundle, aux, hist)
            proba = bundle.model.predict_proba(X_pred.values)
            out = meta.copy()
            out["P_home"] = proba[:,2]; out["P_draw"] = proba[:,1]; out["P_away"] = proba[:,0]
            out["Pick"] = out.apply(lambda r: pick_str(r["P_home"], r["P_draw"], r["P_away"]), axis=1)
            out["PickTeam"] = np.where(out["Pick"]=="Home", out["home"],
                                np.where(out["Pick"]=="Away", out["away"], "Draw"))
            # Nice formatting
            fmt = out.copy()
            fmt["Pick"] = fmt.apply(lambda r: format_pick_cell(r["Pick"], r["home"], r["away"]), axis=1)
            fmt = fmt.rename(columns={"date":"Date","home":"Home","away":"Away"})
            st.dataframe(
                fmt[["date","home","away","P_home","P_draw","P_away","Pick"]]
                    .rename(columns={"date":"Date","home":"Home","away":"Away"})
                    .style.format({"P_home":"{:.3f}","P_draw":"{:.3f}","P_away":"{:.3f}"})
                          .hide(axis="index")
                          .set_properties(subset=["Pick"], **{"text-align":"center"}),
                use_container_width=True, height=420
            )

            # Horizontal probability bars per fixture
            st.markdown("#### Win/Draw Probabilities")
            for _, r in out.iterrows():
                lbl = f"{r['home']} vs {r['away']}"
                fig = go.Figure()
                fig.add_trace(go.Bar(name="Home", x=[r["P_home"]], y=[lbl], orientation="h", marker_color=GOOD))
                fig.add_trace(go.Bar(name="Draw", x=[r["P_draw"]], y=[lbl], orientation="h", marker_color=DRAW))
                fig.add_trace(go.Bar(name="Away", x=[r["P_away"]], y=[lbl], orientation="h", marker_color=BAD))
                fig.update_layout(barmode="stack", height=54, margin=dict(l=6,r=6,t=4,b=4), showlegend=False,
                                  xaxis=dict(range=[0,1], tickformat=".0%"))
                st.plotly_chart(fig, use_container_width=True)

# --------- TAB 3: MD38 Projection
with tab3:
    st.subheader("Projection to MD38 (Expected Points)")
    if fx.empty:
        st.info("`fixtures_2025_26.csv` not found or empty.")
    else:
        # Expected points from predicted probabilities over remaining schedule
        to_play = fx[~fx["played"]].copy()
        if to_play.empty:
            st.info("No remaining fixtures to project.")
        else:
            X_rem, rem_meta = features_for_fixtures(to_play, bundle, aux, hist)
            P = bundle.model.predict_proba(X_rem.values)
            scored = pd.DataFrame({
                "home": rem_meta["home"], "away": rem_meta["away"],
                "ph": P[:,2], "pd": P[:,1], "pa": P[:,0]
            })
            teams = sorted(set(pd.unique(pd.concat([scored["home"], scored["away"]]).map(canon_team))))
            agg = {t: {"ExpPts":0.0} for t in teams}
            for _, r in scored.iterrows():
                h, a = r["home"], r["away"]
                agg[h]["ExpPts"] += 3*r["ph"] + 1*r["pd"]
                agg[a]["ExpPts"] += 3*r["pa"] + 1*r["pd"]
            exp = pd.DataFrame([{"Team":t, **v} for t,v in agg.items()])

            # Merge with current actual table (if any)
            if not tab.empty:
                base = tab[["Team","Pl","W","D","L","GF","GA","GD","Pts"]].copy()
            else:
                base = pd.DataFrame({"Team": teams, "Pl":0,"W":0,"D":0,"L":0,"GF":0,"GA":0,"GD":0,"Pts":0})
            final = base.merge(exp, on="Team", how="left").fillna({"ExpPts":0.0})
            final["Pts_pred"] = final["Pts"] + final["ExpPts"]
            # Round to a plausible integer table (simple rounding)
            final["Pts_pred_round"] = final["Pts_pred"].round().astype(int)
            final = final.sort_values(["Pts_pred","GD","GF"], ascending=[False,False,False]).reset_index(drop=True)
            final.insert(0, "ExpPos", np.arange(1, len(final)+1))
            final["Badge"] = final["ExpPos"].apply(badge_cell)
            champ = final.iloc[0]["Team"] if not final.empty else "—"
            st.success(f"🏆 Projected Champion: **{champ}**")

            st.dataframe(
                final[["ExpPos","Team","Pts_pred_round","Pts_pred","Badge"]]
                     .style.format({"Pts_pred_round":"{:,.0f}","Pts_pred":"{:,.4f}"})
                           .hide(axis="index"),
                use_container_width=True, height=500
            )

            # Extra visuals: expected points vs current points
            if not final.empty:
                st.markdown("#### Expected vs Current Points")
                fig = px.scatter(final, x="Pts", y="Pts_pred", text="Team", color="Pts_pred",
                                 color_continuous_scale="Viridis", height=420)
                fig.update_traces(textposition="top center")
                fig.update_layout(margin=dict(l=10,r=10,t=10,b=10), coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown(
    f"<div style='color:{MUTED};padding-top:16px'>"
    f"Made for analysis & entertainment. Probabilities are model-based and uncertain.</div>",
    unsafe_allow_html=True
)
