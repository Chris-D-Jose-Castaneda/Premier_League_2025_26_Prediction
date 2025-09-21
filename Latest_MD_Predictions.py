# Premier League — Latest Matchday Predictions & Season Projection (Streamlit)
# -----------------------------------------------------------------------------
# Streamlit dashboard: current table, next MD predictions, MD38 projection.
# Robust to messy CSVs/NaNs. No file writes.
# -----------------------------------------------------------------------------

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import streamlit as st
import plotly.express as px
import seaborn as sns

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# -----------------------------------------------------------------------------
# Page + theme
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Premier League — Predictions & Projection",
    page_icon="⚽",
    layout="wide",
)

THEME = dict(
    green="#10b981",
    red="#ef4444",
    blue="#2563eb",
    gray="#6b7280",
)

sns.set_theme(style="ticks")

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def repo_root() -> Path:
    return Path.cwd()

TEAM_ALIASES = {
    "Man City": "Manchester City",
    "Man Utd": "Manchester United",
    "Man United": "Manchester United",
    "Spurs": "Tottenham Hotspur",
    "Tottenham": "Tottenham Hotspur",
    "Leeds": "Leeds United",
    "Newcastle": "Newcastle United",
    "Nott'm Forest": "Nottingham Forest",
    "Wolves": "Wolverhampton Wanderers",
    "West Brom": "West Bromwich Albion",
    "Brighton": "Brighton and Hove Albion",
    "Bournemouth": "AFC Bournemouth",
    "West Ham": "West Ham United",
    "Sheffield Utd": "Sheffield United",
}

def canon_team(x: object) -> Optional[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    if not s:
        return None
    return TEAM_ALIASES.get(s, s)

def to_date(s: Iterable) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", dayfirst=True)

@st.cache_data(show_spinner=False)
def safe_read_csv(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p)
    except Exception:
        try:
            return pd.read_csv(p, encoding="latin-1")
        except Exception:
            return pd.DataFrame()

def ensure_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out

def add_badge_by_position(pos: int) -> str:
    if pos <= 5: return "UCL"
    if pos <= 7: return "UEL"
    if pos == 8: return "UECL"
    if pos >= 18: return "Relegation"
    return ""

@st.cache_data(show_spinner=False)
def list_csvs() -> Dict[str, Path]:
    root = repo_root()
    poss: List[Tuple[str, Path]] = [
        ("fixtures", root / "fixtures_2025_26.csv"),
        ("tm_mv_clubs", root / "tm_market_values_clubs.csv"),
        ("transfers", root / "transfers.csv"),
        ("pl_managers", root / "pl_managers_2025_26.csv"),
        ("tm_jahrestabelle", root / "tm_jahrestabelle.csv"),
    ]
    ct = sorted(root.glob("current_table*.csv"))
    if ct:
        poss.append(("current_table", ct[0]))
    dc = root / "data_cache"
    if dc.exists():
        for p in sorted(dc.glob("E0_*.csv")):
            poss.append((p.stem, p))
    out: Dict[str, Path] = {}
    for k, p in poss:
        if p.exists():
            out[k] = p
    return out

CSV_MAP = list_csvs()

# -----------------------------------------------------------------------------
# Loaders
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_history() -> pd.DataFrame:
    frames = []
    for key, p in CSV_MAP.items():
        if not key.startswith("E0_"):
            continue
        df = safe_read_csv(p)
        if df.empty:
            continue
        if {"Date", "HomeTeam", "AwayTeam"} - set(df.columns):
            continue
        df = df.copy()
        df["Date"] = to_date(df["Date"])
        df["HomeTeam"] = df["HomeTeam"].map(canon_team)
        df["AwayTeam"] = df["AwayTeam"].map(canon_team)
        df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam"]).copy()
        if "FTR" not in df.columns and {"FTHG", "FTAG"} <= set(df.columns):
            df["FTR"] = np.where(df["FTHG"] > df["FTAG"], "H",
                           np.where(df["FTHG"] < df["FTAG"], "A", "D"))
        frames.append(df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]])
    if not frames:
        return pd.DataFrame(columns=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values("Date")
        .reset_index(drop=True)
    )

@st.cache_data(show_spinner=False)
def load_fixtures() -> pd.DataFrame:
    p = CSV_MAP.get("fixtures")
    if not p:
        return pd.DataFrame(columns=["date","home","away","md","played","FTR","FTHG","FTAG"])
    fx = safe_read_csv(p)
    if fx.empty:
        return pd.DataFrame(columns=["date","home","away","md","played","FTR","FTHG","FTAG"])
    fx = fx.rename(columns={k:v for k,v in {"Date":"date","HomeTeam":"home","AwayTeam":"away"}.items() if k in fx.columns})
    for c in ["home","away"]:
        if c in fx.columns:
            fx[c] = fx[c].map(canon_team)
    if "date" in fx.columns:
        fx["date"] = to_date(fx["date"])
    if "played" not in fx.columns:
        fx["played"] = fx.get("FTR").notna() if "FTR" in fx.columns else False
    if "md" not in fx.columns:
        fx = fx.sort_values(["date","home","away"]).reset_index(drop=True)
        fx["md"] = (fx.groupby(fx["date"].dt.to_period("W")).ngroup() + 1).astype(int)
    cols = ["date","home","away","md","played","FTR","FTHG","FTAG"]
    return ensure_cols(fx, cols)[cols]

@st.cache_data(show_spinner=False)
def load_current_table() -> pd.DataFrame:
    p = CSV_MAP.get("current_table")
    if not p:
        return pd.DataFrame(columns=["Pos","Team","Pl","W","D","L","GF","GA","GD","Pts","Badge"])
    tab = safe_read_csv(p)
    if tab.empty:
        return pd.DataFrame(columns=["Pos","Team","Pl","W","D","L","GF","GA","GD","Pts","Badge"])
    ren = {"Pld":"Pl","Pld.":"Pl","Played":"Pl","TeamName":"Team","GoalDiff":"GD"}
    for k,v in ren.items():
        if k in tab.columns and v not in tab.columns:
            tab = tab.rename(columns={k:v})
    if "Team" in tab.columns:
        tab["Team"] = tab["Team"].map(canon_team)
        tab = tab.dropna(subset=["Team"]).copy()
    if "GD" not in tab.columns and {"GF","GA"} <= set(tab.columns):
        tab["GD"] = tab["GF"] - tab["GA"]
    if "Pts" not in tab.columns and {"W","D"} <= set(tab.columns):
        tab["Pts"] = 3*tab.get("W",0) + tab.get("D",0)
    if "Pos" not in tab.columns and {"Pts","GD","GF"} <= set(tab.columns):
        tab = tab.sort_values(["Pts","GD","GF"], ascending=[False,False,False]).reset_index(drop=True)
        tab["Pos"] = np.arange(1, len(tab)+1)
    tab["Badge"] = tab["Pos"].apply(add_badge_by_position) if "Pos" in tab.columns else ""
    keep = ["Pos","Team","Pl","W","D","L","GF","GA","GD","Pts","Badge"]
    return tab[[c for c in keep if c in tab.columns]].copy()

# -----------------------------------------------------------------------------
# Aux (club market values, transfers)
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_market_values_by_club() -> pd.DataFrame:
    p = CSV_MAP.get("tm_mv_clubs")
    if not p:
        return pd.DataFrame(columns=["Team","MarketValueEUR"])
    df = safe_read_csv(p)
    if df.empty:
        return pd.DataFrame(columns=["Team","MarketValueEUR"])
    team_col = next((c for c in df.columns if re.search(r"club|team", c, re.I)), None)
    val_col  = next((c for c in df.columns if re.search(r"value", c, re.I)), None)
    if not team_col or not val_col:
        return pd.DataFrame(columns=["Team","MarketValueEUR"])
    out = df[[team_col, val_col]].rename(columns={team_col:"Team", val_col:"MarketValueEUR"})
    out["Team"] = out["Team"].map(canon_team)

    def _to_eur(x):
        if pd.isna(x): return np.nan
        s = str(x).lower().replace("€","").replace(",","").strip()
        m = re.match(r"([0-9.]+)\\s*([mbk]n?)?", s)
        if not m:
            try: return float(s)
            except: return np.nan
        num = float(m.group(1)); unit = (m.group(2) or "").lower()
        if unit.startswith("b"): mult = 1_000_000_000.0
        elif unit.startswith("m"): mult = 1_000_000.0
        elif unit.startswith("k"): mult = 1_000.0
        else: mult = 1.0
        return num*mult

    out["MarketValueEUR"] = out["MarketValueEUR"].apply(_to_eur)
    return out.dropna(subset=["Team"]).groupby("Team", as_index=False).sum(numeric_only=True)

@st.cache_data(show_spinner=False)
def load_transfers_net() -> pd.DataFrame:
    p = CSV_MAP.get("transfers")
    if not p:
        return pd.DataFrame(columns=["Team","NetSpendEUR","Arrivals","Departures"])
    df = safe_read_csv(p)
    if df.empty:
        return pd.DataFrame(columns=["Team","NetSpendEUR","Arrivals","Departures"])
    team_col = next((c for c in df.columns if re.search(r"club|team", c, re.I)), None)
    out_col  = next((c for c in df.columns if re.search(r"(spend|expend|purchase|bought)", c, re.I)), None)
    in_col   = next((c for c in df.columns if re.search(r"(income|sold|sale)", c, re.I)), None)
    arr_col  = next((c for c in df.columns if re.search(r"arrival|in_count", c, re.I)), None)
    dep_col  = next((c for c in df.columns if re.search(r"depart|out_count", c, re.I)), None)
    if not team_col:
        return pd.DataFrame(columns=["Team","NetSpendEUR","Arrivals","Departures"])

    def _to_eur(x):
        if pd.isna(x): return 0.0
        s = str(x).lower().replace("€","").replace(",","").strip()
        m = re.match(r"([0-9.]+)\\s*([mbk]n?)?", s)
        if not m:
            try: return float(s)
            except: return 0.0
        num = float(m.group(1)); unit = (m.group(2) or "").lower()
        if unit.startswith("b"): mult = 1_000_000_000.0
        elif unit.startswith("m"): mult = 1_000_000.0
        elif unit.startswith("k"): mult = 1_000.0
        else: mult = 1.0
        return num*mult

    out = pd.DataFrame({
        "Team": df[team_col].map(canon_team),
        "Spend": df[out_col].apply(_to_eur) if out_col in df.columns else 0.0,
        "Income": df[in_col].apply(_to_eur) if in_col in df.columns else 0.0,
        "Arrivals": df.get(arr_col, pd.Series([np.nan]*len(df))),
        "Departures": df.get(dep_col, pd.Series([np.nan]*len(df))),
    })
    out = out.dropna(subset=["Team"]).copy()
    out["NetSpendEUR"] = out["Spend"] - out["Income"]
    return out[["Team","NetSpendEUR","Arrivals","Departures"]]

@st.cache_data(show_spinner=False)
def team_market_value_map() -> Dict[str, float]:
    df = load_market_values_by_club()
    return dict(zip(df["Team"], df["MarketValueEUR"]))

@st.cache_data(show_spinner=False)
def transfers_map() -> Dict[str, float]:
    df = load_transfers_net()
    return dict(zip(df["Team"], df["NetSpendEUR"]))

# -----------------------------------------------------------------------------
# Feature engineering (robust to missing Date)
# -----------------------------------------------------------------------------

LABEL_MAP = {"A": 0, "D": 1, "H": 2}

def _infer_ftr(r: pd.Series) -> str:
    if "FTR" in r and pd.notna(r["FTR"]):
        v = str(r["FTR"]).strip().upper()
        if v in {"H","D","A"}:
            return v
    try:
        hg, ag = float(r["FTHG"]), float(r["FTAG"])
        if hg > ag: return "H"
        if hg < ag: return "A"
        return "D"
    except Exception:
        return "D"

def elo_expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

def elo_update(ra: float, rb: float, score_a: float, *, k: float = 20.0, home_adv: float = 60.0, is_home_a: bool = True, goal_diff: Optional[int] = None) -> Tuple[float,float]:
    a_eff = ra + (home_adv if is_home_a else 0.0)
    b_eff = rb + (home_adv if not is_home_a else 0.0)
    ea = elo_expected(a_eff, b_eff)
    g = 1.0 if not goal_diff else (1.0 if goal_diff == 1 else (1.5 if goal_diff == 2 else 1.75))
    d = k * g * (score_a - ea)
    return ra + d, rb - d

def elo_dict_after(history: pd.DataFrame) -> Dict[str, float]:
    if history.empty:
        return {}
    H = history.copy()
    if "Date" not in H.columns:
        H["Date"] = pd.NaT
    H["Date"] = to_date(H["Date"])
    H = H.dropna(subset=["HomeTeam","AwayTeam"]).sort_values("Date")
    teams = pd.unique(pd.concat([H["HomeTeam"], H["AwayTeam"]]).dropna())
    R = {t: 1500.0 for t in teams}
    for _, r in H.iterrows():
        h, a = r["HomeTeam"], r["AwayTeam"]
        ra, rb = R.get(h, 1500.0), R.get(a, 1500.0)
        ftr = _infer_ftr(r)
        score_h = 1.0 if ftr=="H" else (0.5 if ftr=="D" else 0.0)
        try:
            gd = int(abs(float(r.get("FTHG", np.nan)) - float(r.get("FTAG", np.nan))))
        except Exception:
            gd = None
        nh, na = elo_update(ra, rb, score_h, goal_diff=gd)
        R[h], R[a] = nh, na
    return R

def _side_hist(df: pd.DataFrame, team: str, side: str, asof: pd.Timestamp, n: int = 5):
    if df is None or df.empty:
        return dict(GF=0.0, GA=0.0, W=0.0, D=0.0, L=0.0, PPG=0.0)
    D = df.copy()
    if "Date" not in D.columns:
        D["Date"] = pd.NaT
    D["Date"] = to_date(D["Date"])
    H = D[D["Date"] < asof]
    if side == "H":
        H = H[H["HomeTeam"] == team].sort_values("Date").tail(n)
        if H.empty: return dict(GF=0.0, GA=0.0, W=0.0, D=0.0, L=0.0, PPG=0.0)
        GF = H.get("FTHG", 0).astype(float).values
        GA = H.get("FTAG", 0).astype(float).values
        W = (H.get("FTR","").astype(str)=="H").astype(float).values
        Dd= (H.get("FTR","").astype(str)=="D").astype(float).values
        L = (H.get("FTR","").astype(str)=="A").astype(float).values
    else:
        H = H[H["AwayTeam"] == team].sort_values("Date").tail(n)
        if H.empty: return dict(GF=0.0, GA=0.0, W=0.0, D=0.0, L=0.0, PPG=0.0)
        GF = H.get("FTAG", 0).astype(float).values
        GA = H.get("FTHG", 0).astype(float).values
        W = (H.get("FTR","").astype(str)=="A").astype(float).values
        Dd= (H.get("FTR","").astype(str)=="D").astype(float).values
        L = (H.get("FTR","").astype(str)=="H").astype(float).values
    m = max(1, len(GF))
    return dict(GF=float(GF.mean()), GA=float(GA.mean()), W=float(W.mean()), D=float(Dd.mean()), L=float(L.mean()), PPG=float((3*W.sum()+Dd.sum())/m))

def h2h_rates(history: pd.DataFrame, home: str, away: str, asof: pd.Timestamp, n: int = 5):
    if history.empty:
        return (np.nan, np.nan, np.nan)
    H = history.copy()
    if "Date" not in H.columns:
        H["Date"] = pd.NaT
    H["Date"] = to_date(H["Date"])
    h = H[(((H["HomeTeam"]==home)&(H["AwayTeam"]==away)) | ((H["HomeTeam"]==away)&(H["AwayTeam"]==home))) & (H["Date"] < asof)]
    h = h.sort_values("Date").tail(n)
    if h.empty:
        return (np.nan, np.nan, np.nan)
    wins_h = ((h["HomeTeam"]==home)&(h["FTR"]=="H")).sum() + ((h["AwayTeam"]==home)&(h["FTR"]=="A")).sum()
    wins_a = ((h["HomeTeam"]==away)&(h["FTR"]=="H")).sum() + ((h["AwayTeam"]==away)&(h["FTR"]=="A")).sum()
    draws = (h["FTR"]=="D").sum()
    tot = len(h)
    return (wins_h/tot, wins_a/tot, draws/tot)

def rest_days(history: pd.DataFrame, team: str, asof: pd.Timestamp) -> float:
    if history.empty:
        return 7.0
    H = history.copy()
    if "Date" not in H.columns:
        H["Date"] = pd.NaT
    H["Date"] = to_date(H["Date"])
    h = H[((H["HomeTeam"]==team)|(H["AwayTeam"]==team)) & (H["Date"] < asof)]
    if h.empty:
        return 7.0
    last = h["Date"].max()
    try:
        return float((asof - last).days)
    except Exception:
        return 7.0

# -----------------------------------------------------------------------------
# Training bundle
# -----------------------------------------------------------------------------

@dataclass
class TrainBundle:
    model: RandomForestClassifier
    feat_cols: List[str]
    cv_acc: float
    cv_f1: float
    history: pd.DataFrame

BASE_FEATS = [
    "EloHome_pre","EloAway_pre",
    "H_Last5_PPG","A_Last5_PPG",
    "H_rGF","H_rGA","H_rGD",
    "A_rGF","A_rGA","A_rGD",
    "H2H_H_win_rate","H2H_A_win_rate","H2H_draw_rate",
    "RestH","RestA",
    "MV_diff_log","NetSpend_diff",
]

def build_features_for_training(hist: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    hist = ensure_cols(hist, ["Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR"]).copy()
    hist["Date"] = to_date(hist["Date"])
    hist = hist.dropna(subset=["HomeTeam","AwayTeam"]).sort_values(["Date","HomeTeam","AwayTeam"]).reset_index(drop=True)

    rows = []
    seen: List[Dict[str,object]] = []
    teams_all = pd.unique(pd.concat([hist["HomeTeam"], hist["AwayTeam"]]).dropna())

    for _, r in hist.iterrows():
        d = pd.to_datetime(r["Date"])
        h, a = r["HomeTeam"], r["AwayTeam"]
        if pd.isna(h) or pd.isna(a):
            continue

        hist_asof = pd.DataFrame(seen)
        R = elo_dict_after(hist_asof) if not hist_asof.empty else {t:1500.0 for t in teams_all}
        H5 = _side_hist(hist_asof, h, "H", d, 5)
        A5 = _side_hist(hist_asof, a, "A", d, 5)
        H2H_H, H2H_A, H2H_D = h2h_rates(hist_asof, h, a, d, 5)
        RestH = rest_days(hist_asof, h, d)
        RestA = rest_days(hist_asof, a, d)

        MV = team_market_value_map()
        TR = transfers_map()
        mv_h, mv_a = MV.get(h, np.nan), MV.get(a, np.nan)
        MV_diff_log = float(np.log1p(mv_h) - np.log1p(mv_a)) if (pd.notna(mv_h) and pd.notna(mv_a)) else 0.0
        NetSpend_diff = float(TR.get(h, 0.0) - TR.get(a, 0.0))

        rows.append({
            "Date": d,
            "HomeTeam": h,
            "AwayTeam": a,
            "EloHome_pre": R.get(h, 1500.0),
            "EloAway_pre": R.get(a, 1500.0),
            "H_Last5_PPG": H5["PPG"], "A_Last5_PPG": A5["PPG"],
            "H_rGF": H5["GF"], "H_rGA": H5["GA"], "H_rGD": H5["GF"]-H5["GA"],
            "A_rGF": A5["GF"], "A_rGA": A5["GA"], "A_rGD": A5["GF"]-A5["GA"],
            "H2H_H_win_rate": H2H_H, "H2H_A_win_rate": H2H_A, "H2H_draw_rate": H2H_D,
            "RestH": RestH, "RestA": RestA,
            "MV_diff_log": MV_diff_log, "NetSpend_diff": NetSpend_diff,
            "y": LABEL_MAP.get(_infer_ftr(r), 1),
        })

        seen.append({
            "Date": d,
            "HomeTeam": h,
            "AwayTeam": a,
            "FTHG": r.get("FTHG", np.nan),
            "FTAG": r.get("FTAG", np.nan),
            "FTR": _infer_ftr(r),
        })

    df = pd.DataFrame(rows)
    cols = [c for c in BASE_FEATS if c in df.columns]
    return df, cols

@st.cache_data(show_spinner=True)
def load_training_bundle() -> TrainBundle:
    history = load_history()
    if history.empty:
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        return TrainBundle(model=model, feat_cols=BASE_FEATS, cv_acc=0.0, cv_f1=0.0, history=history)

    train_df, feat_cols = build_features_for_training(history)
    X = train_df[feat_cols].fillna(0.0).astype(float)
    y = train_df["y"].astype(int)

    tscv = TimeSeriesSplit(n_splits=5)
    accs, f1s = [], []
    for fold, (tr, va) in enumerate(tscv.split(X), 1):
        rf = RandomForestClassifier(
            n_estimators=650,
            max_depth=None,
            min_samples_split=6,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42+fold,
            n_jobs=-1,
        )
        rf.fit(X.iloc[tr], y.iloc[tr])
        P = rf.predict_proba(X.iloc[va])
        pred = np.argmax(P, axis=1)
        accs.append(float(accuracy_score(y.iloc[va], pred)))
        f1s.append(float(f1_score(y.iloc[va], pred, average="macro")))

    final_rf = RandomForestClassifier(
        n_estimators=900,
        max_depth=None,
        min_samples_split=6,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    final_rf.fit(X, y)

    return TrainBundle(
        model=final_rf,
        feat_cols=feat_cols,
        cv_acc=float(np.mean(accs)),
        cv_f1=float(np.mean(f1s)),
        history=history,
    )

# -----------------------------------------------------------------------------
# Inference helpers
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def fixtures_latest_unplayed(fx: pd.DataFrame) -> Tuple[int, pd.DataFrame]:
    if fx.empty:
        return (0, fx)
    last_played = fx.loc[fx["played"]==True, "md"].max() if fx["played"].any() else 0
    target = int(last_played + 1)
    subset = fx[fx["md"] == target].copy()
    return (target, subset)

def features_for_fixtures(fixtures: pd.DataFrame, bundle: TrainBundle, hist: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if fixtures.empty:
        return pd.DataFrame(), pd.DataFrame()
    MV = team_market_value_map()
    TR = transfers_map()
    rows = []
    for _, m in fixtures.iterrows():
        d = pd.to_datetime(m.get("date"))
        h = canon_team(m.get("home"))
        a = canon_team(m.get("away"))
        if not h or not a:
            continue

        H_asof = hist.copy()
        if "Date" not in H_asof.columns:
            H_asof["Date"] = pd.NaT
        H_asof["Date"] = to_date(H_asof["Date"])
        H_asof = H_asof[H_asof["Date"] < d]

        R = elo_dict_after(H_asof)
        H5 = _side_hist(H_asof, h, "H", d, 5)
        A5 = _side_hist(H_asof, a, "A", d, 5)
        H2H_H, H2H_A, H2H_D = h2h_rates(H_asof, h, a, d, 5)
        RestH = rest_days(H_asof, h, d)
        RestA = rest_days(H_asof, a, d)

        mv_h, mv_a = MV.get(h, np.nan), MV.get(a, np.nan)
        MV_diff_log = float(np.log1p(mv_h) - np.log1p(mv_a)) if (pd.notna(mv_h) and pd.notna(mv_a)) else 0.0
        NetSpend_diff = float(TR.get(h, 0.0) - TR.get(a, 0.0))

        rows.append({
            "date": d,
            "home": h,
            "away": a,
            "EloHome_pre": R.get(h, 1500.0),
            "EloAway_pre": R.get(a, 1500.0),
            "H_Last5_PPG": H5["PPG"], "A_Last5_PPG": A5["PPG"],
            "H_rGF": H5["GF"], "H_rGA": H5["GA"], "H_rGD": H5["GF"]-H5["GA"],
            "A_rGF": A5["GF"], "A_rGA": A5["GA"], "A_rGD": A5["GF"]-A5["GA"],
            "H2H_H_win_rate": H2H_H, "H2H_A_win_rate": H2H_A, "H2H_draw_rate": H2H_D,
            "RestH": RestH, "RestA": RestA,
            "MV_diff_log": MV_diff_log, "NetSpend_diff": NetSpend_diff,
        })
    X = pd.DataFrame(rows)
    if X.empty:
        return X, pd.DataFrame()
    meta = X[["date","home","away"]].copy()
    X = X[bundle.feat_cols].fillna(0.0).astype(float)
    return X, meta

def pick_from_probs(pA: float, pD: float, pH: float) -> str:
    arr = np.array([pA, pD, pH])
    i = int(np.argmax(arr))
    return "Home" if i==2 else ("Away" if i==0 else "Draw")

# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------

def plot_current_table(tab: pd.DataFrame):
    if tab.empty:
        st.info("Current table not available.")
        return
    st.subheader("Current Table")
    st.dataframe(tab, use_container_width=True, hide_index=True)
    c1,c2,c3 = st.columns(3)
    with c1:
        st.caption("Goal Difference")
        fig = px.bar(tab.sort_values("GD", ascending=False), x="Team", y="GD", color="GD", color_continuous_scale="Tealgrn")
        fig.update_layout(height=320, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.caption("Goals Scored")
        fig = px.bar(tab.sort_values("GF", ascending=False), x="Team", y="GF", color="GF", color_continuous_scale="Blues")
        fig.update_layout(height=320, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c3:
        st.caption("Goals Conceded")
        fig = px.bar(tab.sort_values("GA", ascending=True), x="Team", y="GA", color="GA", color_continuous_scale="Reds")
        fig.update_layout(height=320, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig, use_container_width=True)

def render_predictions_table(pred_df: pd.DataFrame):
    if pred_df.empty:
        st.info("No fixtures to predict.")
        return
    def _fixture_html(row):
        pick = row["Pick"]
        if pick == "Home":
            return f"<span style='font-weight:700;color:{THEME['green']}'>{row['home']}</span> vs <span style='font-weight:700;color:{THEME['red']}'>{row['away']}</span>"
        if pick == "Away":
            return f"<span style='font-weight:700;color:{THEME['red']}'>{row['home']}</span> vs <span style='font-weight:700;color:{THEME['green']}'>{row['away']}</span>"
        return f"<span style='font-style:italic;color:{THEME['gray']}'>{row['home']} vs {row['away']}</span>"

    view = pred_df.copy()
    view["Fixture"] = view.apply(_fixture_html, axis=1)
    view = view[["date","Fixture","P_home","P_draw","P_away","Pick"]]
    st.dataframe(
        view,
        use_container_width=True,
        hide_index=True,
        column_config={
            "date": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
            "P_home": st.column_config.NumberColumn(format="%.3f"),
            "P_draw": st.column_config.NumberColumn(format="%.3f"),
            "P_away": st.column_config.NumberColumn(format="%.3f"),
            "Fixture": st.column_config.Column(width="large"),
        },
    )

def expected_points_from_probs(scored: pd.DataFrame) -> pd.DataFrame:
    teams = pd.unique(pd.concat([scored["home"], scored["away"]]))
    agg = {t: {"ExpPts":0.0} for t in teams}
    for _, r in scored.iterrows():
        h, a = r["home"], r["away"]
        ph, pdw, pa = float(r["P_home"]), float(r["P_draw"]), float(r["P_away"])
        agg[h]["ExpPts"] += 3*ph + pdw
        agg[a]["ExpPts"] += 3*pa + pdw
    return pd.DataFrame([{"Team":t, **v} for t, v in agg.items()])

def project_to_md38(fx: pd.DataFrame, played_tab: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    exp_pts = expected_points_from_probs(scored)
    base = played_tab[["Team","Pl","W","D","L","GF","GA","GD","Pts"]].copy() if not played_tab.empty else pd.DataFrame()
    if base.empty:
        teams = sorted(set(list(scored["home"]) + list(scored["away"])))
        base = pd.DataFrame({"Team": teams, "Pl":0, "W":0, "D":0, "L":0, "GF":0, "GA":0, "GD":0, "Pts":0})
    out = base.merge(exp_pts, on="Team", how="left").fillna({"ExpPts":0.0})
    out["Pts_pred"] = (out["Pts"] + out["ExpPts"]).round(1)
    out = out.sort_values(["Pts_pred","GD","GF"], ascending=[False,False,False]).reset_index(drop=True)
    out["ExpPos"] = np.arange(1, len(out)+1)
    out["Badge"] = out["ExpPos"].apply(add_badge_by_position)
    return out

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------

st.title("⚽ English Premier League — Matchday Predictions Dashboard")

bundle = load_training_bundle()

tab1, tab2, tab3 = st.tabs(["Current Table", "Next Matchday", "MD38 Projection"])

with tab1:
    plot_current_table(load_current_table())

with tab2:
    st.subheader("Next Matchday")
    fx = load_fixtures()
    if fx.empty:
        st.info("Fixtures not found.")
    else:
        target_md, subset = fixtures_latest_unplayed(fx)
        st.caption(f"Matchday {target_md}")

        # Build history = training history + already played rows from fixtures
        hist = bundle.history.copy()
        if fx["played"].any():
            add = fx[fx["played"]].rename(columns={"date":"Date","home":"HomeTeam","away":"AwayTeam"})
            if "Date" not in add.columns: add["Date"] = pd.NaT
            add["Date"] = to_date(add["Date"])
            if "FTR" not in add.columns and {"FTHG","FTAG"} <= set(add.columns):
                add["FTR"] = np.where(add["FTHG"]>add["FTAG"], "H", np.where(add["FTHG"]<add["FTAG"], "A", "D"))
            add = ensure_cols(add, ["Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR"]).dropna(subset=["HomeTeam","AwayTeam"])
            hist = pd.concat([hist, add[["Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR"]]], ignore_index=True)
            hist = hist.sort_values("Date").reset_index(drop=True)

        X_pred, meta = features_for_fixtures(subset, bundle, hist)
        if X_pred.empty:
            st.info("No fixtures available to score.")
        else:
            P = bundle.model.predict_proba(X_pred.values)
            out = meta.copy()
            out["P_away"] = P[:,0]
            out["P_draw"] = P[:,1]
            out["P_home"] = P[:,2]
            out["Pick"]   = [pick_from_probs(a, d, h) for a, d, h in zip(out["P_away"], out["P_draw"], out["P_home"])]
            render_predictions_table(out.sort_values("date"))

with tab3:
    st.subheader("Projection to MD38 (Expected Points)")
    fx = load_fixtures()
    if fx.empty:
        st.info("Fixtures not found.")
    else:
        tab_now = load_current_table()
        remaining = fx[~fx["played"]].copy()

        hist = bundle.history.copy()
        if fx["played"].any():
            add = fx[fx["played"]].rename(columns={"date":"Date","home":"HomeTeam","away":"AwayTeam"})
            if "Date" not in add.columns: add["Date"] = pd.NaT
            add["Date"] = to_date(add["Date"])
            if "FTR" not in add.columns and {"FTHG","FTAG"} <= set(add.columns):
                add["FTR"] = np.where(add["FTHG"]>add["FTAG"], "H", np.where(add["FTHG"]<add["FTAG"], "A", "D"))
            add = ensure_cols(add, ["Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR"]).dropna(subset=["HomeTeam","AwayTeam"])
            hist = pd.concat([hist, add[["Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR"]]], ignore_index=True).sort_values("Date").reset_index(drop=True)

        Xr, Mr = features_for_fixtures(remaining, bundle, hist)
        if Xr.empty:
            st.info("No remaining fixtures to score.")
        else:
            Pr = bundle.model.predict_proba(Xr.values)
            scored = pd.DataFrame({
                "home": Mr["home"], "away": Mr["away"],
                "P_home": Pr[:,2], "P_draw": Pr[:,1], "P_away": Pr[:,0]
            })
            proj = project_to_md38(fx, tab_now, scored)
            champ = proj.iloc[0]["Team"] if not proj.empty else "—"
            st.success(f"🏆 Projected Champion: {champ}")
            st.dataframe(
                proj[["ExpPos","Team","Pts_pred","Badge"]].rename(columns={"Pts_pred":"Pts_predicted"}),
                use_container_width=True,
                hide_index=True,
            )
