# Latest_MD_Predictions.py

from __future__ import annotations
import math, re, warnings
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Premier League — Latest Predictions",
                   page_icon="⚽", layout="wide")

# ----------------------------- Paths & Inputs -----------------------------
ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"

# Expected files in repo root
FIXTURES = ROOT / "fixtures_2025_26.csv"
CURR_TAB = ROOT / "current_table.csv"          # you said you saved this
TRANSFERS = ROOT / "transfers.csv"
MV_CLUBS = ROOT / "tm_market_values_clubs.csv"
MV_PLAYERS = ROOT / "tm_market_values_players.csv"
PL_MANAGERS = ROOT / "pl_managers_2025_26.csv"

# ----------------------------- Helpers -----------------------------------
TEAM_ALIASES = {
    "Man United": "Manchester United", "Man Utd":"Manchester United",
    "Man City":"Manchester City", "Tottenham":"Tottenham Hotspur",
    "Spurs":"Tottenham Hotspur", "Nott'm Forest":"Nottingham Forest",
    "Wolves":"Wolverhampton Wanderers", "West Ham":"West Ham United",
    "West Brom":"West Bromwich Albion", "Brighton":"Brighton and Hove Albion",
    "Newcastle":"Newcastle United", "Leeds":"Leeds United",
    "Bournemouth":"AFC Bournemouth", "Sheffield Utd":"Sheffield United",
}
def canon(team: str) -> str:
    if pd.isna(team): return team
    t = str(team).strip()
    return TEAM_ALIASES.get(t, t)

def season_code(yy: int) -> str:  # 12 -> "1213"
    return f"{yy:02d}{(yy+1)%100:02d}"

def read_all_e0(cache: Path) -> pd.DataFrame:
    files = sorted(cache.glob("E0_*.csv"))
    frames = []
    for p in files:
        try:
            df = pd.read_csv(p)
            # normalize + keep finished
            for c in ("HomeTeam","AwayTeam"): 
                if c in df.columns: df[c] = df[c].map(canon)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
            df = df.dropna(subset=["FTR"])
            df["Season"] = re.search(r"E0_(\d{4})", p.name).group(1)
            frames.append(df)
        except Exception:
            pass
    if not frames: 
        raise FileNotFoundError("No E0_* files found under data_cache/")
    return pd.concat(frames, ignore_index=True)

def build_table_from_results(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Pos","Team","Pl","W","D","L","GF","GA","GD","Pts"])
    rows = []
    for _, r in df.iterrows():
        h, a = r["HomeTeam"], r["AwayTeam"]
        hg, ag = int(r["FTHG"]), int(r["FTAG"])
        if r["FTR"] == "H":
            rows += [{"Team":h,"Pl":1,"W":1,"D":0,"L":0,"GF":hg,"GA":ag,"Pts":3},
                     {"Team":a,"Pl":1,"W":0,"D":0,"L":1,"GF":ag,"GA":hg,"Pts":0}]
        elif r["FTR"] == "A":
            rows += [{"Team":h,"Pl":1,"W":0,"D":0,"L":1,"GF":hg,"GA":ag,"Pts":0},
                     {"Team":a,"Pl":1,"W":1,"D":0,"L":0,"GF":ag,"GA":hg,"Pts":3}]
        else:
            rows += [{"Team":h,"Pl":1,"W":0,"D":1,"L":0,"GF":hg,"GA":ag,"Pts":1},
                     {"Team":a,"Pl":1,"W":0,"D":1,"L":0,"GF":ag,"GA":hg,"Pts":1}]
    tab = pd.DataFrame(rows).groupby("Team", as_index=False).sum(numeric_only=True)
    tab["GD"] = tab["GF"] - tab["GA"]
    tab["Pts"] = 3*tab["W"] + tab["D"]
    tab = tab.sort_values(["Pts","GD","GF"], ascending=[False,False,False]).reset_index(drop=True)
    tab["Pos"] = np.arange(1, len(tab)+1)
    return tab[["Pos","Team","Pl","W","D","L","GF","GA","GD","Pts"]]

# -- Elo
def elo_E(ra, rb): return 1.0/(1.0 + 10**((rb-ra)/400.0))
def elo_update(ra, rb, sA, k=20.0, hadv=60.0, homeA=True, gd=None):
    a_eff = ra + (hadv if homeA else 0.0)
    b_eff = rb + (hadv if not homeA else 0.0)
    eA = elo_E(a_eff, b_eff)
    g = 1.0 if not gd or gd==1 else (1.5 if gd==2 else 1.75)
    d = k*g*(sA - eA)
    return ra + d, rb - d

def attach_elo(df: pd.DataFrame) -> pd.DataFrame:
    need = {"Date","HomeTeam","AwayTeam","FTR","FTHG","FTAG"}
    if need - set(df.columns): return df
    df = df.sort_values(["Date","HomeTeam","AwayTeam"]).copy()
    teams = pd.unique(pd.concat([df["HomeTeam"], df["AwayTeam"]]))
    R = {t:1500.0 for t in teams}
    eH, eA = [], []
    for _, r in df.iterrows():
        h, a = r["HomeTeam"], r["AwayTeam"]
        eH.append(R.get(h,1500.0)); eA.append(R.get(a,1500.0))
        score_h = 1.0 if r["FTR"]=="H" else (0.5 if r["FTR"]=="D" else 0.0)
        try: gd = int(abs(float(r["FTHG"])-float(r["FTAG"])))
        except: gd = None
        R[h], R[a] = elo_update(R.get(h,1500.0), R.get(a,1500.0), score_h, gd=gd)
    df["EloHome_pre"], df["EloAway_pre"] = eH, eA
    return df

# -- last-5 form (home/away specific)
def add_last5(df: pd.DataFrame, w=5) -> pd.DataFrame:
    df = df.sort_values(["Date","HomeTeam","AwayTeam"]).copy()
    out = df.copy()
    def hist(team, side, asof):
        h = df[df["Date"] < asof]
        if side=="H":
            h = h[h["HomeTeam"]==team].tail(w)
            if h.empty: return 0,0,0,0,0,0
            GF, GA = h["FTHG"].values, h["FTAG"].values
            W = (h["FTR"]=="H").mean(); D = (h["FTR"]=="D").mean(); L = (h["FTR"]=="A").mean()
        else:
            h = h[h["AwayTeam"]==team].tail(w)
            if h.empty: return 0,0,0,0,0,0
            GF, GA = h["FTAG"].values, h["FTHG"].values
            W = (h["FTR"]=="A").mean(); D = (h["FTR"]=="D").mean(); L = (h["FTR"]=="H").mean()
        PPG = (3*(W*w) + 1*(D*w))/max(1,len(GF))
        return GF.mean(), GA.mean(), (GF-GA).mean(), W, D, L, PPG
    H_cols = {"H_rGF":[],"H_rGA":[],"H_rGD":[],"H_rW":[],"H_rD":[],"H_rL":[],"H_Last5_PPG":[]}
    A_cols = {"A_rGF":[],"A_rGA":[],"A_rGD":[],"A_rW":[],"A_rD":[],"A_rL":[],"A_Last5_PPG":[]}
    for _, r in df.iterrows():
        hg,ga,gd,W,D,L,PPG = hist(r["HomeTeam"],"H",r["Date"])
        for k,v in zip(H_cols.keys(), [hg,ga,gd,W,D,L,PPG]): H_cols[k].append(v)
        hg,ga,gd,W,D,L,PPG = hist(r["AwayTeam"],"A",r["Date"])
        for k,v in zip(A_cols.keys(), [hg,ga,gd,W,D,L,PPG]): A_cols[k].append(v)
    for k,v in H_cols.items(): out[k]=v
    for k,v in A_cols.items(): out[k]=v
    return out

# -- H2H (last 5) + rivalry flag (small set)
RIVAL_BASE = {
    ("Arsenal","Tottenham Hotspur"), ("Chelsea","Tottenham Hotspur"),
    ("Liverpool","Everton"), ("Liverpool","Manchester United"),
    ("Manchester City","Manchester United"), ("Crystal Palace","Brighton and Hove Albion"),
    ("Brentford","Fulham")
}
def add_h2h(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Date","HomeTeam","AwayTeam"]).copy()
    hist: Dict[Tuple[str,str], deque] = defaultdict(lambda: deque(maxlen=5))
    hH,hA,hD,rv = [],[],[],[]
    clubs = set(pd.unique(pd.concat([df["HomeTeam"], df["AwayTeam"]])))
    rivals = set()
    for a,b in RIVAL_BASE:
        if a in clubs and b in clubs: rivals.add((a,b)); rivals.add((b,a))
    for _, r in df.iterrows():
        h,a = r["HomeTeam"], r["AwayTeam"]
        key = tuple(sorted((h,a)))
        d = hist[key]
        if len(d)==0: hH.append(np.nan); hA.append(np.nan); hD.append(np.nan)
        else:
            wins_h = sum(1 for w in d if w == h)
            wins_a = sum(1 for w in d if w == a)
            draws  = sum(1 for w in d if w is None)
            tot = len(d)
            hH.append(wins_h/tot); hA.append(wins_a/tot); hD.append(draws/tot)
        rv.append(1 if (h,a) in rivals else 0)
        # update
        if   r["FTR"]=="H": d.append(h)
        elif r["FTR"]=="A": d.append(a)
        else:               d.append(None)
    df["H2H_H_win_rate"]=hH; df["H2H_A_win_rate"]=hA; df["H2H_draw_rate"]=hD; df["Rivalry"]=rv
    return df

# -- Rest days
def add_rest(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Date","HomeTeam","AwayTeam"]).copy()
    last = {}
    rH, rA = [], []
    for _, r in df.iterrows():
        d = r["Date"]; h=r["HomeTeam"]; a=r["AwayTeam"]
        rH.append((d-last[h]).days if h in last else 7.0)
        rA.append((d-last[a]).days if a in last else 7.0)
        last[h]=d; last[a]=d
    df["RestH"], df["RestA"] = rH, rA
    return df

# -- Money / managers
def _to_eur(x):
    if pd.isna(x): return np.nan
    s = str(x).replace("€","").replace(",","").strip().lower()
    m = re.match(r"([0-9.]+)\s*([mbk])?", s)
    if not m:
        try: return float(s)
        except: return np.nan
    num = float(m.group(1)); u = (m.group(2) or "")
    mult = 1.0 if not u else (1e9 if u=="b" else 1e6 if u=="m" else 1e3)
    return num*mult

def load_priors() -> dict:
    pri = {"mv_club":{}, "net_spend":{}, "mgr_days":{}}
    # club market values
    if MV_CLUBS.exists():
        df = pd.read_csv(MV_CLUBS)
        c_team = next((c for c in df.columns if re.search(r"club|team",c,re.I)), "team_name")
        c_val  = next((c for c in df.columns if re.search(r"value",c,re.I)), df.columns[-1])
        for _,r in df.iterrows():
            pri["mv_club"][canon(r[c_team])] = float(_to_eur(r[c_val]))
    # net spend
    if TRANSFERS.exists():
        df = pd.read_csv(TRANSFERS)
        c_team = next((c for c in df.columns if re.search(r"club|team",c,re.I)), "Team")
        # try to infer in/out columns; fallback to 'Net'
        money_like = [c for c in df.columns if df[c].astype(str).str.contains("€|,").mean() > 0.2]
        c_net = next((c for c in df.columns if re.search(r"net",c,re.I)), None)
        c_in  = next((c for c in df.columns if re.search(r"in(come)?|sold",c,re.I)), None)
        c_out = next((c for c in df.columns if re.search(r"expend|spent|purchase|out(?!put)",c,re.I)), None)
        for _, r in df.iterrows():
            t = canon(r[c_team])
            if c_net:
                pri["net_spend"][t] = float(_to_eur(r[c_net]))  # positive = net income
            else:
                inn = float(_to_eur(r[c_in])) if c_in in df.columns else 0.0
                out = float(_to_eur(r[c_out])) if c_out in df.columns else 0.0
                pri["net_spend"][t] = inn - out
    # manager tenure
    if PL_MANAGERS.exists():
        df = pd.read_csv(PL_MANAGERS)
        c_team = next((c for c in df.columns if re.search(r"team|club",c,re.I)), "Team")
        c_days = next((c for c in df.columns if re.search(r"tenure|days",c,re.I)), None)
        if c_days:
            for _, r in df.iterrows():
                pri["mgr_days"][canon(r[c_team])] = float(r[c_days])
    return pri

# ----------------------------- Build training frame -----------------------
@st.cache_data(show_spinner=False)
def load_training_bundle():
    raw = read_all_e0(CACHE)
    for c in ("HomeTeam","AwayTeam"): raw[c] = raw[c].map(canon)
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw = raw.dropna(subset=["FTR","FTHG","FTAG"])
    df = attach_elo(raw)
    df = add_last5(df, w=5)
    df = add_h2h(df)
    df = add_rest(df)

    pri = load_priors()
    mv = pri["mv_club"]; net = pri["net_spend"]; mgr = pri["mgr_days"]
    mv_med  = float(np.nanmedian(list(mv.values()))) if mv else 3e8
    net_med = float(np.nanmedian(list(net.values()))) if net else 0.0
    mgr_med = float(np.nanmedian(list(mgr.values()))) if mgr else 365.0

    def feat_diff(mapper, med):
        H = df["HomeTeam"].map(lambda t: mapper.get(t, med)).astype(float)
        A = df["AwayTeam"].map(lambda t: mapper.get(t, med)).astype(float)
        return (H - A).values

    df["mv_diff_log"]  = np.log1p(feat_diff(mv,  mv_med))
    df["net_diff"]     = feat_diff(net, net_med)
    df["mgr_diff"]     = feat_diff(mgr, mgr_med)

    y = df["FTR"].map({"A":0,"D":1,"H":2}).astype(int).values
    FEATS = [
        "EloHome_pre","EloAway_pre",
        "H_Last5_PPG","A_Last5_PPG",
        "H_rGF","H_rGA","H_rGD","H_rW","H_rD","H_rL",
        "A_rGF","A_rGA","A_rGD","A_rW","A_rD","A_rL",
        "H2H_H_win_rate","H2H_A_win_rate","H2H_draw_rate",
        "Rivalry","RestH","RestA",
        "mv_diff_log","net_diff","mgr_diff",
    ]
    X = df[FEATS].fillna(0.0).astype(float).reset_index(drop=True)

    # Time-ordered CV + calibration
    tscv = TimeSeriesSplit(n_splits=5)
    oof = np.zeros((len(X),3))
    for k,(tr,va) in enumerate(tscv.split(X), start=1):
        rf = RandomForestClassifier(
            n_estimators=650, min_samples_split=6, min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=42+k, n_jobs=-1
        )
        rf.fit(X.iloc[tr], y[tr])
        cal = CalibratedClassifierCV(rf, method="isotonic", cv=3)
        cal.fit(X.iloc[tr], y[tr])
        oof[va] = cal.predict_proba(X.iloc[va])
    acc = accuracy_score(y, np.argmax(oof,axis=1))
    ll  = log_loss(y, oof, labels=[0,1,2])

    # final model
    rf_final = RandomForestClassifier(
        n_estimators=800, min_samples_split=6, min_samples_leaf=2,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1
    ).fit(X, y)
    model = CalibratedClassifierCV(rf_final, method="isotonic", cv=3).fit(X, y)

    bundle = {"df": df, "X": X, "y": y, "model": model, "feats": FEATS,
              "pri": pri, "cv_acc": acc, "cv_logloss": ll}
    return bundle

bundle = load_training_bundle()

# ----------------------------- Fixtures & current table -------------------
@st.cache_data(show_spinner=False)
def load_fixtures():
    fx = pd.read_csv(FIXTURES)
    fx["date"] = pd.to_datetime(fx["date"], errors="coerce")
    for c in ("home","away"): fx[c] = fx[c].map(canon)
    if "played" not in fx.columns:
        fx["played"] = fx["FTR"].notna()
    return fx.sort_values("date")

@st.cache_data(show_spinner=False)
def load_current_table():
    if CURR_TAB.exists():
        tab = pd.read_csv(CURR_TAB)
        if "Team" in tab.columns: tab["Team"] = tab["Team"].map(canon)
        if "Pos" not in tab.columns:
            tab = tab.sort_values(["Pts","GD","GF"], ascending=[False,False,False]).reset_index(drop=True)
            tab["Pos"] = np.arange(1, len(tab)+1)
        return tab[["Pos","Team","Pl","W","D","L","GF","GA","GD","Pts"]]
    # fallback build from fixtures if not provided
    fx = load_fixtures()
    played = fx[fx["played"]].rename(columns={"home":"HomeTeam","away":"AwayTeam"})
    if {"HomeTeam","AwayTeam","FTR","FTHG","FTAG"} <= set(played.columns):
        return build_table_from_results(played[["HomeTeam","AwayTeam","FTR","FTHG","FTAG"]])
    return pd.DataFrame()

fixtures = load_fixtures()
table_now = load_current_table()

def next_md_number(fx: pd.DataFrame) -> int:
    if "md" in fx.columns:
        done = fx[fx["played"]]["md"].max()
        return int(done) + 1 if not np.isnan(done) else int(fx["md"].min())
    # if no md column, guess by date quartiles (rare)
    return int((fx["played"].sum() // 10) + 1)

# ----------------------------- Feature builder for predictions ------------
def elo_now_from_history(history: pd.DataFrame) -> Dict[str,float]:
    hist = history.sort_values("Date")
    teams = pd.unique(pd.concat([hist["HomeTeam"],hist["AwayTeam"]]))
    R={t:1500.0 for t in teams}
    for _, r in hist.iterrows():
        sH = 1.0 if r["FTR"]=="H" else (0.5 if r["FTR"]=="D" else 0.0)
        try: gd = int(abs(float(r["FTHG"])-float(r["FTAG"])))
        except: gd = None
        R[r["HomeTeam"]],R[r["AwayTeam"]] = elo_update(R[r["HomeTeam"]], R[r["AwayTeam"]], sH, gd=gd)
    return R

def make_features_for_list(games: pd.DataFrame) -> pd.DataFrame:
    hist = bundle["df"][["Date","HomeTeam","AwayTeam","FTR","FTHG","FTAG"]]
    R = elo_now_from_history(hist)
    rows = []
    for _, m in games.iterrows():
        d = pd.to_datetime(m["date"]); h=m["home"]; a=m["away"]
        # last-5 side
        def last5(team, side):
            hst = hist[hist["Date"]<d]
            if side=="H":
                ss = hst[hst["HomeTeam"]==team].tail(5)
                if ss.empty: return 0,0,0,0,0,0,0
                GF,GA = ss["FTHG"].values, ss["FTAG"].values
                W = (ss["FTR"]=="H").mean(); D = (ss["FTR"]=="D").mean(); L = (ss["FTR"]=="A").mean()
            else:
                ss = hst[hst["AwayTeam"]==team].tail(5)
                if ss.empty: return 0,0,0,0,0,0,0
                GF,GA = ss["FTAG"].values, ss["FTHG"].values
                W = (ss["FTR"]=="A").mean(); D = (ss["FTR"]=="D").mean(); L = (ss["FTR"]=="H").mean()
            PPG = (3*(W*5)+1*(D*5))/5.0
            return GF.mean(), GA.mean(), (GF-GA).mean(), W, D, L, PPG

        Hg,Ha,Hgd,Hw,Hd,Hl,Hppg = last5(h,"H")
        Ag,Aa,Agd,Aw,Ad,Al,Appg = last5(a,"A")

        # h2h rates
        H2H = hist[((hist["HomeTeam"]==h)&(hist["AwayTeam"]==a))|
                   ((hist["HomeTeam"]==a)&(hist["AwayTeam"]==h))].sort_values("Date").tail(5)
        if H2H.empty:
            hH=hA=hD=np.nan
        else:
            hH = ((H2H["HomeTeam"]==h) & (H2H["FTR"]=="H")).sum() + ((H2H["AwayTeam"]==h)&(H2H["FTR"]=="A")).sum()
            hA = ((H2H["HomeTeam"]==a) & (H2H["FTR"]=="H")).sum() + ((H2H["AwayTeam"]==a)&(H2H["FTR"]=="A")).sum()
            hD = (H2H["FTR"]=="D").sum()
            tot=len(H2H); hH/=tot; hA/=tot; hD/=tot

        # rest
        def rest(team):
            hst = hist[(hist["HomeTeam"]==team)|(hist["AwayTeam"]==team)]
            if hst.empty: return 7.0
            last = hst["Date"].max()
            return float((d - last).days)
        RestH, RestA = rest(h), rest(a)

        # money priors
        pri = bundle["pri"]
        mvH = math.log1p(pri["mv_club"].get(h, np.nanmedian(list(pri["mv_club"].values()) or [3e8])))
        mvA = math.log1p(pri["mv_club"].get(a, np.nanmedian(list(pri["mv_club"].values()) or [3e8])))
        netH = pri["net_spend"].get(h, 0.0); netA = pri["net_spend"].get(a, 0.0)
        mgrH = pri["mgr_days"].get(h, 365.0);  mgrA = pri["mgr_days"].get(a, 365.0)

        rows.append({
            "date": d, "home": h, "away": a,
            "EloHome_pre": R.get(h,1500.0), "EloAway_pre": R.get(a,1500.0),
            "H_Last5_PPG": Hppg, "A_Last5_PPG": Appg,
            "H_rGF":Hg,"H_rGA":Ha,"H_rGD":Hgd,"H_rW":Hw,"H_rD":Hd,"H_rL":Hl,
            "A_rGF":Ag,"A_rGA":Aa,"A_rGD":Agd,"A_rW":Aw,"A_rD":Ad,"A_rL":Al,
            "H2H_H_win_rate":hH,"H2H_A_win_rate":hA,"H2H_draw_rate":hD,
            "Rivalry":0,"RestH":RestH,"RestA":RestA,
            "mv_diff_log": (mvH - mvA), "net_diff": (netH - netA), "mgr_diff": (mgrH - mgrA),
        })
    return pd.DataFrame(rows)

def badge_for_position(pos: int) -> str:
    if pos <= 5: return "UCL"
    if pos <= 7: return "UEL"
    if pos == 8: return "UECL"
    if pos >= 18: return "Relegation"
    return ""

def style_picks(row):
    # green bold winner, red bold loser, italic draw
    if row["Pick"] == "Home":
        return f"<span style='font-weight:700;color:#16a34a'>{row['home']}</span> vs <span style='color:#ef4444;font-weight:700'>{row['away']}</span>"
    elif row["Pick"] == "Away":
        return f"<span style='color:#ef4444;font-weight:700'>{row['home']}</span> vs <span style='font-weight:700;color:#16a34a'>{row['away']}</span>"
    else:
        return f"<i>{row['home']} vs {row['away']}</i>"

# ----------------------------- UI ----------------------------------------
st.markdown("""
# 🏟️ **Premier League — Latest Matchday Predictions & Season Projection**
Random-Forest model trained on historical E0 seasons with **Elo + form + H2H + rest + money priors**.  
_No files are written; everything is displayed live._
""")

tab_curr, tab_next, tab_md38, tab_money = st.tabs(
    ["📊 Current Table", "🧮 Next Matchday", "📈 MD38 Projection", "💶 Money & Squads"]
)

# ---------- Current Table
with tab_curr:
    st.subheader(f"Current Table{' ' if table_now.empty else f' (after MD {int(fixtures[fixtures.played].md.max()) if 'md' in fixtures.columns else ''})'}")
    if table_now.empty:
        st.info("No current_table.csv detected and not enough played fixtures to reconstruct the table.")
    else:
        show = table_now.copy()
        show["Badge"] = [badge_for_position(p) for p in show["Pos"]]
        st.dataframe(show, use_container_width=True, hide_index=True)
        # charts
        col1,col2,col3 = st.columns(3)
        with col1:
            st.caption("Goal Difference")
            fig = px.bar(show.sort_values("GD",ascending=False), x="Team", y="GD", template="plotly_white",
                         color="GD", color_continuous_scale=px.colors.sequential.Blues)
            fig.update_layout(height=320, margin=dict(l=10,r=10,t=10,b=10), xaxis=dict(tickangle=60))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.caption("Goals Scored")
            fig = px.bar(show.sort_values("GF",ascending=False), x="Team", y="GF", template="plotly_white",
                         color="GF", color_continuous_scale=px.colors.sequential.Greens)
            fig.update_layout(height=320, margin=dict(l=10,r=10,t=10,b=10), xaxis=dict(tickangle=60))
            st.plotly_chart(fig, use_container_width=True)
        with col3:
            st.caption("Goals Conceded")
            fig = px.bar(show.sort_values("GA",ascending=True), x="Team", y="GA", template="plotly_white",
                         color="GA", color_continuous_scale=px.colors.sequential.Reds)
            fig.update_layout(height=320, margin=dict(l=10,r=10,t=10,b=10), xaxis=dict(tickangle=60))
            st.plotly_chart(fig, use_container_width=True)

# ---------- Next Matchday
with tab_next:
    md = next_md_number(fixtures)
    st.subheader(f"Next Matchday: **MD {md}**")
    md_games = fixtures[(fixtures["md"]==md) if "md" in fixtures.columns else ~fixtures["played"]].copy()
    if md_games.empty:
        st.info("No upcoming matchday detected.")
    else:
        F = make_features_for_list(md_games)
        Xp = F[bundle["feats"]].fillna(0.0).astype(float)
        proba = bundle["model"].predict_proba(Xp.values)
        out = F[["date","home","away"]].copy()
        out["P_home"], out["P_draw"], out["P_away"] = proba[:,2], proba[:,1], proba[:,0]
        out["PickIdx"] = np.argmax(proba, axis=1)
        out["Pick"] = out["PickIdx"].map({2:"Home",1:"Draw",0:"Away"})

        # pretty matchup line w/ color rules
        out["Match"] = out.apply(style_picks, axis=1)
        disp = out[["date","Match","P_home","P_draw","P_away","Pick"]].sort_values("date").reset_index(drop=True)
        # nicer formatting
        disp["P_home"] = disp["P_home"].map(lambda x:f"{x:.3f}")
        disp["P_draw"] = disp["P_draw"].map(lambda x:f"{x:.3f}")
        disp["P_away"] = disp["P_away"].map(lambda x:f"{x:.3f}")

        st.markdown("**Model probabilities (Random Forest)**")
        st.write(disp.to_html(escape=False, index=False), unsafe_allow_html=True)

        # probability bars per match
        st.markdown("**Per-match probability bars**")
        for _, r in out.sort_values("date").iterrows():
            fig = go.Figure(data=[
                go.Bar(name="Home", x=["Home"], y=[r.P_home]),
                go.Bar(name="Draw", x=["Draw"], y=[r.P_draw]),
                go.Bar(name="Away", x=["Away"], y=[r.P_away]),
            ])
            fig.update_layout(barmode="group", height=220, template="plotly_white",
                              title=f"{r['home']} vs {r['away']} — pick: {r['Pick']}")
            st.plotly_chart(fig, use_container_width=True)

# ---------- MD38 Projection
with tab_md38:
    st.subheader("Projection to MD38 (Expected Points)")
    # expected points for remaining fixtures
    played = fixtures[fixtures["played"]].rename(columns={"home":"HomeTeam","away":"AwayTeam"})
    hist_now = played[["HomeTeam","AwayTeam","FTR","FTHG","FTAG"]] if not played.empty else pd.DataFrame()
    base_tab = build_table_from_results(hist_now) if not hist_now.empty else \
        pd.DataFrame({"Team": sorted(set(pd.unique(pd.concat([fixtures["home"],fixtures["away"]]).map(canon)))), 
                      "Pl":[0]*20,"W":[0]*20,"D":[0]*20,"L":[0]*20,"GF":[0]*20,"GA":[0]*20,"GD":[0]*20,"Pts":[0]*20})
    remain = fixtures[~fixtures["played"]][["date","home","away"]]
    if remain.empty:
        st.info("No remaining fixtures to project.")
    else:
        F = make_features_for_list(remain)
        Xr = F[bundle["feats"]].fillna(0.0).astype(float)
        P = bundle["model"].predict_proba(Xr.values)     # [:,0]=A, [:,1]=D, [:,2]=H
        scored = pd.DataFrame({"home":F["home"],"away":F["away"],
                               "P_home":P[:,2], "P_draw":P[:,1], "P_away":P[:,0]})
        teams = pd.unique(pd.concat([scored["home"], scored["away"]]))
        agg = {t: {"ExpPts":0.0} for t in teams}
        for _, r in scored.iterrows():
            agg[r["home"]]["ExpPts"] += 3*r["P_home"] + 1*r["P_draw"]
            agg[r["away"]]["ExpPts"] += 3*r["P_away"] + 1*r["P_draw"]

        exp = pd.DataFrame([{"Team":t, **v} for t,v in agg.items()])
        final = base_tab.merge(exp, on="Team", how="left").fillna({"ExpPts":0.0})
        final["Pts_pred"] = final["Pts"] + final["ExpPts"]
        final = final.sort_values("Pts_pred", ascending=False).reset_index(drop=True)
        final.insert(0, "ExpPos", np.arange(1, len(final)+1))
        final["Pts_pred_round"] = final["Pts_pred"].round().astype(int)
        final["Badge"] = [badge_for_position(p) for p in final["ExpPos"]]

        champ = final.iloc[0]["Team"] if not final.empty else "—"
        st.success(f"🏆 **Projected Champion:** {champ}")

        st.dataframe(final[["ExpPos","Team","Pts_pred_round","Pts_pred","Badge"]],
                     use_container_width=True, hide_index=True)

        # scatter of Pts_pred vs squad value
        mv = bundle["pri"]["mv_club"]
        final["SquadValue€"] = final["Team"].map(lambda t: mv.get(t, np.nan))
        fig = px.scatter(final, x="SquadValue€", y="Pts_pred",
                         text="Team", size="SquadValue€", template="plotly_white",
                         color="Badge", color_discrete_map={"UCL":"#2563eb","UEL":"#7c3aed","UECL":"#0ea5e9","Relegation":"#ef4444"})
        fig.update_traces(textposition="top center")
        fig.update_layout(height=520, title="Projected Points vs Squad Value")
        st.plotly_chart(fig, use_container_width=True)

# ---------- Money & Squads
with tab_money:
    st.subheader("Money & Squads")
    pri = bundle["pri"]
    # Treemap: squad value
    sv = pd.DataFrame({"Team": list(pri["mv_club"].keys()),
                       "Value": list(pri["mv_club"].values())})
    if not sv.empty:
        fig = px.treemap(sv.sort_values("Value",ascending=False),
                         path=["Team"], values="Value",
                         color="Value", color_continuous_scale=px.colors.sequential.Tealgrn,
                         template="plotly_white", title="Squad Market Values (€)")
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)

    # Net spend bars (positive = net income)
    ns = pd.DataFrame({"Team": list(pri["net_spend"].keys()),
                       "Net": list(pri["net_spend"].values())})
    if not ns.empty:
        ns = ns.sort_values("Net", ascending=False)
        fig = px.bar(ns, x="Team", y="Net", template="plotly_white",
                     color="Net", color_continuous_scale=px.colors.diverging.RdYlGn)
        fig.update_layout(height=420, xaxis=dict(tickangle=60),
                          title="Net Transfer (Income - Expenditure)")
        st.plotly_chart(fig, use_container_width=True)

    # Manager tenure
    if pri["mgr_days"]:
        mg = pd.DataFrame({"Team": list(pri["mgr_days"].keys()),
                           "TenureDays": list(pri["mgr_days"].values())}).sort_values("TenureDays", ascending=False)
        fig = px.bar(mg, x="Team", y="TenureDays", template="plotly_white",
                     color="TenureDays", color_continuous_scale=px.colors.sequential.Magma)
        fig.update_layout(height=420, xaxis=dict(tickangle=60),
                          title="Manager Tenure (days)")
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------- Footer / meta ------------------------------
st.markdown(
    f"<div style='color:#64748b;font-size:12px;padding-top:16px'>"
    f"RF CV Accuracy (time-ordered 5-fold): <b>{bundle['cv_acc']:.3f}</b> · "
    f"LogLoss: <b>{bundle['cv_logloss']:.3f}</b>"
    f"</div>", unsafe_allow_html=True
)
