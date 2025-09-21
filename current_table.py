#!/usr/bin/env python3
# Actual_table.py
# Build live Premier League table + detect current Matchday from fixtures_2025_26.csv

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
FIXTURES_PATH = REPO_ROOT / "fixtures_2025_26.csv"
OUT_TABLE_PATH = REPO_ROOT / "current_table_2025_26.csv"

# --- Team aliases (keep in sync with your modeling code) ----------------------
TEAM_ALIASES: Dict[str, str] = {
    "Man United": "Manchester United",
    "Man Utd": "Manchester United",
    "Man City": "Manchester City",
    "Tottenham": "Tottenham Hotspur",
    "Spurs": "Tottenham Hotspur",
    "Nott'm Forest": "Nottingham Forest",
    "Wolves": "Wolverhampton Wanderers",
    "West Brom": "West Bromwich Albion",
    "Brighton": "Brighton and Hove Albion",
    "Newcastle": "Newcastle United",
    "Leeds": "Leeds United",
    "Bournemouth": "AFC Bournemouth",
    "West Ham": "West Ham United",
    "Sheffield Utd": "Sheffield United",
    # idempotent / common canonical forms
    "Sheffield United": "Sheffield United",
}

def canon_team(name: str) -> str:
    if pd.isna(name):
        return name
    s = str(name).strip()
    return TEAM_ALIASES.get(s, s)

# --- Helpers ------------------------------------------------------------------
def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        hit = next((col for col in df.columns if col.lower() == c.lower()), None)
        if hit:
            return hit
    return None

def _parse_score_as_goals(txt: str) -> Optional[Tuple[int, int]]:
    if not isinstance(txt, str):
        return None
    m = re.match(r"^\s*(\d+)\s*[-:]\s*(\d+)\s*$", txt)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))

def _infer_played(row: pd.Series, ftr_col: Optional[str], hg_col: Optional[str],
                  ag_col: Optional[str], score_col: Optional[str]) -> bool:
    if "played" in row.index and pd.notna(row["played"]):
        try:
            return bool(row["played"])
        except Exception:
            pass
    if hg_col and ag_col and pd.notna(row.get(hg_col)) and pd.notna(row.get(ag_col)):
        return True
    if ftr_col and pd.notna(row.get(ftr_col)):
        return True
    if score_col and pd.notna(row.get(score_col)) and _parse_score_as_goals(str(row[score_col])):
        return True
    return False

def _md_column(df: pd.DataFrame) -> Optional[str]:
    return _find_col(df, ["md", "matchday", "gw", "round"])

def _side_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    home = _find_col(df, ["home", "hometeam", "home_team", "home team"])
    away = _find_col(df, ["away", "awayteam", "away_team", "away team"])
    return home, away

def _goals_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    fthg = _find_col(df, ["fthg", "home_goals", "homegoals", "hg"])
    ftag = _find_col(df, ["ftag", "away_goals", "awaygoals", "ag"])
    score = _find_col(df, ["score", "ft", "final"])
    return fthg, ftag, score

def _result_col(df: pd.DataFrame) -> Optional[str]:
    return _find_col(df, ["ftr", "result"])

def _date_col(df: pd.DataFrame) -> Optional[str]:
    return _find_col(df, ["date", "matchdate", "kickoff", "ko"])

# --- Core logic ---------------------------------------------------------------
def build_current_table(fixtures_csv: Path = FIXTURES_PATH) -> Tuple[pd.DataFrame, Optional[int]]:
    if not fixtures_csv.exists():
        raise FileNotFoundError(f"Missing {fixtures_csv}")

    fx = pd.read_csv(fixtures_csv)
    # identify key columns
    home_col, away_col = _side_cols(fx)
    if not home_col or not away_col:
        raise ValueError("Could not find home/away columns in fixtures CSV.")

    ftr_col = _result_col(fx)
    fthg_col, ftag_col, score_col = _goals_cols(fx)
    date_col = _date_col(fx)
    md_col = _md_column(fx)

    # normalize teams
    fx[home_col] = fx[home_col].map(canon_team)
    fx[away_col] = fx[away_col].map(canon_team)

    # parse dates if present
    if date_col:
        fx[date_col] = pd.to_datetime(fx[date_col], errors="coerce")

    # infer 'played'
    fx["__played__"] = fx.apply(lambda r: _infer_played(r, ftr_col, fthg_col, ftag_col, score_col), axis=1)

    # detect current MD: max MD that has ANY played match
    cur_md = None
    if md_col:
        md_numeric = pd.to_numeric(fx[md_col], errors="coerce")
        if md_numeric.notna().any():
            played_mds = md_numeric[fx["__played__"]]
            if played_mds.notna().any():
                cur_md = int(played_mds.max())

    # use only played matches for the table
    played = fx[fx["__played__"]].copy()
    if played.empty:
        # empty table if no matches played yet
        cols = ["Pos", "Team", "Pl", "W", "D", "L", "GF", "GA", "GD", "Pts"]
        return pd.DataFrame(columns=cols), cur_md

    # ensure we have goals; if not, try to parse from Score
    if (not fthg_col or not ftag_col) and score_col:
        gh, ga = [], []
        for val in played[score_col].astype(str).tolist():
            parsed = _parse_score_as_goals(val)
            if parsed:
                gh.append(parsed[0]); ga.append(parsed[1])
            else:
                gh.append(np.nan); ga.append(np.nan)
        played["__FTHG__"] = gh
        played["__FTAG__"] = ga
        fthg_col, ftag_col = "__FTHG__", "__FTAG__"

    # if still missing goals for some rows, we can’t compute GF/GA properly → drop those rows from GF/GA,
    # but still compute points if FTR exists.
    have_goals = fthg_col and ftag_col
    if have_goals:
        # rows with both goals present
        gmask = played[fthg_col].notna() & played[ftag_col].notna()
    else:
        gmask = pd.Series(False, index=played.index)

    # Build per-match contributions
    rows = []

    # Points via FTR (fallback if goals missing)
    def _pts_from_ftr(side: str, ftr: str) -> int:
        if ftr == "D":
            return 1
        if (ftr == "H" and side == "H") or (ftr == "A" and side == "A"):
            return 3
        return 0

    for _, r in played.iterrows():
        h, a = r[home_col], r[away_col]

        # determine goals if available
        if have_goals and pd.notna(r.get(fthg_col)) and pd.notna(r.get(ftag_col)):
            hg, ag = int(r[fthg_col]), int(r[ftag_col])
        else:
            # Without goals, set 0s for GF/GA but keep points via FTR (safer than guessing).
            hg, ag = 0, 0

        # determine result (prefer explicit FTR; else derive from goals if available)
        ftr = r.get(ftr_col) if ftr_col in played.columns else None
        if pd.isna(ftr) and have_goals and (hg is not None) and (ag is not None):
            ftr = "H" if hg > ag else ("A" if ag > hg else "D")

        if pd.isna(ftr):
            # cannot assign points without a result; skip this row
            continue

        # home contribution
        rows.append({
            "Team": h, "Pl": 1,
            "W": 1 if ftr == "H" else 0,
            "D": 1 if ftr == "D" else 0,
            "L": 1 if ftr == "A" else 0,
            "GF": hg, "GA": ag,
            "Pts": _pts_from_ftr("H", ftr)
        })
        # away contribution
        rows.append({
            "Team": a, "Pl": 1,
            "W": 1 if ftr == "A" else 0,
            "D": 1 if ftr == "D" else 0,
            "L": 1 if ftr == "H" else 0,
            "GF": ag, "GA": hg,
            "Pts": _pts_from_ftr("A", ftr)
        })

    if not rows:
        cols = ["Pos", "Team", "Pl", "W", "D", "L", "GF", "GA", "GD", "Pts"]
        return pd.DataFrame(columns=cols), cur_md

    tab = pd.DataFrame(rows).groupby("Team", as_index=False).sum(numeric_only=True)
    tab["GD"] = tab["GF"] - tab["GA"]
    tab["Pts"] = 3 * tab["W"] + tab["D"]  # ensure Pts consistency

    tab = tab.sort_values(["Pts", "GD", "GF"], ascending=[False, False, False]).reset_index(drop=True)
    tab.insert(0, "Pos", np.arange(1, len(tab) + 1))

    # ensure standard column order
    tab = tab[["Pos", "Team", "Pl", "W", "D", "L", "GF", "GA", "GD", "Pts"]]
    return tab, cur_md

def main() -> None:
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 200)

    table, md = build_current_table(FIXTURES_PATH)

    # Save
    table.to_csv(OUT_TABLE_PATH, index=False)

    # Print summary
    print("✔ Saved", OUT_TABLE_PATH.name)
    if md is not None:
        print(f"Current Matchday (detected): MD {md}")
    else:
        print("Current Matchday (detected): unavailable (no played matches with MD found)")

    print("\n=== Current Premier League Table ===")
    if table.empty:
        print("(no played matches yet)")
    else:
        # show first 20 rows (entire league)
        with pd.option_context("display.max_rows", 50):
            print(table.to_string(index=False))

if __name__ == "__main__":
    main()
