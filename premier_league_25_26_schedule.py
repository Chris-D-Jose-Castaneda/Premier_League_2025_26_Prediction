#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PL 2025/26 results merger (button-driven navigation + skip-when-filled).

Fixes:
  - Avoids ValueError when a game's score is NaN (unfinished).
  - Only navigates to matchweeks that still need results AND should have started
    (min scheduled date <= today). Prevents endless back-clicking once MW1–4 are saved.

Usage:
  python premier_league_25_26_schedule.py --csv fixtures_2025_26.csv --from-md 1 --to-md 38
  # optional:
  #   --no-headless  --sleep-min 0.8 --sleep-max 2.0
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
import time
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.firefox.options import Options as FFOptions
from selenium.webdriver.firefox.service import Service as FFService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC  # noqa: N813
from webdriver_manager.firefox import GeckoDriverManager

# ----------------------------- Config ---------------------------------

SEASON_QS = "competition=8&season=2025"
SEED_URLS = [
    f"https://www.premierleague.com/en/matches?{SEASON_QS}&matchweek=5&month=09",  # stable render
    f"https://www.premierleague.com/en/matches?{SEASON_QS}&matchweek=2&month=09",
    f"https://www.premierleague.com/en/matches?{SEASON_QS}",
    "https://www.premierleague.com/en/matches",
]

SCORE_RE = re.compile(r"^\s*(\d+)\s*[-–]\s*(\d+)\s*$")

TEAM_ALIASES: Dict[str, str] = {
    "Man United": "Manchester United",
    "Man Utd": "Manchester United",
    "Man City": "Manchester City",
    "Tottenham": "Tottenham Hotspur",
    "Spurs": "Tottenham Hotspur",
    "Nott'm Forest": "Nottingham Forest",
    "Nott'm": "Nottingham Forest",
    "Wolves": "Wolverhampton Wanderers",
    "West Ham": "West Ham United",
    "Brighton": "Brighton and Hove Albion",
    "Brighton & Hove Albion": "Brighton and Hove Albion",
    "Newcastle": "Newcastle United",
    "Leeds": "Leeds United",
    "Leeds Utd": "Leeds United",
    "Bournemouth": "AFC Bournemouth",
    "Liverpool FC": "Liverpool",
    "Tottenham Hotspur Hotspur": "Tottenham Hotspur",
    # identities
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "AFC Bournemouth": "AFC Bournemouth",
    "Brentford": "Brentford",
    "Burnley": "Burnley",
    "Chelsea": "Chelsea",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Liverpool": "Liverpool",
    "Manchester City": "Manchester City",
    "Manchester United": "Manchester United",
    "Newcastle United": "Newcastle United",
    "Nottingham Forest": "Nottingham Forest",
    "Leeds United": "Leeds United",
    "Sunderland": "Sunderland",
    "Tottenham Hotspur": "Tottenham Hotspur",
    "West Ham United": "West Ham United",
    "Wolverhampton Wanderers": "Wolverhampton Wanderers",
}
WEB_ALIAS = {
    "Liverpool FC": "Liverpool",
    "Brighton & Hove Albion": "Brighton and Hove Albion",
    "Newcastle Utd": "Newcastle United",
    "Leeds Utd": "Leeds United",
    "Tottenham": "Tottenham Hotspur",
    "Tottenham Hotspur Hotspur": "Tottenham Hotspur",
}


def normalize_team(name: str) -> str:
    s = re.sub(r"\s+", " ", str(name).strip())
    s = WEB_ALIAS.get(s, s)
    return TEAM_ALIASES.get(s, s)


# ----------------------------- Selenium ---------------------------------

def make_driver(headless: bool, timeout: int = 35) -> webdriver.Firefox:
    opts = FFOptions()
    opts.headless = headless
    # faster: limit heavy assets
    opts.set_preference("permissions.default.image", 2)
    opts.set_preference("dom.ipc.processCount", 1)
    service = FFService(GeckoDriverManager().install())
    driver = webdriver.Firefox(service=service, options=opts)
    driver.set_page_load_timeout(timeout)
    return driver


def snooze(a: float, b: float) -> None:
    time.sleep(random.uniform(max(0.0, a), max(a, b)))


def accept_cookies_if_present(driver: webdriver.Firefox) -> None:
    for sel in (
        "button[aria-label*='Accept']",
        "button[aria-label*='accept']",
        "button[title*='Accept']",
    ):
        try:
            WebDriverWait(driver, 3).until(EC.element_to_be_clickable((By.CSS_SELECTOR, sel))).click()
            break
        except Exception:
            pass


def open_seed(driver: webdriver.Firefox, sleep_min: float, sleep_max: float) -> Optional[BeautifulSoup]:
    for url in SEED_URLS:
        try:
            driver.get(url)
        except Exception:
            continue
        snooze(sleep_min, sleep_max)
        accept_cookies_if_present(driver)
        try:
            WebDriverWait(driver, 25).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.match-list"))
            )
            return BeautifulSoup(driver.page_source, "html.parser")
        except Exception:
            continue
    return None


def read_header_md(soup: BeautifulSoup) -> Optional[int]:
    hd = soup.select_one(".match-list-header__title")
    if not hd:
        return None
    m = re.search(r"Matchweek\s*(\d+)", hd.get_text(" ", strip=True), re.IGNORECASE)
    return int(m.group(1)) if m else None


def click_nav(driver: webdriver.Firefox, direction: str) -> bool:
    # prev is left button, next is right button
    btns = driver.find_elements(By.CSS_SELECTOR, ".match-list-header__button-container .match-list-header__btn")
    if not btns or len(btns) < 2:
        return False
    btn = btns[0] if direction == "prev" else btns[-1]
    try:
        driver.execute_script("arguments[0].click();", btn)
        return True
    except Exception:
        try:
            btn.click()
            return True
        except Exception:
            return False


def wait_dom(driver: webdriver.Firefox, sleep_min: float, sleep_max: float) -> None:
    snooze(sleep_min * 0.6, sleep_max * 0.9)
    WebDriverWait(driver, 25).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div.match-list"))
    )


def navigate_to_md(driver: webdriver.Firefox, current_md: Optional[int], target_md: int,
                   sleep_min: float, sleep_max: float) -> Tuple[Optional[BeautifulSoup], Optional[int]]:
    """Drive the page via Previous/Next until header shows target_md."""
    limit = 60
    steps = 0
    soup = BeautifulSoup(driver.page_source, "html.parser")
    header = current_md if current_md is not None else read_header_md(soup)

    while header != target_md and steps < limit:
        direction = "prev" if (header is None or header > target_md) else "next"
        if not click_nav(driver, direction):
            break
        wait_dom(driver, sleep_min, sleep_max)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        header = read_header_md(soup)
        steps += 1

    return (soup, header)


def scrape_current_page(soup: BeautifulSoup, md: int) -> pd.DataFrame:
    rows = []
    for card in soup.select("a[data-testid='matchCard']"):
        h_el = card.select_one(".match-card__team--home [data-testid='matchCardTeamFullName']")
        a_el = card.select_one(".match-card__team--away [data-testid='matchCardTeamFullName']")
        if not h_el or not a_el:
            continue
        home = normalize_team(h_el.get_text(strip=True))
        away = normalize_team(a_el.get_text(strip=True))

        score_el = card.select_one("span[data-testid='matchCardScore']")
        status_el = card.select_one(".match-card__full-time")
        status = status_el.get_text(strip=True) if status_el else ""

        FTHG = FTAG = None
        if score_el:
            m = SCORE_RE.match(score_el.get_text(strip=True))
            if m:
                FTHG, FTAG = int(m.group(1)), int(m.group(2))

        rows.append({"md": int(md), "home": home, "away": away, "FTHG": FTHG, "FTAG": FTAG, "status": status})
    return pd.DataFrame(rows)


# ----------------------------- Merge + Plan ---------------------------------

def ensure_result_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    defaults = {
        "FTHG": pd.NA,
        "FTAG": pd.NA,
        "FTR": pd.NA,
        "played": False,
        "status": "",
        "ResultStr": "",
    }
    for c, default in defaults.items():
        if c not in out.columns:
            out[c] = default
    return out


def compute_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    def _ftr(row):
        if pd.isna(row["FTHG"]) or pd.isna(row["FTAG"]):
            return pd.NA
        return "H" if row["FTHG"] > row["FTAG"] else ("A" if row["FTAG"] > row["FTHG"] else "D")

    df["FTR"] = df.apply(_ftr, axis=1)
    df["played"] = df["FTR"].notna()
    df["ResultStr"] = df.apply(
        lambda r: f"{int(r['FTHG'])}-{int(r['FTAG'])} ({r['FTR']})" if r["played"] else "",
        axis=1,
    )
    return df


def merge_md_results(csv_path: str, md: int, md_df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    fx = pd.read_csv(csv_path)
    fx = ensure_result_columns(fx)
    fx["home"] = fx["home"].map(normalize_team)
    fx["away"] = fx["away"].map(normalize_team)

    mask = fx["md"].astype(int) == int(md)
    if mask.sum() == 0 or md_df.empty:
        fx.to_csv(csv_path, index=False)
        return fx, 0

    md_df = md_df.copy()
    md_df["home"] = md_df["home"].map(normalize_team)
    md_df["away"] = md_df["away"].map(normalize_team)

    # Map for quick lookups
    rmap = {(r.home, r.away): r for r in md_df.itertuples(index=False)}

    merged_scores = 0
    for idx in fx.index[mask]:
        tup = (fx.at[idx, "home"], fx.at[idx, "away"])
        r = rmap.get(tup)
        if not r:
            continue

        # Only set scores if they exist (avoid NaN->int crash)
        if pd.notna(r.FTHG) and pd.notna(r.FTAG):
            fx.at[idx, "FTHG"] = int(r.FTHG)
            fx.at[idx, "FTAG"] = int(r.FTAG)
            merged_scores += 1

        # Update status if present
        if isinstance(r.status, str) and r.status:
            fx.at[idx, "status"] = r.status

    fx = compute_outcomes(fx)
    fx.to_csv(csv_path, index=False)
    return fx, merged_scores


def plan_matchweeks_to_fetch(csv_path: str, md_from: int, md_to: int) -> List[int]:
    """
    Return the ordered list of MDs that still need results (some FTHG/FTAG missing)
    and whose min scheduled date is <= today. This stops us from clicking back
    when earlier weeks are already fully populated.
    """
    fx = pd.read_csv(csv_path)
    fx = ensure_result_columns(fx)
    fx["md"] = fx["md"].astype(int)

    # Today in local date
    today = date.today()

    needs: List[int] = []
    for md in range(md_from, md_to + 1):
        slice_md = fx.loc[fx["md"] == md]
        if slice_md.empty:
            continue

        # Only attempt MDs whose earliest scheduled date has arrived
        # (so we don't scrape future weeks)
        try:
            dmin = pd.to_datetime(slice_md["date"], errors="coerce").min()
        except Exception:
            dmin = None
        if pd.isna(dmin) or dmin.date() > today:
            continue

        # If ALL 10 have scores, skip
        if len(slice_md) == 10 and slice_md["FTHG"].notna().all() and slice_md["FTAG"].notna().all():
            continue

        needs.append(md)

    return needs


# ----------------------------- CLI ---------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Merge PL results into fixtures CSV (button navigation, skip when filled).")
    ap.add_argument("--csv", default="fixtures_2025_26.csv", help="Fixtures CSV with columns: date, home, away, md")
    ap.add_argument("--from-md", type=int, default=1)
    ap.add_argument("--to-md", type=int, default=38)
    ap.add_argument("--sleep-min", type=float, default=0.8)
    ap.add_argument("--sleep-max", type=float, default=2.0)
    ap.add_argument("--no-headless", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        print(f"[ERROR] CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(2)

    fx = pd.read_csv(args.csv)
    need_cols = {"date", "home", "away", "md"}
    if not need_cols.issubset(fx.columns):
        print(f"[ERROR] CSV missing {need_cols}. Found: {list(fx.columns)}", file=sys.stderr)
        sys.exit(2)

    # Ensure result columns exist from the start
    ensure_result_columns(fx).to_csv(args.csv, index=False)

    md_first = max(1, int(args.from_md))
    md_last = min(38, int(args.to_md))

    # PLAN: which MDs actually need scraping?
    target_mds = plan_matchweeks_to_fetch(args.csv, md_first, md_last)
    if not target_mds:
        print(f"✔ Nothing to do: all MDs {md_first}–{md_last} that have started are already populated.")
        return

    print(f"✔ Will scrape MDs: {target_mds}, CSV={args.csv}")

    headless = not args.no_headless
    sleep_min = max(0.0, args.sleep_min)
    sleep_max = max(sleep_min, args.sleep_max)

    driver = make_driver(headless=headless)
    try:
        # 1) Load seed page once
        soup = open_seed(driver, sleep_min, sleep_max)
        if not soup:
            print("[ERROR] Could not load any seed page.", file=sys.stderr)
            sys.exit(3)
        current = read_header_md(soup)
        print(f"Seed header reads Matchweek {current}")

        # 2) Visit ONLY the planned MDs, in ascending order (minimal back clicks)
        for md in target_mds:
            print(f"\n[MD {md}] current header={current}")
            soup, current = navigate_to_md(driver, current, md, sleep_min, sleep_max)
            if current != md:
                print(f"  ! Could not land on MW{md} header (got {current}); parsing anyway…")

            df_md = scrape_current_page(soup, md)
            updated_df, merged_scores = merge_md_results(args.csv, md, df_md)

            md_slice = updated_df.loc[updated_df["md"].astype(int) == md]
            played = int(md_slice["played"].sum()) if not md_slice.empty else 0
            print(f"  - merged rows with scores: {merged_scores}")
            print(f"  - played now in MD {md}: {played} / {len(md_slice)}")

            snooze(sleep_min, sleep_max)

        final = pd.read_csv(args.csv)
        total_played = int(final["played"].sum())
        print(f"\n✔ Done. Total played: {total_played}/{len(final)}")
        per = final.loc[final["played"]].groupby("md").size()
        if not per.empty:
            print("Per-MD finished:")
            print(per.to_string())

        # Quick preview
        cols = ["date", "md", "home", "away", "FTHG", "FTAG", "FTR", "played", "status", "ResultStr"]
        cols = [c for c in cols if c in final.columns]
        print("\nPreview (first 12):")
        print(final[cols].head(12).to_string(index=False))

    finally:
        driver.quit()


if __name__ == "__main__":
    main()