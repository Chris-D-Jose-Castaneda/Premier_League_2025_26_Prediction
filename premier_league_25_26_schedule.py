#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
premier_league_25_26_schedule.py

End-to-end builder for the 2025/26 Premier League dataset.

Phase A (fixtures):
  - Loads the official PL article (Selenium) and parses all 380 fixtures
  - Robust Matchweek (md) assignment (uses headers when present; otherwise auto-chunks in 10s)
  - Repairs md using NBC’s "Matchweek X" sections if the PL article is inconsistent
  - Writes CSV with: date (YYYY-MM-DD), home, away, md

Phase B (results):
  - Scrapes the official /matches pages per matchweek (Selenium, client-rendered)
  - Uses a month hint derived from fixture dates (adds &month=MM)
  - Merges any finished results into the CSV by (md, home, away)
  - Adds columns (and only these): FTHG, FTAG, FTR, played, status, ResultStr
  - (No Date_actual column)

Usage:
  python premier_league_25_26_schedule.py --out fixtures_2025_26.csv
  # options:
  #   --results-from 1 --results-to 38
  #   --no-results
  #   --sleep-min 0.8 --sleep-max 2.2
  #   --no-headless  (watch the browser)

Requires:
  pip install selenium webdriver-manager beautifulsoup4 pandas requests
"""

from __future__ import annotations

import argparse
import calendar
import random
import re
import time
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

# --- Selenium setup -----------------------------------------------------------
from selenium import webdriver
from selenium.webdriver.firefox.options import Options as FFOptions
from selenium.webdriver.firefox.service import Service as FFService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC  # noqa: N813
from webdriver_manager.firefox import GeckoDriverManager


# -----------------------------------------------------------------------------#
# Constants
# -----------------------------------------------------------------------------#

PL_FIXTURES_URL = "https://www.premierleague.com/en/news/4324539/all-380-fixtures-for-202526-premier-league-season"
NBC_FIXTURES_URL = "https://www.nbcsports.com/soccer/news/premier-league-2025-26-fixtures-released-dates-schedule-how-to-watch-live"
PL_MATCHES_URL_TMPL = "https://www.premierleague.com/en/matches?competition=8&season=2025&matchweek={mw}"  # we'll append &month=MM

SEASON_START_YEAR = 2025
SEASON_END_YEAR = 2026

# Team normalization
TEAM_ALIASES: Dict[str, str] = {
    # short → canonical
    "Man United": "Manchester United",
    "Man Utd": "Manchester United",
    "Man City": "Manchester City",
    "Tottenham": "Tottenham Hotspur",
    "Spurs": "Tottenham Hotspur",
    "Nott'm Forest": "Nottingham Forest",
    "Nott'm": "Nottingham Forest",
    "Wolves": "Wolverhampton Wanderers",
    "West Ham": "West Ham United",
    "West Brom": "West Bromwich Albion",
    "Brighton": "Brighton and Hove Albion",
    "Brighton & Hove Albion": "Brighton and Hove Albion",
    "Newcastle": "Newcastle United",
    "Leeds": "Leeds United",
    "Leeds Utd": "Leeds United",
    "Bournemouth": "AFC Bournemouth",
    "Liverpool FC": "Liverpool",
    "Tottenham Hotspur Hotspur": "Tottenham Hotspur",
    # idempotent
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
WEB_ALIAS: Dict[str, str] = {
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


def month_to_year(month_name: str) -> int:
    m = month_name.lower()
    if m in {"august", "september", "october", "november", "december"}:
        return SEASON_START_YEAR
    return SEASON_END_YEAR


# -----------------------------------------------------------------------------#
# Selenium helpers
# -----------------------------------------------------------------------------#


def make_driver(headless: bool = True, timeout: int = 30) -> webdriver.Firefox:
    """Create a tuned Firefox driver."""
    opts = FFOptions()
    opts.headless = headless
    opts.set_preference("permissions.default.image", 2)  # faster: don't download images
    opts.set_preference("dom.ipc.processCount", 1)
    service = FFService(GeckoDriverManager().install())
    driver = webdriver.Firefox(service=service, options=opts)
    driver.set_page_load_timeout(timeout)
    return driver


def polite_sleep(smin: float, smax: float) -> None:
    """Randomized sleep to be polite."""
    time.sleep(random.uniform(max(0.0, smin), max(smin, smax)))


def fetch_html_with_selenium(
    url: str,
    *,
    headless: bool = True,
    timeout: int = 30,
    sleep_min: float = 0.8,
    sleep_max: float = 2.2,
    waiter: Optional[tuple] = None,
) -> str:
    """
    Generic Selenium fetcher. If `waiter` is provided, it must be a tuple:
      (by, selector) that will be waited for presence.
    """
    driver = make_driver(headless=headless, timeout=timeout)
    try:
        driver.get(url)
        polite_sleep(sleep_min, sleep_max)
        # Try cookie accept buttons (best-effort)
        for sel in ("button[aria-label*='Accept']", "button[aria-label*='accept']", "button[title*='Accept']"):
            try:
                WebDriverWait(driver, 4).until(EC.element_to_be_clickable((By.CSS_SELECTOR, sel))).click()
                break
            except Exception:
                pass
        if waiter:
            WebDriverWait(driver, timeout).until(EC.presence_of_element_located(waiter))
        html = driver.page_source
        return html
    finally:
        driver.quit()


# -----------------------------------------------------------------------------#
# Phase A — FIXTURES from PL article (+ NBC fallback)
# -----------------------------------------------------------------------------#

# Regexes for fixtures parsing
RE_MW_HEADER = re.compile(r"\bmatchweek\s*(\d+)\b", re.IGNORECASE)
RE_MW_PREFIX = re.compile(r"\bmw\s*(\d+)\b", re.IGNORECASE)
RE_DATE_LINE = re.compile(
    r"^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+(\d{1,2})\s+([A-Za-z]+)(?:\s+(\d{4}))?$",
    re.IGNORECASE,
)
RE_FIX = re.compile(r"^(?:\d{1,2}:\d{2}\s*(?:GMT)?\s*)?(.+?)\s+v\s+(.+?)(?:\s*\(.*\))?\s*$", re.IGNORECASE)
RE_FIX_NBC_V = re.compile(r"(?::\s*)?(.+?)\s+v\s+(.+?)\s*(?:—|$)", re.IGNORECASE)
RE_FIX_NBC_RES = re.compile(r"(.+?)\s+(\d+)\-(\d+)\s+(.+?)\s*(?:—|$)", re.IGNORECASE)


def _lines_from_p(p) -> list[str]:
    for br in p.find_all("br"):
        br.replace_with("\n")
    t = p.get_text("\n", strip=True)
    lines = []
    for ln in t.split("\n"):
        ln = re.sub(r"\s*\*.*$", "", ln).strip()
        if ln:
            lines.append(ln)
    return lines


def parse_pl_article_to_fixtures(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    container = soup.select_one("div.article__content, .js-article__content")
    if not container:
        raise RuntimeError("Could not locate article content container")

    rows = []
    cur_md, md_count, cur_date = None, 0, None
    for p in container.find_all("p"):
        for line in _lines_from_p(p):
            m1 = RE_MW_HEADER.search(line)
            m2 = RE_MW_PREFIX.search(line)
            if m1 or m2:
                cur_md = int((m1 or m2).group(1))
                md_count = 0
                # optional date on same line
                mdt = RE_DATE_LINE.search(line)
                if mdt:
                    d, mon, y = int(mdt.group(1)), mdt.group(2), mdt.group(3)
                    year = int(y) if y else month_to_year(mon)
                    cur_date = pd.Timestamp(year=year, month=pd.to_datetime(mon, format="%B").month, day=d)
                continue
            mdt = RE_DATE_LINE.match(line)
            if mdt:
                d, mon, y = int(mdt.group(1)), mdt.group(2), mdt.group(3)
                year = int(y) if y else month_to_year(mon)
                cur_date = pd.Timestamp(year=year, month=pd.to_datetime(mon, format="%B").month, day=d)
                continue
            fx = RE_FIX.match(line)
            if fx:
                if cur_date is None:
                    continue
                home = normalize_team(fx.group(1).strip())
                away = normalize_team(fx.group(2).strip())
                if cur_md is None:
                    cur_md, md_count = 1, 0
                if md_count == 10:
                    cur_md += 1
                    md_count = 0
                rows.append({"date": cur_date.date().isoformat(), "home": home, "away": away, "md": int(cur_md)})
                md_count += 1

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No fixtures parsed; the page layout may have changed.")
    df["md"] = df["md"].astype(int)
    return df.sort_values(["md", "date", "home", "away"]).reset_index(drop=True)


def parse_nbc_matchweeks(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    body = soup.select_one(".RichTextArticleBody, .RichTextBody")
    if not body:
        return pd.DataFrame(columns=["home", "away", "md"])

    out = []
    cur_md = None
    for el in body.find_all(["h3", "p"], recursive=True):
        if el.name == "h3":
            m = re.search(r"Matchweek\s*(\d+)", el.get_text(" ", strip=True), re.IGNORECASE)
            if m:
                cur_md = int(m.group(1))
            continue
        if el.name == "p" and cur_md is not None:
            txt = el.get_text(" ", strip=True)
            m_res = RE_FIX_NBC_RES.search(txt)
            if m_res:
                out.append(
                    {"home": normalize_team(m_res.group(1).strip()), "away": normalize_team(m_res.group(4).strip()), "md": cur_md}
                )
                continue
            m_v = RE_FIX_NBC_V.search(txt)
            if m_v:
                out.append(
                    {"home": normalize_team(m_v.group(1).strip()), "away": normalize_team(m_v.group(2).strip()), "md": cur_md}
                )
    return pd.DataFrame(out).drop_duplicates()


def repair_matchweeks_with_nbc(df_pl: pd.DataFrame, df_nbc: pd.DataFrame) -> pd.DataFrame:
    if df_nbc.empty:
        return df_pl
    m = df_pl.merge(df_nbc, on=["home", "away"], how="left", suffixes=("", "_nbc"))
    need = m["md"].isna() | ((m["md_nbc"].notna()) & (m["md"] != m["md_nbc"]))
    m.loc[need, "md"] = m.loc[need, "md_nbc"]
    m = m.drop(columns=["md_nbc"])
    if m["md"].isna().any():
        m = m.sort_values(["date", "home", "away"]).reset_index(drop=True)
        cur = int(m["md"].dropna().min()) if m["md"].notna().any() else 1
        cnt = 0
        for i in range(len(m)):
            if pd.isna(m.at[i, "md"]):
                if cnt == 10:
                    cur += 1
                    cnt = 0
                m.at[i, "md"] = cur
                cnt += 1
            else:
                cnt = 1 if (i > 0 and m.at[i, "md"] != m.at[i - 1, "md"]) else (cnt + 1 if i > 0 else 1)
    m["md"] = m["md"].astype(int)
    return m.sort_values(["md", "date", "home", "away"]).reset_index(drop=True)


def build_fixtures_from_sources(headless: bool, sleep_min: float, sleep_max: float) -> pd.DataFrame:
    # Fetch PL article
    html_pl = fetch_html_with_selenium(
        PL_FIXTURES_URL,
        headless=headless,
        timeout=30,
        sleep_min=sleep_min,
        sleep_max=sleep_max,
        waiter=(By.CSS_SELECTOR, "div.article__content, .js-article__content"),
    )
    df_pl = parse_pl_article_to_fixtures(html_pl)

    # Validate / repair with NBC if needed
    counts = df_pl.groupby("md").size().reindex(range(1, 39), fill_value=0)
    needs_repair = ((counts != 10).any()) or (len(df_pl) != 380)

    if needs_repair:
        try:
            r = requests.get(NBC_FIXTURES_URL, headers={"User-Agent": "EPL-RF/1.0"}, timeout=25)
            if r.ok:
                df_nbc = parse_nbc_matchweeks(r.text)
                df_pl = repair_matchweeks_with_nbc(df_pl, df_nbc)
        except Exception:
            pass

    # Final sanity print
    counts_final = df_pl.groupby("md").size().reindex(range(1, 39), fill_value=0)
    print("Per-matchweek counts (should all be 10):")
    print(counts_final.to_string())
    if (counts_final != 10).any():
        print("[WARN] Some matchweeks still not at 10; check sources or regexes.")
    if len(df_pl) != 380:
        print(f"[WARN] Total rows = {len(df_pl)} (expected 380).")

    return df_pl[["date", "home", "away", "md"]].reset_index(drop=True)


# -----------------------------------------------------------------------------#
# Phase B — RESULTS from PL /matches (Selenium + month hint)
# -----------------------------------------------------------------------------#

SCORE_RE = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")


def md_to_month_hint(fixtures: pd.DataFrame) -> Dict[int, int]:
    """Derive a month (1..12) hint for each matchweek from scheduled dates."""
    fx = fixtures.copy()
    fx["date"] = pd.to_datetime(fx["date"], errors="coerce")
    s = fx.dropna(subset=["date"]).groupby("md")["date"].min().dt.month
    return {int(md): int(mon) for md, mon in s.items() if pd.notna(mon)}


def matches_url_for_md(md: int, md_month: Dict[int, int]) -> str:
    base = PL_MATCHES_URL_TMPL.format(mw=md)
    mon = md_month.get(int(md))
    return f"{base}&month={mon:02d}" if mon else base


def scrape_results_for_md(
    md: int,
    *,
    headless: bool,
    sleep_min: float,
    sleep_max: float,
) -> pd.DataFrame:
    """
    Returns: columns [home, away, FTHG, FTAG, status, md]
    (No Date_actual in this project.)
    """
    # Load page with Selenium and wait for client-rendered list
    # We derive the URL including &month externally
    # (the caller provides a fully built URL)
    raise NotImplementedError  # replaced by fetch below


def fetch_results_range_with_selenium(
    md_from: int,
    md_to: int,
    fixtures: pd.DataFrame,
    *,
    headless: bool,
    sleep_min: float,
    sleep_max: float,
) -> pd.DataFrame:
    """Fetch finished results for matchweeks in [md_from, md_to]."""
    md_month = md_to_month_hint(fixtures)
    driver = make_driver(headless=headless, timeout=30)
    try:
        frames = []
        for md in range(md_from, md_to + 1):
            url = matches_url_for_md(md, md_month)
            driver.get(url)
            polite_sleep(sleep_min, sleep_max)
            # wait for client render
            try:
                WebDriverWait(driver, 30).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.match-list div[data-testid='dayContainer']"))
                )
            except Exception:
                continue

            soup = BeautifulSoup(driver.page_source, "html.parser")

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

                rows.append({"home": home, "away": away, "FTHG": FTHG, "FTAG": FTAG, "status": status, "md": int(md)})

            if rows:
                frames.append(pd.DataFrame(rows))

            # extra jitter between pages
            polite_sleep(sleep_min, sleep_max)

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    finally:
        driver.quit()


def merge_results_into_fixtures(csv_path: str, results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge results by (md, home, away) into the fixtures CSV.
    Adds/updates only: FTHG, FTAG, FTR, played, status, ResultStr
    """
    fx = pd.read_csv(csv_path).copy()

    # Drop any legacy Date_actual column if present (not used here)
    if "Date_actual" in fx.columns:
        fx = fx.drop(columns=["Date_actual"])

    # Ensure columns exist
    for col, default in [
        ("FTHG", pd.NA),
        ("FTAG", pd.NA),
        ("FTR", pd.NA),
        ("played", False),
        ("status", ""),
        ("ResultStr", ""),
    ]:
        if col not in fx.columns:
            fx[col] = default

    fx["home"] = fx["home"].map(normalize_team)
    fx["away"] = fx["away"].map(normalize_team)

    if results_df.empty:
        fx.to_csv(csv_path, index=False)
        return fx

    pl = results_df.copy()
    pl["home"] = pl["home"].map(normalize_team)
    pl["away"] = pl["away"].map(normalize_team)

    merged = fx.merge(
        pl.rename(columns={"FTHG": "FTHG_pl", "FTAG": "FTAG_pl", "status": "status_pl"}),
        on=["md", "home", "away"],
        how="left",
    )

    has_score = merged["FTHG_pl"].notna() & merged["FTAG_pl"].notna()
    merged.loc[has_score, "FTHG"] = merged.loc[has_score, "FTHG_pl"].astype(int)
    merged.loc[has_score, "FTAG"] = merged.loc[has_score, "FTAG_pl"].astype(int)

    merged.loc[merged["status_pl"].notna(), "status"] = merged.loc[merged["status_pl"].notna(), "status_pl"]

    def _ftr(row):
        if pd.isna(row["FTHG"]) or pd.isna(row["FTAG"]):
            return pd.NA
        return "H" if row["FTHG"] > row["FTAG"] else ("A" if row["FTAG"] > row["FTHG"] else "D")

    merged["FTR"] = merged.apply(_ftr, axis=1)
    merged["played"] = merged["FTR"].notna()
    merged["ResultStr"] = merged.apply(
        lambda r: f"{int(r['FTHG'])}-{int(r['FTAG'])} ({r['FTR']})" if r["played"] else "", axis=1
    )

    # Clean helper cols
    merged = merged.drop(columns=[c for c in ["FTHG_pl", "FTAG_pl", "status_pl"] if c in merged.columns])
    merged.to_csv(csv_path, index=False)
    return merged


# -----------------------------------------------------------------------------#
# CLI
# -----------------------------------------------------------------------------#

def main() -> None:
    ap = argparse.ArgumentParser(description="Build PL 2025/26 fixtures (with md) and merge official results.")
    ap.add_argument("--out", default="fixtures_2025_26.csv", help="Output CSV path.")
    ap.add_argument("--results-from", type=int, default=1, help="First MW to fetch results for (default 1).")
    ap.add_argument("--results-to", type=int, default=38, help="Last MW to fetch results for (default 38).")
    ap.add_argument("--no-results", action="store_true", help="Skip results aggregation.")
    ap.add_argument("--no-headless", action="store_true", help="Run Selenium with a visible browser.")
    ap.add_argument("--sleep-min", type=float, default=0.8, help="Min randomized sleep between page ops (sec).")
    ap.add_argument("--sleep-max", type=float, default=2.2, help="Max randomized sleep between page ops (sec).")
    args = ap.parse_args()

    headless = not args.no_headless
    sleep_min = max(0.0, args.sleep_min)
    sleep_max = max(sleep_min, args.sleep_max)

    # Phase A — Fixtures
    fixtures = build_fixtures_from_sources(headless=headless, sleep_min=sleep_min, sleep_max=sleep_max)
    fixtures.to_csv(args.out, index=False, encoding="utf-8")
    print(f"\n✔ Wrote {args.out} with {len(fixtures)} rows")

    counts = fixtures["md"].value_counts().sort_index()
    print("\nMatch count by MD (should be 10 each):")
    print(counts.to_string())

    # Phase B — Results
    if not args.no_results:
        print("\nFetching official /matches results (Selenium) …")
        results = fetch_results_range_with_selenium(
            max(1, args.results_from),
            min(38, args.results_to),
            fixtures=fixtures,
            headless=headless,
            sleep_min=sleep_min,
            sleep_max=sleep_max,
        )
        print(f"  - scraped {len(results)} rows from /matches")
        updated = merge_results_into_fixtures(args.out, results)
        played = int(updated["played"].sum())
        print(f"✔ Results merged: {played} matches marked played.")
        if played:
            print("\nFinished per MD:")
            print(updated.loc[updated["played"]].groupby("md").size().to_string())

    # Preview
    print("\nPreview (first 12):")
    cols = ["date", "md", "home", "away", "FTHG", "FTAG", "FTR", "played", "status", "ResultStr"]
    try:
        df_show = pd.read_csv(args.out)
        cols = [c for c in cols if c in df_show.columns]
        print(df_show[cols].head(12).to_string(index=False))
    except Exception:
        print(fixtures.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
