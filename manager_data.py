#!/usr/bin/env python3
# manager_data.py — Premier League managers (official page only)
"""
Scrape current Premier League managers from the official page and save:
  pl_managers_2025_26.csv  with columns -> Team, Manager, StartDate, TenureDays

Usage (live):
  python manager_data.py --out pl_managers_2025_26.csv

Use a saved HTML instead of live fetch:
  python manager_data.py --html-file managers.html --out pl_managers_2025_26.csv

Requirements:
  pip install requests beautifulsoup4 pandas python-dateutil
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import time
from datetime import date
from typing import Dict, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

# Optional tolerant date parsing
try:
    from dateutil import parser as du_parser
except Exception:
    du_parser = None

PL_MANAGERS_URL = "https://www.premierleague.com/en/managers"

UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) PL-Manager-Scraper/1.2",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/125 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_7) Safari/605.1.15",
]

# Canonical mapping (keeps your pipeline names stable)
TEAM_ALIASES: Dict[str, str] = {
    "Bournemouth": "AFC Bournemouth",
    "Brighton": "Brighton and Hove Albion",
    "Man City": "Manchester City",
    "Man Utd": "Manchester United",
    "Newcastle": "Newcastle United",
    "Nott'm Forest": "Nottingham Forest",
    "Nott’ m Forest": "Nottingham Forest",
    "Nott’m Forest": "Nottingham Forest",
    "Spurs": "Tottenham Hotspur",
    "West Ham": "West Ham United",
    "Wolves": "Wolverhampton Wanderers",
    "Leeds": "Leeds United",
    # identities
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
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
    "Sunderland": "Sunderland",
    "Tottenham Hotspur": "Tottenham Hotspur",
    "West Ham United": "West Ham United",
    "Wolverhampton Wanderers": "Wolverhampton Wanderers",
}

MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

def canon_team(name: str) -> str:
    if not isinstance(name, str):
        return name
    s = re.sub(r"\s+", " ", name.strip())
    return TEAM_ALIASES.get(s, s)

def parse_start_date(text: str) -> Optional[date]:
    """Parse '22 Dec 2019', '3 June 2024', '5 July 2024', '1 Jul 2023', etc."""
    if not isinstance(text, str) or not text.strip():
        return None
    s = text.strip().replace("–", "-").replace("—", "-")

    if du_parser is not None:
        try:
            return du_parser.parse(s, dayfirst=True, fuzzy=True).date()
        except Exception:
            pass

    parts = re.split(r"\s+", s)
    if len(parts) >= 3:
        try:
            day = int(re.sub(r"[^\d]", "", parts[0]))
        except Exception:
            day = None
        mon_raw = re.sub(r"[^\w]", "", parts[1]).lower()
        month = MONTHS.get(mon_raw)
        m = re.search(r"\d{4}", s)
        year = int(m.group(0)) if m else None
        if all([day, month, year]):
            try:
                return date(year, month, day)
            except Exception:
                return None
    return None

def tenure_days(start: Optional[date]) -> Optional[int]:
    if not isinstance(start, date):
        return None
    return (date.today() - start).days

def new_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": random.choice(UA_POOL),
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
    })
    return s

def get_html(url: str, ses: requests.Session, sleep: float, timeout: int, retries: int) -> str:
    """Robust GET with polite delay, UA rotation, and backoff. Returns '' if all attempts fail."""
    last_err = None
    backoff = 1.25
    for attempt in range(1, retries + 1):
        try:
            time.sleep(max(0.0, sleep) + random.random() * 0.4)
            ses.headers["User-Agent"] = random.choice(UA_POOL)
            r = ses.get(url, timeout=timeout)
            r.raise_for_status()
            return r.text
        except requests.RequestException as e:
            last_err = e
            sys.stderr.write(f"[HTTP] {url} attempt {attempt}/{retries} failed: {e}\n")
            time.sleep(backoff + random.random() * 0.6)
            backoff *= 1.8
    sys.stderr.write(f"[ERROR] giving up on {url}: {last_err}\n")
    return ""

def parse_pl_managers(html: str) -> pd.DataFrame:
    """Extract Manager / Club / Start date table; return Team, Manager, StartDate, TenureDays."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("div.article__table table") or soup.find("table")
    if not table:
        raise RuntimeError("Managers table not found on the page.")

    rows = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if len(cells) < 3:
            continue

        # Skip header
        c0 = cells[0].get_text(" ", strip=True).lower()
        c1 = cells[1].get_text(" ", strip=True).lower()
        if "manager" in c0 and "club" in c1:
            continue

        manager = re.sub(r"\s+", " ", cells[0].get_text(" ", strip=True)).strip()
        club    = re.sub(r"\s+", " ", cells[1].get_text(" ", strip=True)).strip()
        start_s = re.sub(r"\s+", " ", cells[2].get_text(" ", strip=True)).strip()

        team = canon_team(club)
        start_dt = parse_start_date(start_s)
        rows.append({
            "Team": team,
            "Manager": manager,
            "StartDate": start_dt.isoformat() if start_dt else "",
            "TenureDays": tenure_days(start_dt) if start_dt else ""
        })

    if not rows:
        raise RuntimeError("No manager rows parsed from the PL page.")
    df = pd.DataFrame(rows).drop_duplicates(subset=["Team"]).sort_values("Team").reset_index(drop=True)
    return df

def main():
    ap = argparse.ArgumentParser(description="Premier League managers (official page only).")
    ap.add_argument("--out", default="pl_managers_2025_26.csv", help="Output CSV path.")
    ap.add_argument("--url", default=PL_MANAGERS_URL, help="Override the managers page URL.")
    ap.add_argument("--html-file", default=None, help="Use a local HTML file instead of HTTP.")
    ap.add_argument("--sleep", type=float, default=1.0, help="Base sleep between HTTP attempts (sec).")
    ap.add_argument("--timeout", type=int, default=45, help="Request timeout (sec).")
    ap.add_argument("--retries", type=int, default=5, help="HTTP retries.")
    args = ap.parse_args()

    ses = new_session()

    try:
        if args.html_file:
            html = open(args.html_file, "r", encoding="utf-8", errors="ignore").read()
        else:
            html = get_html(args.url, ses, sleep=args.sleep, timeout=args.timeout, retries=args.retries)
            if not html:
                raise RuntimeError("Failed to download managers page.")
        df = parse_pl_managers(html)
        df.to_csv(args.out, index=False, encoding="utf-8")
        print(f"✔ Wrote {args.out} ({len(df)} rows)")
        print(df.head(10).to_string(index=False))
    except Exception as e:
        print(f"[ERROR] {e}")
        # still create an empty CSV with the right headers
        pd.DataFrame(columns=["Team","Manager","StartDate","TenureDays"]).to_csv(args.out, index=False, encoding="utf-8")
        print(f"[WARN] Wrote empty {args.out}")

if __name__ == "__main__":
    main()
