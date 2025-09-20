# tm_jahrestabelle.py  — Transfermarkt "Annual Table" scraper (no Year column)
"""
Scrape Transfermarkt Premier League 'Annual Table' which stands for Jahrestabelle. 
Save to csv.

Usage (live):
  python tm_jahrestabelle.py --out tm_jahrestabelle.csv --year 2025

Use a saved HTML:
  python tm_jahrestabelle.py --html-file jahrestabelle_2025.html --out tm_jahrestabelle.csv

Output columns:
  Rank, Team, Matches, W, D, L, GF, GA, GD, Pts
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import time
from typing import Dict, Optional, List, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.transfermarkt.us/premier-league/jahrestabelle/wettbewerb/GB1"

UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) TM-Jahrestabelle/1.3",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/125.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Safari/605.1.15",
]

OUT_COLS = ["Rank","Team","Matches","W","D","L","GF","GA","GD","Pts"]

TEAM_ALIASES: Dict[str, str] = {
    # common short → canonical
    "Bournemouth": "AFC Bournemouth",
    "Brighton": "Brighton and Hove Albion",
    "Brighton & Hove Albion": "Brighton and Hove Albion",
    "Man City": "Manchester City",
    "Man Utd": "Manchester United",
    "Newcastle": "Newcastle United",
    "Forest": "Nottingham Forest",
    "Nottingham": "Nottingham Forest",
    "West Ham": "West Ham United",
    "Wolves": "Wolverhampton Wanderers",
    "Leeds": "Leeds United",
    "Tottenham": "Tottenham Hotspur",
    # identity (keep for clarity)
    "Arsenal": "Arsenal",
    "Arsenal FC": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Brentford": "Brentford",
    "Brentford FC": "Brentford",
    "Burnley": "Burnley",
    "Chelsea": "Chelsea",
    "Chelsea FC": "Chelsea",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Everton FC": "Everton",
    "Fulham": "Fulham",
    "Liverpool": "Liverpool",
    "Liverpool FC": "Liverpool",
    "Manchester City": "Manchester City",
    "Manchester United": "Manchester United",
    "Newcastle United": "Newcastle United",
    "Nottingham Forest": "Nottingham Forest",
    "Sunderland": "Sunderland",
    "West Ham United": "West Ham United",
    "Wolverhampton Wanderers": "Wolverhampton Wanderers",
    "Leicester": "Leicester City",
    "Leicester City": "Leicester City",
    "Southampton": "Southampton",
    "Ipswich": "Ipswich",
    "Ipswich Town": "Ipswich",
}

def _canon_team(name: str) -> str:
    if not isinstance(name, str):
        return name
    s = re.sub(r"\s+", " ", name.strip())
    s = {"Man City": "Manchester City"}.get(s, s)
    return TEAM_ALIASES.get(s, s)

def _session() -> requests.Session:
    ses = requests.Session()
    ses.headers.update({
        "User-Agent": random.choice(UA_POOL),
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Referer": "https://www.transfermarkt.us/",
    })
    return ses

def _get_html(url: str, ses: requests.Session, sleep_sec: float, timeout: int, retries: int, debug_path: Optional[str]) -> str:
    last_err = None
    backoff = 1.3
    for attempt in range(1, retries + 1):
        try:
            time.sleep(max(0.0, sleep_sec) + random.random() * 0.6)
            ses.headers["User-Agent"] = random.choice(UA_POOL)
            r = ses.get(url, timeout=timeout)
            r.raise_for_status()
            html = r.text
            if debug_path:
                try:
                    with open(debug_path, "w", encoding="utf-8", errors="ignore") as f:
                        f.write(html)
                except Exception:
                    pass
            return html
        except requests.RequestException as e:
            last_err = e
            sys.stderr.write(f"[HTTP] {url} attempt {attempt}/{retries} failed: {e}\n")
            time.sleep(backoff + random.random() * 0.8)
            backoff *= 1.7
    sys.stderr.write(f"[ERROR] giving up on {url}: {last_err}\n")
    return ""

def _split_goals(s: str) -> Tuple[Optional[int], Optional[int]]:
    if not isinstance(s, str):
        return (None, None)
    m = re.match(r"^\s*(\d+)\s*:\s*(\d+)\s*$", s.strip())
    if not m:
        return (None, None)
    try:
        return (int(m.group(1)), int(m.group(2)))
    except Exception:
        return (None, None)

def _extract_headers(th_row) -> List[str]:
    hdrs = []
    for th in th_row.find_all(["th","td"]):
        hdrs.append(th.get_text(" ", strip=True))
    return hdrs

def _find_target_table(soup: BeautifulSoup):
    """
    Choose the table whose headers contain 'Club', 'Matches', 'Pts'.
    """
    candidates = []
    candidates += soup.select("div.responsive-table div.grid-view table.items")
    candidates += soup.select("div.responsive-table table")
    candidates += soup.select("table")

    seen = set()
    for tbl in candidates:
        if id(tbl) in seen:
            continue
        seen.add(id(tbl))
        thead = tbl.find("thead")
        if not thead:
            continue
        hdr_row = thead.find("tr")
        if not hdr_row:
            continue
        hdrs = [h.lower() for h in _extract_headers(hdr_row)]
        if any("club" in h for h in hdrs) and any("match" in h for h in hdrs) and any(h in ("pts","points") for h in hdrs):
            return tbl, hdrs
    return None, None

def parse_jahrestabelle(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")

    tbl, hdrs = _find_target_table(soup)
    rows_out: List[dict] = []

    if tbl:
        body = tbl.find("tbody")
        if body:
            for tr in body.find_all("tr"):
                # Skip separator rows (e.g., "Change(s)...")
                if tr.get("class") and any("extrarow" in c for c in tr.get("class", [])):
                    continue
                tds = tr.find_all("td")
                if len(tds) < 6:
                    continue

                rank_txt = tds[0].get_text(strip=True)
                if not re.fullmatch(r"\d+", rank_txt or ""):
                    continue
                rank = int(rank_txt)

                # Team in 3rd <td> (name link)
                team_cell = tds[2]
                link = team_cell.select_one("a[title]") or team_cell.select_one("a")
                team = (link.get("title", "").strip() or link.get_text(" ", strip=True)) if link else team_cell.get_text(" ", strip=True)
                team = _canon_team(team)

                def _to_int(x: str) -> Optional[int]:
                    s = (x or "").strip()
                    return int(s) if re.fullmatch(r"-?\d+", s) else None

                # column order observed on TM:
                # 0=rank, 1=badge, 2=club name, 3=matches, 4=W, 5=D, 6=L, 7=Goals 'GF:GA', 8=+/- (GD), 9=Pts
                matches = _to_int(tds[3].get_text(strip=True)) if len(tds) > 3 else None
                W = _to_int(tds[4].get_text(strip=True)) if len(tds) > 4 else None
                D = _to_int(tds[5].get_text(strip=True)) if len(tds) > 5 else None
                L = _to_int(tds[6].get_text(strip=True)) if len(tds) > 6 else None
                GF, GA = _split_goals(tds[7].get_text(strip=True)) if len(tds) > 7 else (None, None)
                GD = _to_int(tds[8].get_text(strip=True)) if len(tds) > 8 else None
                Pts = _to_int(tds[9].get_text(strip=True)) if len(tds) > 9 else None

                rows_out.append({
                    "Rank": rank,
                    "Team": team,
                    "Matches": matches,
                    "W": W, "D": D, "L": L,
                    "GF": GF, "GA": GA, "GD": GD, "Pts": Pts,
                })

    # Fallback: pandas.read_html if DOM parse found nothing
    if not rows_out:
        try:
            all_tables = pd.read_html(html)
            chosen = None
            for df in all_tables:
                cols = [str(c).lower() for c in df.columns]
                if any("club" in c for c in cols) and any(("pts" in c) or ("points" in c) for c in cols) and any("match" in c for c in cols):
                    chosen = df
                    break
            if chosen is not None and not chosen.empty:
                df = chosen.copy()

                # try columns by heuristic
                def _pick_col(df, names):
                    for n in names:
                        if n in df.columns:
                            return n
                    for c in df.columns:
                        if any(n.lower() in str(c).lower() for n in names):
                            return c
                    return None

                rank_col = _pick_col(df, ["#", "No", "Rank"])
                club_col = _pick_col(df, ["Club"])
                matches_col = _pick_col(df, ["Matches", "Games", "Spiele"])
                w_col = _pick_col(df, ["W","Wins"])
                d_col = _pick_col(df, ["D","Draws"])
                l_col = _pick_col(df, ["L","Losses"])
                goals_col = _pick_col(df, ["Goals","Tore"])
                gd_col = _pick_col(df, ["+/-","GD","Diff"])
                pts_col = _pick_col(df, ["Pts","Points"])

                def _num_s(col):
                    if col and col in df.columns:
                        return pd.to_numeric(df[col], errors="coerce")
                    return pd.Series([None]*len(df))

                def _split_goals_series(s):
                    GF, GA = [], []
                    for x in s.astype(str):
                        g1, g2 = _split_goals(x)
                        GF.append(g1); GA.append(g2)
                    return pd.Series(GF), pd.Series(GA)

                Rank = _num_s(rank_col)
                Team = df[club_col].astype(str).map(_canon_team) if club_col else pd.Series([""]*len(df))
                Matches = _num_s(matches_col)
                W = _num_s(w_col); D = _num_s(d_col); L = _num_s(l_col)
                if goals_col and goals_col in df.columns:
                    GF, GA = _split_goals_series(df[goals_col])
                else:
                    GF, GA = pd.Series([None]*len(df)), pd.Series([None]*len(df))
                GD = _num_s(gd_col); Pts = _num_s(pts_col)

                parsed = pd.DataFrame({
                    "Rank": Rank, "Team": Team, "Matches": Matches,
                    "W": W, "D": D, "L": L, "GF": GF, "GA": GA, "GD": GD, "Pts": Pts
                }).dropna(subset=["Rank","Team"], how="any")
                parsed = parsed.sort_values("Rank").reset_index(drop=True)
                return parsed.reindex(columns=OUT_COLS)
        except Exception as e:
            sys.stderr.write(f"[WARN] pandas.read_html fallback failed: {e}\n")

    if not rows_out:
        return pd.DataFrame(columns=OUT_COLS)

    df = pd.DataFrame(rows_out)
    if "Rank" in df.columns and not df.empty:
        df = df.sort_values("Rank").reset_index(drop=True)
    return df.reindex(columns=OUT_COLS)

def main() -> None:
    ap = argparse.ArgumentParser(description="Scrape Transfermarkt PL Annual Table (Jahrestabelle).")
    ap.add_argument("--out", default="tm_jahrestabelle.csv", help="Output CSV path.")
    ap.add_argument("--year", type=int, default=None, help="Calendar year (e.g., 2025). Adds /jahr/<year> to URL.")
    ap.add_argument("--html-file", default=None, help="Parse a local HTML file instead of HTTP.")
    ap.add_argument("--sleep", type=float, default=1.3, help="Base sleep (sec) between HTTP tries.")
    ap.add_argument("--timeout", type=int, default=60, help="Request timeout (sec).")
    ap.add_argument("--retries", type=int, default=5, help="HTTP retries.")
    ap.add_argument("--debug-html", default=None, help="If set, save fetched HTML to this path for debugging.")
    args = ap.parse_args()

    ses = _session()
    url = f"{BASE_URL}/jahr/{args.year}" if args.year is not None else BASE_URL

    if args.html_file:
        html = open(args.html_file, "r", encoding="utf-8", errors="ignore").read()
    else:
        html = _get_html(url, ses, args.sleep, args.timeout, args.retries, args.debug_html)
        if not html:
            print("[ERROR] No HTML fetched; writing empty CSV.")
            pd.DataFrame(columns=OUT_COLS).to_csv(args.out, index=False)
            return

    df = parse_jahrestabelle(html)
    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"✔ Wrote {args.out} ({len(df)} rows)")
    if not df.empty:
        print(df.head(12).to_string(index=False))
    else:
        print("[INFO] No rows parsed. If you suspect a cookie/anti-bot wall, re-run with --debug-html then inspect the saved HTML.")

if __name__ == "__main__":
    main()
